"""
真实仿真评估脚本 - AdaStep在Robomimic环境中的在线评估
=============================================================

学术严谨性说明:
- 本脚本用于获取真实的任务成功率数据（非估计值）
- 通过在MuJoCo物理引擎中实际运行机器人策略来验证AdaStep的有效性
- 这是将"离线推理节省率"转化为"在线任务成功率"的关键步骤

核心逻辑:
1. 从环境获取观测 (image + qpos)
2. AdaStep预测步长k和动作序列
3. **关键**: 连续执行k步动作，期间不进行推理（这是推理节省的来源）
4. k步后再次推理
5. 统计成功率和实际推理次数

使用方法:
    python run_real_simulation.py --task transport --ckpt policy_best.ckpt --num_rollouts 50

作者: AdaStep Research Team
日期: 2026-01-13
"""

import torch
import numpy as np
import h5py
import json
import argparse
import os
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt

# Robomimic 环境工具
try:
    import robomimic
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.file_utils as FileUtils
    ROBOMIMIC_AVAILABLE = True
except ImportError:
    print("⚠️  警告: 未安装robomimic，仿真功能不可用")
    print("   安装方法: pip install robomimic")
    ROBOMIMIC_AVAILABLE = False

# 导入 AdaStep 模块
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.adastep_module import AdaStepACTPolicy


class RealSimulationEvaluator:
    """
    真实仿真评估器
    
    功能:
    1. 加载训练好的AdaStep策略
    2. 在Robomimic环境中运行rollout
    3. 统计成功率、推理次数、执行效率
    """
    
    def __init__(self, 
                 checkpoint_path: str,
                 dataset_path: str,
                 device: str = 'cuda',
                 camera_names: list = None):
        """
        Args:
            checkpoint_path: 训练好的模型权重路径
            dataset_path: HDF5数据集路径（用于加载环境元数据）
            device: 'cuda' or 'cpu'
            camera_names: 相机列表，如 ['agentview_image', 'eye_in_hand_image']
        """
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.dataset_path = dataset_path
        self.camera_names = camera_names or ['agentview_image', 'eye_in_hand_image']
        
        # 加载环境元数据
        self._load_env_meta()
        
        # 加载策略模型
        self._load_policy()
        
        print(f"✓ 仿真评估器初始化完成")
        print(f"  - 任务: {self.env_meta['env_name']}")
        print(f"  - 相机: {self.camera_names}")
        print(f"  - 设备: {self.device}")
    
    def _load_env_meta(self):
        """从HDF5文件加载环境元数据"""
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"数据集不存在: {self.dataset_path}")
        
        with h5py.File(self.dataset_path, 'r') as f:
            # Robomimic标准格式
            env_args = json.loads(f["data"].attrs["env_args"])
            self.env_meta = env_args
            
        print(f"✓ 环境元数据加载完成: {self.env_meta['env_name']}")
    
    def _load_policy(self):
        """加载训练好的AdaStep策略"""
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"模型文件不存在: {self.checkpoint_path}")
        
        # 加载检查点
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # 从检查点中提取配置（如果有）
        if 'config' in checkpoint:
            policy_config = checkpoint['config']
        else:
            # 使用默认配置
            print("⚠️  检查点中未找到config，使用默认配置")
            policy_config = self._get_default_policy_config()
        
        # 初始化策略
        self.policy = AdaStepACTPolicy(policy_config)
        
        # 加载权重
        if 'model_state_dict' in checkpoint:
            self.policy.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.policy.load_state_dict(checkpoint)
        
        self.policy.to(self.device)
        self.policy.eval()
        
        print(f"✓ 模型加载完成: {self.checkpoint_path}")
    
    def _get_default_policy_config(self):
        """默认策略配置"""
        return {
            'lr': 1e-5,
            'num_queries': 100,
            'kl_weight': 10,
            'hidden_dim': 512,
            'dim_feedforward': 3200,
            'lr_backbone': 1e-5,
            'backbone': 'resnet18',
            'enc_layers': 4,
            'dec_layers': 7,
            'nheads': 8,
            'camera_names': self.camera_names,
            'state_dim': 14,  # 根据任务调整
            'use_adastep': True,
            'k_min': 5,
            'k_max': 50,
        }
    
    def _prepare_observation(self, obs):
        """
        将环境观测转换为模型输入格式
        
        Args:
            obs: Robomimic环境返回的观测字典
            
        Returns:
            qpos: [1, state_dim] - 机器人状态
            image: [1, C, H, W] - 拼接后的图像
        """
        # 1. 提取关节位置（qpos）
        # 不同任务的qpos格式可能不同，这里使用通用方法
        if 'robot0_eef_pos' in obs:
            # 末端执行器位置 + 姿态
            qpos_list = []
            qpos_list.append(obs['robot0_eef_pos'])  # (3,)
            if 'robot0_eef_quat' in obs:
                qpos_list.append(obs['robot0_eef_quat'])  # (4,)
            if 'robot0_gripper_qpos' in obs:
                qpos_list.append(obs['robot0_gripper_qpos'])  # (2,)
            qpos = np.concatenate(qpos_list)
        elif 'robot0_joint_pos' in obs:
            qpos = obs['robot0_joint_pos']
        else:
            raise ValueError("无法从观测中提取qpos")
        
        qpos = torch.from_numpy(qpos).float().unsqueeze(0).to(self.device)
        
        # 2. 提取并拼接图像
        curr_images = []
        for cam_name in self.camera_names:
            # Robomimic返回的图像格式: (H, W, C), uint8, [0, 255]
            img_key = cam_name if cam_name in obs else cam_name.replace('_image', '')
            
            if img_key not in obs:
                print(f"⚠️  警告: 观测中未找到相机 {cam_name}，可用的键: {obs.keys()}")
                # 使用第一个可用的图像
                img_key = [k for k in obs.keys() if 'image' in k][0]
                print(f"   使用替代相机: {img_key}")
            
            img = obs[img_key]
            
            # 归一化到 [0, 1]
            img = img.astype(np.float32) / 255.0
            
            # HWC -> CHW
            img = np.moveaxis(img, -1, 0)
            
            curr_images.append(img)
        
        # 拼接所有相机图像: [C1+C2, H, W]
        image = np.concatenate(curr_images, axis=0)
        image = torch.from_numpy(image).float().unsqueeze(0).to(self.device)
        
        return qpos, image
    
    def run_rollout(self, env, max_steps: int = 400, render: bool = False):
        """
        运行单条轨迹
        
        Args:
            env: Robomimic环境实例
            max_steps: 最大步数
            render: 是否渲染可视化
            
        Returns:
            result: 包含成功/失败、推理次数等信息的字典
        """
        obs = env.reset()
        
        # 统计数据
        inference_count = 0  # 推理次数
        env_step_count = 0   # 环境步数
        action_queue = []    # 动作队列
        success = False
        
        # 轨迹记录（用于调试）
        trajectory = {
            'observations': [],
            'actions': [],
            'k_predictions': [],
        }
        
        for step in range(max_steps):
            # ========== 核心逻辑开始 ==========
            
            # 如果动作队列为空，进行推理
            if len(action_queue) == 0:
                # 准备输入
                qpos, image = self._prepare_observation(obs)
                
                # AdaStep 推理
                with torch.no_grad():
                    # 调用策略的前向传播
                    # 注意: 这里需要根据你的策略实现调整
                    if hasattr(self.policy, 'predict_action'):
                        # 如果策略有专门的predict方法
                        k_pred, actions = self.policy.predict_action(qpos, image)
                    else:
                        # 使用标准的forward
                        # 注意: 这里需要根据你的实现调整参数
                        output = self.policy(qpos, image, actions=None, is_pad=None)
                        
                        # 从输出中提取k和动作
                        if 'k_pred' in output:
                            k_pred = output['k_pred'].item()
                        else:
                            k_pred = 1  # 默认值
                        
                        if 'actions' in output:
                            actions = output['actions']
                        else:
                            raise ValueError("策略输出中未找到actions")
                
                # 提取动作序列
                # actions shape: [1, num_queries, action_dim]
                actions = actions.squeeze(0).cpu().numpy()  # [num_queries, action_dim]
                
                # 截取前k步
                k_pred = int(k_pred)
                k_pred = max(1, min(k_pred, len(actions)))  # 限制在合理范围
                action_queue = list(actions[:k_pred])
                
                # 记录
                inference_count += 1
                trajectory['k_predictions'].append(k_pred)
            
            # ========== 核心逻辑结束 ==========
            
            # 从队列取动作执行
            action = action_queue.pop(0)
            
            # 执行动作
            obs, reward, done, info = env.step(action)
            env_step_count += 1
            
            # 记录轨迹
            trajectory['observations'].append(obs)
            trajectory['actions'].append(action)
            
            # 渲染（可选）
            if render:
                env.render(mode='human')
            
            # 检查成功
            if info.get("success", False):
                success = True
                break
            
            if done:
                break
        
        # 计算指标
        avg_k = env_step_count / max(inference_count, 1)
        inference_saving = (1 - inference_count / max(env_step_count, 1)) * 100
        
        result = {
            'success': success,
            'env_steps': env_step_count,
            'inference_count': inference_count,
            'avg_k': avg_k,
            'inference_saving': inference_saving,
            'trajectory': trajectory,
        }
        
        return result
    
    def evaluate(self, num_rollouts: int = 50, render: bool = False):
        """
        运行多条轨迹的评估
        
        Args:
            num_rollouts: 评估轨迹数
            render: 是否可视化
            
        Returns:
            results: 评估结果汇总
        """
        if not ROBOMIMIC_AVAILABLE:
            raise RuntimeError("Robomimic未安装，无法运行仿真")
        
        # 创建环境
        print(f"\n🚀 开始仿真评估...")
        print(f"  - 任务: {self.env_meta['env_name']}")
        print(f"  - 轨迹数: {num_rollouts}")
        
        env = EnvUtils.create_env_from_metadata(
            env_meta=self.env_meta,
            env_name=self.env_meta["env_name"],
            render=render,
            render_offscreen=False,
            use_image_obs=True,
        )
        
        # 运行评估
        results_list = []
        success_count = 0
        
        for i in tqdm(range(num_rollouts), desc="运行轨迹"):
            result = self.run_rollout(env, render=render)
            results_list.append(result)
            
            if result['success']:
                success_count += 1
        
        # 汇总统计
        success_rate = (success_count / num_rollouts) * 100
        avg_env_steps = np.mean([r['env_steps'] for r in results_list])
        avg_inference_count = np.mean([r['inference_count'] for r in results_list])
        avg_k = np.mean([r['avg_k'] for r in results_list])
        avg_inference_saving = np.mean([r['inference_saving'] for r in results_list])
        
        results = {
            'task': self.env_meta['env_name'],
            'num_rollouts': num_rollouts,
            'success_count': success_count,
            'success_rate': success_rate,
            'avg_env_steps': avg_env_steps,
            'avg_inference_count': avg_inference_count,
            'avg_k': avg_k,
            'avg_inference_saving': avg_inference_saving,
            'rollouts': results_list,
        }
        
        # 打印结果
        self._print_results(results)
        
        return results
    
    def _print_results(self, results):
        """打印评估结果"""
        print("\n" + "="*70)
        print("📊 仿真评估结果")
        print("="*70)
        print(f"任务: {results['task']}")
        print(f"评估轨迹数: {results['num_rollouts']}")
        print("-"*70)
        print(f"✅ 成功率: {results['success_rate']:.2f}% ({results['success_count']}/{results['num_rollouts']})")
        print(f"📈 平均环境步数: {results['avg_env_steps']:.1f}")
        print(f"🧠 平均推理次数: {results['avg_inference_count']:.1f}")
        print(f"⚡ 平均步长 k: {results['avg_k']:.1f}")
        print(f"💡 推理节省率: {results['avg_inference_saving']:.2f}%")
        print("="*70 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description='AdaStep真实仿真评估')
    
    parser.add_argument('--task', type=str, required=True,
                       help='任务名称: square/lift/can/transport')
    parser.add_argument('--ckpt', type=str, required=True,
                       help='模型检查点路径')
    parser.add_argument('--data', type=str, default=None,
                       help='HDF5数据集路径（用于加载环境元数据）')
    parser.add_argument('--num_rollouts', type=int, default=50,
                       help='评估轨迹数')
    parser.add_argument('--render', action='store_true',
                       help='是否可视化渲染')
    parser.add_argument('--device', type=str, default='cuda',
                       help='计算设备: cuda/cpu')
    parser.add_argument('--save_results', type=str, default=None,
                       help='结果保存路径')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 如果没有指定数据路径，使用默认路径
    if args.data is None:
        args.data = f"../robomimic_data/{args.task}/mh/low_dim_v141.hdf5"
    
    # 检查文件存在
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"模型文件不存在: {args.ckpt}")
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"数据文件不存在: {args.data}")
    
    # 初始化评估器
    evaluator = RealSimulationEvaluator(
        checkpoint_path=args.ckpt,
        dataset_path=args.data,
        device=args.device,
    )
    
    # 运行评估
    results = evaluator.evaluate(
        num_rollouts=args.num_rollouts,
        render=args.render,
    )
    
    # 保存结果
    if args.save_results:
        import pickle
        with open(args.save_results, 'wb') as f:
            pickle.dump(results, f)
        print(f"✓ 结果已保存到: {args.save_results}")


if __name__ == "__main__":
    main()
