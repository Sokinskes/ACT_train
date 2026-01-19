"""
仿真环境诊断脚本
==================

在运行真实仿真之前，使用此脚本检查:
1. Robomimic 是否正确安装
2. MuJoCo 是否正确配置
3. 数据集格式是否正确
4. 模型是否能正确加载
5. 环境元数据是否完整

使用方法:
    python diagnostic_simulation.py --task transport
"""

import torch
import numpy as np
import h5py
import json
import os
import sys
import argparse


def check_robomimic():
    """检查Robomimic安装"""
    print("\n" + "="*70)
    print("1️⃣  检查 Robomimic 安装")
    print("="*70)
    
    try:
        import robomimic
        import robomimic.utils.env_utils as EnvUtils
        print(f"✅ Robomimic 已安装")
        print(f"   版本: {robomimic.__version__ if hasattr(robomimic, '__version__') else '未知'}")
        return True
    except ImportError as e:
        print(f"❌ Robomimic 未安装")
        print(f"   错误: {e}")
        print(f"\n   安装方法:")
        print(f"   pip install robomimic")
        return False


def check_mujoco():
    """检查MuJoCo"""
    print("\n" + "="*70)
    print("2️⃣  检查 MuJoCo")
    print("="*70)
    
    try:
        import mujoco_py
        print(f"✅ MuJoCo-py 已安装")
        return True
    except ImportError:
        try:
            import mujoco
            print(f"✅ MuJoCo (新版) 已安装")
            return True
        except ImportError as e:
            print(f"❌ MuJoCo 未安装")
            print(f"   错误: {e}")
            return False


def check_dataset(data_path):
    """检查数据集格式"""
    print("\n" + "="*70)
    print("3️⃣  检查数据集")
    print("="*70)
    
    if not os.path.exists(data_path):
        print(f"❌ 数据集文件不存在: {data_path}")
        return False
    
    print(f"✅ 文件存在: {data_path}")
    print(f"   大小: {os.path.getsize(data_path) / 1024 / 1024:.2f} MB")
    
    try:
        with h5py.File(data_path, 'r') as f:
            print(f"\n📦 HDF5 结构:")
            
            # 检查data组
            if 'data' not in f:
                print(f"❌ 缺少 'data' 组")
                return False
            
            print(f"✅ 'data' 组存在")
            
            # 检查环境元数据
            if 'env_args' not in f['data'].attrs:
                print(f"❌ 缺少 'env_args' 属性")
                return False
            
            env_args = json.loads(f['data'].attrs['env_args'])
            print(f"\n🌍 环境信息:")
            print(f"   任务: {env_args.get('env_name', '未知')}")
            print(f"   类型: {env_args.get('type', '未知')}")
            
            # 检查演示数据
            if 'demo_0' not in f['data']:
                print(f"❌ 缺少演示数据")
                return False
            
            demo = f['data/demo_0']
            print(f"\n📊 演示数据 (demo_0):")
            print(f"   观测键: {list(demo['obs'].keys())}")
            print(f"   动作形状: {demo['actions'].shape}")
            print(f"   轨迹长度: {len(demo['actions'])}")
            
            # 检查图像数据
            obs_keys = list(demo['obs'].keys())
            image_keys = [k for k in obs_keys if 'image' in k]
            if image_keys:
                print(f"\n📷 图像数据:")
                for key in image_keys:
                    img_shape = demo['obs'][key].shape
                    print(f"   {key}: {img_shape}")
            else:
                print(f"⚠️  未找到图像数据（可能是low_dim数据集）")
            
            return True
            
    except Exception as e:
        print(f"❌ 读取数据集失败:")
        print(f"   错误: {e}")
        return False


def check_model(ckpt_path):
    """检查模型文件"""
    print("\n" + "="*70)
    print("4️⃣  检查模型文件")
    print("="*70)
    
    if not os.path.exists(ckpt_path):
        print(f"❌ 模型文件不存在: {ckpt_path}")
        return False
    
    print(f"✅ 文件存在: {ckpt_path}")
    print(f"   大小: {os.path.getsize(ckpt_path) / 1024 / 1024:.2f} MB")
    
    try:
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        
        print(f"\n🧠 模型信息:")
        print(f"   检查点键: {list(checkpoint.keys())}")
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f"   模型参数数: {len(state_dict)}")
            
            # 检查AdaStep相关参数
            adastep_keys = [k for k in state_dict.keys() if 'horizon_predictor' in k]
            if adastep_keys:
                print(f"   ✅ 包含 AdaStep 参数: {len(adastep_keys)} 个")
            else:
                print(f"   ⚠️  未找到 AdaStep 参数")
        
        if 'config' in checkpoint:
            config = checkpoint['config']
            print(f"\n⚙️  训练配置:")
            print(f"   use_adastep: {config.get('use_adastep', False)}")
            print(f"   k_min: {config.get('k_min', '未知')}")
            print(f"   k_max: {config.get('k_max', '未知')}")
        
        if 'epoch' in checkpoint:
            print(f"\n📈 训练状态:")
            print(f"   Epoch: {checkpoint['epoch']}")
        
        return True
        
    except Exception as e:
        print(f"❌ 加载模型失败:")
        print(f"   错误: {e}")
        return False


def test_environment_creation(data_path):
    """测试环境创建"""
    print("\n" + "="*70)
    print("5️⃣  测试环境创建")
    print("="*70)
    
    try:
        import robomimic.utils.env_utils as EnvUtils
        
        # 读取环境元数据
        with h5py.File(data_path, 'r') as f:
            env_meta = json.loads(f['data'].attrs['env_args'])
        
        print(f"正在创建环境: {env_meta['env_name']}...")
        
        # 创建环境
        env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta,
            env_name=env_meta["env_name"],
            render=False,
            render_offscreen=False,
            use_image_obs=True,
        )
        
        print(f"✅ 环境创建成功")
        
        # 测试reset
        obs = env.reset()
        print(f"\n🔄 环境重置成功")
        print(f"   观测键: {list(obs.keys())}")
        
        # 测试step
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(f"\n▶️  环境步进成功")
        print(f"   动作空间: {env.action_space}")
        print(f"   奖励: {reward}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ 环境创建失败:")
        print(f"   错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='仿真环境诊断')
    parser.add_argument('--task', type=str, default='transport',
                       help='任务名称')
    parser.add_argument('--data', type=str, default=None,
                       help='数据集路径')
    parser.add_argument('--ckpt', type=str, default=None,
                       help='模型路径')
    args = parser.parse_args()
    
    # 默认路径
    if args.data is None:
        args.data = f"../robomimic_data/{args.task}/mh/low_dim_v141.hdf5"
    if args.ckpt is None:
        args.ckpt = f"../checkpoints/{args.task}_mh/policy_best.ckpt"
    
    print("\n" + "="*70)
    print("🔍 AdaStep 仿真环境诊断")
    print("="*70)
    print(f"任务: {args.task}")
    print(f"数据: {args.data}")
    print(f"模型: {args.ckpt}")
    
    # 运行检查
    results = []
    
    results.append(("Robomimic", check_robomimic()))
    results.append(("MuJoCo", check_mujoco()))
    results.append(("数据集", check_dataset(args.data)))
    results.append(("模型", check_model(args.ckpt)))
    results.append(("环境", test_environment_creation(args.data)))
    
    # 总结
    print("\n" + "="*70)
    print("📋 诊断总结")
    print("="*70)
    
    all_passed = True
    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {name:15s}: {status}")
        if not passed:
            all_passed = False
    
    print("="*70)
    
    if all_passed:
        print("\n🎉 所有检查通过！可以运行真实仿真。")
        print("\n下一步:")
        print("  python run_real_simulation.py --task transport --ckpt <ckpt_path> --num_rollouts 50")
    else:
        print("\n⚠️  部分检查未通过，请先解决上述问题。")
    
    print()


if __name__ == "__main__":
    main()
