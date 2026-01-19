"""
AdaStep状态级自适应仿真验证
============================

目标: 验证新算法是否能在Transport任务中正确实现状态级自适应

关键验证点:
1. k值是否在"抓取瞬间"自动降低 (保守模式)
2. k值是否在"搬运瞬间"自动升高 (激进模式)
3. 整体成功率是否维持

方法:
- 使用训练好的HorizonPredictor
- 在Robomimic Transport环境中运行
- 记录每个时间步的k值变化和任务状态
"""

import torch
import numpy as np
import os
import sys
import json
import matplotlib.pyplot as plt
from pathlib import Path
import h5py

# Robomimic imports
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.file_utils as FileUtils

# 添加路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.adastep_module import HorizonPredictor

# 设置matplotlib
import matplotlib
matplotlib.use('Agg')

class TransportStateAnalyzer:
    """
    Transport任务状态分析器

    分析当前状态属于哪个阶段:
    - Reaching: 接近物体阶段 (应该用较大k)
    - Grasping: 抓取阶段 (应该用较小k)
    - Transporting: 搬运阶段 (应该用较大k)
    """

    def __init__(self):
        # Transport任务的关键位置 (基于经验值)
        self.object_init_pos = np.array([0.0, 0.0, 0.0])  # 物体初始位置
        self.goal_pos = np.array([0.3, 0.0, 0.0])         # 目标位置

    def analyze_state(self, obs):
        """
        分析当前状态的任务阶段

        Args:
            obs: Robomimic观测字典

        Returns:
            phase: 'reaching', 'grasping', 'transporting'
            confidence: 置信度 (0-1)
        """
        # 提取关键信息
        eef_pos = obs['robot0_eef_pos']
        eef_quat = obs['robot0_eef_quat']

        # 物体位置 (如果可用) — 支持不同数据布局 (长度>3 时取前3维)
        if 'object' in obs:
            object_pos = np.asarray(obs['object'])
            if object_pos.ndim > 1:
                # 有时为 (N, D) 的数组，取第一行
                object_pos = object_pos.reshape(-1)[:3]
            elif object_pos.size > 3:
                object_pos = object_pos.reshape(-1)[:3]
        else:
            # 使用默认位置
            object_pos = self.object_init_pos

        # 计算距离（防止维度不匹配）
        try:
            eef_to_object = np.linalg.norm(eef_pos - object_pos)
        except Exception:
            eef_to_object = np.linalg.norm(np.asarray(eef_pos).reshape(-1)[:3] - np.asarray(object_pos).reshape(-1)[:3])
        object_to_goal = np.linalg.norm(np.asarray(object_pos).reshape(-1)[:3] - self.goal_pos)

        # 简单的启发式规则
        if eef_to_object < 0.05:  # 接近物体
            return 'grasping', 0.8
        elif object_to_goal < 0.1:  # 物体接近目标
            return 'transporting', 0.7
        else:
            return 'reaching', 0.6


def run_transport_validation(hdf5_path, predictor_path, num_episodes=10, device='cuda', render_offscreen=False):
    """
    运行Transport任务验证

    Args:
        hdf5_path: 数据文件路径
        predictor_path: 预测器模型路径
        num_episodes: 测试episode数量
        device: 计算设备
        render_offscreen: 如果为 True，则在失败的 episode 中保存若干帧用于诊断

    Returns:
        results: 验证结果字典
    """

    print(f"\n{'='*80}")
    print(f"🚀 AdaStep状态级自适应仿真验证 - Transport任务")
    print(f"{'='*80}")

    # 用于保存失败帧的文件夹
    diagnostic_dir = Path('simulation_validation_results') / 'diagnostic_frames'
    if render_offscreen:
        diagnostic_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载环境
    print("🌍 创建Transport仿真环境...")
    env_meta = FileUtils.get_env_metadata_from_dataset(hdf5_path)
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=False,
        use_image_obs=False
    )
    print(f"✓ 环境: {env.name}")

    # probe environment for action dimensionality (robust to wrappers)
    action_dim = None
    try:
        action_dim = int(getattr(env, 'action_dim', None) or getattr(env, 'action_size', None))
    except Exception:
        action_dim = None
    if action_dim is None:
        try:
            action_dim = env.action_space.shape[0]
        except Exception:
            action_dim = None
    # fallback: infer from dataset demo actions
    if action_dim is None:
        try:
            import h5py
            with h5py.File(hdf5_path, 'r') as f:
                first_demo = next(iter(f['data'].keys()))
                demo_grp = f[f'data/' + first_demo]
                if 'actions' in demo_grp:
                    action_dim = demo_grp['actions'].shape[1]
        except Exception:
            action_dim = None
    print(f"Detected action_dim={action_dim}")

    # Ensure ObsUtils is initialized (some robomimic/robosuite builds expect this before reset)
    try:
        import robomimic.utils.obs_utils as ObsUtils
        if getattr(ObsUtils, 'OBS_KEYS_TO_MODALITIES', None) is None:
            # derive keys from dataset and initialize (list-of-dict accepted)
            import h5py
            with h5py.File(hdf5_path, 'r') as f:
                first_demo = next(iter(f['data'].keys()))
                demo_grp = f[f'data/' + first_demo]
                obs_keys = list(demo_grp['obs'].keys()) if 'obs' in demo_grp else []
            mapping = {'low_dim': [], 'rgb': []}
            for k in obs_keys:
                if 'rgb' in k or 'image' in k:
                    mapping['rgb'].append(k)
                else:
                    mapping['low_dim'].append(k)
            ObsUtils.initialize_obs_utils_with_obs_specs([mapping])
            print('Initialized ObsUtils from dataset (keys:', len(obs_keys), ')')
    except Exception:
        # non-fatal — env may still work with default obs setup
        pass

    # 2. 加载HorizonPredictor
    print(f"🧠 加载HorizonPredictor: {predictor_path}")

    # 获取状态维度 (使用末端执行器状态)
    state_dim = 7  # eef_pos (3) + eef_quat (4)

    horizon_predictor = HorizonPredictor(
        input_dim=state_dim,
        hidden_dim=256
    ).to(device)

    if os.path.exists(predictor_path):
        checkpoint = torch.load(predictor_path, map_location=device)
        horizon_predictor.load_state_dict(checkpoint)
        horizon_predictor.eval()
        print("✓ 模型加载成功")
    else:
        print("❌ 模型文件不存在")
        return None

    # 3. 创建状态分析器
    state_analyzer = TransportStateAnalyzer()

    # 4. 运行验证
    print(f"\n🎯 运行验证 ({num_episodes} episodes)...")

    all_results = []

    for ep in range(num_episodes):
        print(f"\n--- Episode {ep+1}/{num_episodes} ---")

        # 重置环境
        obs = env.reset()
        done = False
        step = 0
        max_steps = 400

        # 记录数据
        episode_data = {
            'k_values': [],
            'eef_positions': [],
            'task_phases': [],
            'success': False,
            'total_steps': 0
        }

        while step < max_steps and not done:
            # 提取状态特征 (末端执行器位姿)
            eef_pos = obs['robot0_eef_pos']
            eef_quat = obs['robot0_eef_quat']
            state_feature = np.concatenate([eef_pos, eef_quat])

            # 转换为tensor
            state_tensor = torch.from_numpy(state_feature).float().to(device).unsqueeze(0)

            # 预测k值
            with torch.no_grad():
                k_pred = horizon_predictor(state_tensor)
                k_normalized = torch.sigmoid(k_pred).item()
                k_value = int(5 + k_normalized * (50 - 5))  # 映射到[5,50]

            # 分析任务阶段
            phase, confidence = state_analyzer.analyze_state(obs)

            # 记录数据
            episode_data['k_values'].append(k_value)
            episode_data['eef_positions'].append(eef_pos.copy())
            episode_data['task_phases'].append(phase)

            # DEBUG: show action_dim value just before action computation
            try:
                print(f"[run] action_dim (loop) = {action_dim} (type={type(action_dim)})")
                print(f"[run] compute_transport_action fn = {compute_transport_action}")
            except Exception:
                pass

            # 执行动作 (使用专家动作或简单策略)
            # 这里使用简单的启发式策略来完成transport任务
            action = compute_transport_action(obs, phase, action_dim=action_dim)

            # 执行动作并（可选）保存诊断帧
            obs, reward, done, info = env.step(action)

            if render_offscreen and (not episode_data['success']):
                try:
                    # robosuite 的 render(mode='offscreen') 返回 RGB 图像
                    frame = env.render(mode='offscreen')
                    if frame is not None and len(frame.shape) == 3:
                        if len(episode_data.get('diagnostic_frames', [])) < 200:
                            episode_data.setdefault('diagnostic_frames', []).append(frame)
                except Exception:
                    pass

            step += 1

            # 检查成功
            if info.get('success', False):
                episode_data['success'] = True
                break

        episode_data['total_steps'] = step

        # save diagnostic frames (if requested and available)
        if render_offscreen and episode_data.get('diagnostic_frames'):
            frames = episode_data['diagnostic_frames']
            # save up to 10 evenly spaced frames for debugging
            nsave = min(10, len(frames))
            idxs = np.linspace(0, len(frames)-1, nsave).astype(int)
            for i, ix in enumerate(idxs):
                frm = frames[ix]
                outp = diagnostic_dir / f'ep{ep+1:02d}_frame{i+1:02d}.png'
                try:
                    import imageio
                    imageio.imwrite(str(outp), frm)
                except Exception:
                    pass

        all_results.append(episode_data)

        print(f"  结果: {'✅ 成功' if episode_data['success'] else '❌ 失败'} "
              f"({step}步, 平均k={np.mean(episode_data['k_values']):.1f})")

    # some env wrappers do not implement close(); be defensive
    if hasattr(env, 'close'):
        env.close()
    elif hasattr(env, 'shutdown'):
        env.shutdown()
    else:
        try:
            env.viewer = None
        except Exception:
            pass

    # 5. 分析结果
    print(f"\n{'='*80}")
    print("📊 验证结果分析")
    print(f"{'='*80}")

    # 整体统计
    success_rate = np.mean([r['success'] for r in all_results]) * 100
    avg_k_values = [np.mean(r['k_values']) for r in all_results]
    avg_steps = [r['total_steps'] for r in all_results]

    print(f"整体统计:")
    print(f"  成功率: {success_rate:.1f}%")
    print(f"  平均k值: {np.mean(avg_k_values):.1f} ± {np.std(avg_k_values):.1f}")
    print(f"  平均步数: {np.mean(avg_steps):.1f} ± {np.std(avg_steps):.1f}")

    # 阶段分析
    print(f"\n阶段k值分析:")
    phase_k_stats = analyze_phase_k_values(all_results)
    for phase, stats in phase_k_stats.items():
        print(f"  {phase.capitalize()}: k={stats['mean']:.1f} ± {stats['std']:.1f} "
              f"({stats['count']}个样本)")

    # 验证状态级自适应
    validation_results = validate_state_adaptation(all_results, phase_k_stats)

    # 6. 生成可视化
    output_dir = Path("simulation_validation_results")
    output_dir.mkdir(exist_ok=True)

    create_validation_plots(all_results, phase_k_stats, output_dir)

    # 7. 保存结果
    results_summary = {
        'task': 'transport',
        'success_rate': success_rate,
        'avg_k': np.mean(avg_k_values),
        'phase_analysis': phase_k_stats,
        'validation': validation_results,
        'episodes': len(all_results)
    }

    output_file = output_dir / "transport_validation_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)

    print(f"\n✓ 结果已保存: {output_file}")
    print(f"✓ 可视化图表已生成: {output_dir}")

    return results_summary


def compute_transport_action(obs, phase, env=None, action_dim=None):
    """Compute a heuristic action that supports single- and two-arm envs.

    - For single-arm envs the action is [dx,dy,dz, gripper].
    - For multi-arm envs we populate the primary arm (robot0) and pad/mirror
      the secondary arm entries to match env.action_dim.

    This heuristic is intentionally simple (PID-like) — it is only used for
    closed-loop sanity checks and not for final evaluations.
    """
    eef_pos = np.asarray(obs.get('robot0_eef_pos', np.zeros(3))).reshape(-1)[:3]

    if phase == 'reaching':
        target_pos = np.array([0.0, 0.0, 0.05])  # above object
        pos_error = target_pos - eef_pos
        action_pos = pos_error * 2.0
        action_gripper = np.array([1.0])

    elif phase == 'grasping':
        target_pos = np.array([0.0, 0.0, 0.02])
        pos_error = target_pos - eef_pos
        action_pos = pos_error * 3.0
        action_gripper = np.array([-1.0])

    else:  # transporting
        target_pos = np.array([0.3, 0.0, 0.05])
        pos_error = target_pos - eef_pos
        action_pos = pos_error * 2.0
        action_gripper = np.array([-1.0])

    # single-arm base action
    base_action = np.concatenate([action_pos, action_gripper])
    base_action = np.clip(base_action, -1.0, 1.0)

    # determine required action dimensionality (respect a caller-provided value)
    if action_dim is None and env is not None:
        try:
            action_dim = int(getattr(env, 'action_dim', env.action_space.shape[0]))
        except Exception:
            action_dim = None

    # debug/info: log detected action_dim and base size (helps diagnose mismatches)
    try:
        _ad = action_dim if action_dim is not None else 'None'
        print(f"[compute_transport_action] detected action_dim={_ad}, base_size={base_action.size}")
    except Exception:
        pass

    # if unknown or matches base size, return single-arm action
    if action_dim is None or action_dim == base_action.size:
        return base_action

    # otherwise expand for multi-arm: assume arms are concatenated
    # strategy: place base_action into robot0 slot and zero-fill remaining dims
    expanded = np.zeros(action_dim, dtype=float)
    expanded[:base_action.size] = base_action

    # if action_dim matches two-arm common layouts (e.g. 14), try mirroring gripper
    if action_dim >= 8:
        # copy position control to second arm's position slots if available
        # heuristic: copy first 3 pos deltas to slots [7:10] (approximate)
        if action_dim >= 10:
            expanded[7:10] = expanded[0:3]
        # set secondary gripper to same command if slot exists near the end
        if action_dim >= 14:
            expanded[13] = expanded[3]

    print(f"[compute_transport_action] returning expanded action (len={len(expanded)})")
    return expanded



def analyze_phase_k_values(all_results):
    """
    分析不同阶段的k值统计

    Returns:
        phase_stats: {phase: {'mean': float, 'std': float, 'count': int}}
    """
    phase_k_values = {'reaching': [], 'grasping': [], 'transporting': []}

    for episode in all_results:
        for k_val, phase in zip(episode['k_values'], episode['task_phases']):
            phase_k_values[phase].append(k_val)

    phase_stats = {}
    for phase, k_vals in phase_k_values.items():
        if k_vals:
            phase_stats[phase] = {
                'mean': np.mean(k_vals),
                'std': np.std(k_vals),
                'count': len(k_vals)
            }
        else:
            phase_stats[phase] = {'mean': 0, 'std': 0, 'count': 0}

    return phase_stats


def validate_state_adaptation(all_results, phase_stats):
    """
    验证状态级自适应是否正确工作

    检查:
    1. Grasping阶段的k值是否显著低于其他阶段
    2. k值变化是否合理
    """
    validation = {}

    # 1. 阶段间k值差异
    grasping_k = phase_stats['grasping']['mean']
    reaching_k = phase_stats['reaching']['mean']
    transporting_k = phase_stats['transporting']['mean']

    # Grasping应该有最低的k值
    validation['grasping_has_lowest_k'] = (
        grasping_k < reaching_k and grasping_k < transporting_k
    )

    # k值差异显著性 (至少相差5)
    validation['significant_k_difference'] = (
        min(reaching_k, transporting_k) - grasping_k >= 5
    )

    # 2. 整体k值分布
    all_k_values = []
    for episode in all_results:
        all_k_values.extend(episode['k_values'])

    k_std = np.std(all_k_values)
    validation['sufficient_k_variability'] = k_std >= 5  # 标准差至少5

    unique_k = len(set(all_k_values))
    validation['multiple_k_values'] = unique_k >= 3  # 至少3种不同k值

    print(f"\n🔍 状态级自适应验证:")
    print(f"  ✅ Grasping阶段k值最低: {validation['grasping_has_lowest_k']}")
    print(f"  ✅ k值差异显著: {validation['significant_k_difference']}")
    print(f"  ✅ k值变化充分: {validation['sufficient_k_variability']}")
    print(f"  ✅ 使用多种k值: {validation['multiple_k_values']}")

    # 计算自适应得分
    adaptation_score = sum(validation.values()) / len(validation) * 100
    validation['adaptation_score'] = adaptation_score
    print(f"  🎯 自适应得分: {adaptation_score:.1f}%")

    return validation


def create_validation_plots(all_results, phase_stats, output_dir):
    """
    生成验证结果的可视化图表
    """
    # 1. k值时序变化图
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 子图1: 多个episode的k值变化
    ax1 = axes[0, 0]
    for i, episode in enumerate(all_results[:5]):  # 只显示前5个
        k_values = episode['k_values']
        steps = range(len(k_values))
        ax1.plot(steps, k_values, label=f'Episode {i+1}', alpha=0.7)

    ax1.set_title('k值时序变化 (前5个Episodes)')
    ax1.set_xlabel('时间步')
    ax1.set_ylabel('预测步长 k')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 子图2: 阶段k值分布
    ax2 = axes[0, 1]
    phases = list(phase_stats.keys())
    means = [phase_stats[p]['mean'] for p in phases]
    stds = [phase_stats[p]['std'] for p in phases]

    bars = ax2.bar(phases, means, yerr=stds, capsize=5,
                   color=['lightblue', 'lightcoral', 'lightgreen'])
    ax2.set_title('各阶段平均k值')
    ax2.set_ylabel('预测步长 k')
    ax2.grid(True, alpha=0.3)

    # 添加数值标签
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{mean:.1f}', ha='center', va='bottom')

    # 子图3: k值分布直方图
    ax3 = axes[1, 0]
    all_k_values = []
    for episode in all_results:
        all_k_values.extend(episode['k_values'])

    ax3.hist(all_k_values, bins=20, alpha=0.7, edgecolor='black')
    ax3.set_title('k值分布直方图')
    ax3.set_xlabel('预测步长 k')
    ax3.set_ylabel('频次')
    ax3.grid(True, alpha=0.3)

    # 子图4: 成功率和k值关系
    ax4 = axes[1, 1]
    success_episodes = [ep for ep in all_results if ep['success']]
    failed_episodes = [ep for ep in all_results if not ep['success']]

    success_k = [np.mean(ep['k_values']) for ep in success_episodes]
    failed_k = [np.mean(ep['k_values']) for ep in failed_episodes]

    ax4.scatter(success_k, [1] * len(success_k), color='green',
                label='成功', s=50, alpha=0.7)
    ax4.scatter(failed_k, [0] * len(failed_k), color='red',
                label='失败', s=50, alpha=0.7)

    ax4.set_title('成功率 vs 平均k值')
    ax4.set_xlabel('平均预测步长 k')
    ax4.set_ylabel('成功 (1) / 失败 (0)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_yticks([0, 1])
    ax4.set_yticklabels(['失败', '成功'])

    plt.tight_layout()
    plt.savefig(output_dir / 'transport_validation_plots.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ 可视化图表已保存: {output_dir / 'transport_validation_plots.png'}")


if __name__ == "__main__":
    # 配置
    data_path = "/home/yhj/桌面/ACT/adastep_extension/robomimic_data/transport/mh/low_dim_v15.hdf5"
    predictor_path = "/home/yhj/桌面/ACT/adastep_extension/experiments/results_transport_mh/stage2_training/best_predictor.pth"

    # 检查文件存在
    if not os.path.exists(data_path):
        print(f"❌ 数据文件不存在: {data_path}")
        # 尝试查找替代文件
        transport_dir = "/home/yhj/桌面/ACT/adastep_extension/robomimic_data/transport"
        if os.path.exists(transport_dir):
            files = [f for f in os.listdir(transport_dir) if f.endswith('.hdf5')]
            if files:
                data_path = os.path.join(transport_dir, files[0])
                print(f"✓ 使用替代文件: {data_path}")
            else:
                print("❌ 未找到Transport数据文件")
                sys.exit(1)

    if not os.path.exists(predictor_path):
        print(f"❌ 预测器模型不存在: {predictor_path}")
        sys.exit(1)

    # 运行验证
    results = run_transport_validation(
        hdf5_path=data_path,
        predictor_path=predictor_path,
        num_episodes=10,  # 先用少量episode测试
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    if results:
        print(f"\n🎉 验证完成！")
        print(f"📊 关键结果:")
        print(f"  成功率: {results['success_rate']:.1f}%")
        print(f"  自适应得分: {results['validation']['adaptation_score']:.1f}%")
        print(f"  k值范围: {results['phase_analysis']['grasping']['mean']:.1f} - "
              f"{max([results['phase_analysis'][p]['mean'] for p in results['phase_analysis']]):.1f}")

        # 判断是否通过验证
        if (results['validation']['adaptation_score'] >= 75 and
            results['success_rate'] >= 50):
            print(f"\n✅ 状态级自适应验证通过！算法工作正常。")
        else:
            print(f"\n⚠️  需要进一步优化算法。")
    else:
        print("❌ 验证失败")