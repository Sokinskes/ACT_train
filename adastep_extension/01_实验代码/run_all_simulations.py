"""
仿真评估配置和快速启动脚本
==================================

这个脚本帮助您快速运行AdaStep的真实仿真评估

使用步骤:
1. 修改下面的配置路径
2. 运行: bash run_simulation.sh
"""

import os
import subprocess


# ============ 配置区 ============

# 任务配置
TASKS = {
    'transport': {
        'data_path': '../robomimic_data/transport/mh/low_dim_v141.hdf5',
        'ckpt_path': '../checkpoints/transport_mh/policy_best.ckpt',
        'camera_names': ['agentview_image', 'eye_in_hand_image'],
        'state_dim': 14,
    },
    'square': {
        'data_path': '../robomimic_data/square/mh/low_dim_v141.hdf5',
        'ckpt_path': '../checkpoints/square_mh/policy_best.ckpt',
        'camera_names': ['agentview_image', 'eye_in_hand_image'],
        'state_dim': 14,
    },
    'lift': {
        'data_path': '../robomimic_data/lift/mh/low_dim_v141.hdf5',
        'ckpt_path': '../checkpoints/lift_mh/policy_best.ckpt',
        'camera_names': ['agentview_image', 'eye_in_hand_image'],
        'state_dim': 14,
    },
    'can': {
        'data_path': '../robomimic_data/can/mh/low_dim_v141.hdf5',
        'ckpt_path': '../checkpoints/can_mh/policy_best.ckpt',
        'camera_names': ['agentview_image', 'eye_in_hand_image'],
        'state_dim': 14,
    },
}

# 评估配置
EVAL_CONFIG = {
    'num_rollouts': 50,  # 每个任务运行50条轨迹
    'device': 'cuda',
    'render': False,  # 是否可视化（会很慢）
}

# ============ 配置区结束 ============


def check_files_exist(task_name, config):
    """检查必需文件是否存在"""
    errors = []
    
    if not os.path.exists(config['data_path']):
        errors.append(f"❌ 数据文件不存在: {config['data_path']}")
    
    if not os.path.exists(config['ckpt_path']):
        errors.append(f"❌ 模型文件不存在: {config['ckpt_path']}")
    
    if errors:
        print(f"\n任务 {task_name} 缺少文件:")
        for err in errors:
            print(f"  {err}")
        return False
    return True


def run_simulation(task_name, config):
    """运行单个任务的仿真"""
    print(f"\n{'='*70}")
    print(f"🚀 开始评估任务: {task_name.upper()}")
    print(f"{'='*70}")
    
    # 检查文件
    if not check_files_exist(task_name, config):
        print(f"⚠️  跳过任务 {task_name}")
        return None
    
    # 构建命令
    cmd = [
        'python', 'run_real_simulation.py',
        '--task', task_name,
        '--ckpt', config['ckpt_path'],
        '--data', config['data_path'],
        '--num_rollouts', str(EVAL_CONFIG['num_rollouts']),
        '--device', EVAL_CONFIG['device'],
        '--save_results', f'results_{task_name}_real_sim.pkl',
    ]
    
    if EVAL_CONFIG['render']:
        cmd.append('--render')
    
    # 运行命令
    print(f"执行命令: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 评估失败:")
        print(e.stderr)
        return False


def main():
    """主函数 - 运行所有任务的评估"""
    print("\n" + "="*70)
    print("AdaStep 真实仿真评估启动器")
    print("="*70)
    print(f"配置:")
    print(f"  - 每任务轨迹数: {EVAL_CONFIG['num_rollouts']}")
    print(f"  - 计算设备: {EVAL_CONFIG['device']}")
    print(f"  - 可视化: {EVAL_CONFIG['render']}")
    print(f"  - 任务列表: {list(TASKS.keys())}")
    
    # 运行每个任务
    results_summary = {}
    
    for task_name, config in TASKS.items():
        success = run_simulation(task_name, config)
        results_summary[task_name] = success
    
    # 打印总结
    print("\n" + "="*70)
    print("📊 评估总结")
    print("="*70)
    for task, success in results_summary.items():
        status = "✅ 成功" if success else "❌ 失败/跳过"
        print(f"  {task:15s}: {status}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
