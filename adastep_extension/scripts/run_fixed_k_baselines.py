"""
运行 Fixed-k 基线实验,获取 Pareto Frontier 数据

用法:
    python run_fixed_k_baselines.py --task square --k_values 5 10 20 30 50
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime


def run_fixed_k_experiment(task, k_value, output_dir):
    """
    运行单个 Fixed-k 实验
    
    Args:
        task: 任务名称 (transport, can, lift, square)
        k_value: 固定步长
        output_dir: 输出目录
    """
    print(f"\n{'='*70}")
    print(f"运行 Fixed-k 实验: Task={task}, k={k_value}")
    print(f"{'='*70}\n")
    
    # 获取experiments目录路径
    script_dir = Path(__file__).parent
    experiments_dir = script_dir.parent / 'experiments'
    eval_script = experiments_dir / 'eval_offline_trajectory.py'
    
    # 构建命令
    cmd = [
        sys.executable,
        str(eval_script),
        "--task", task,
        "--fixed_k", str(k_value),
        "--device", "cuda",
        "--output_dir", str(output_dir / f"{task}_k{k_value}")
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            cwd=str(experiments_dir)  # 在experiments目录下运行
        )
        
        print(result.stdout)
        
        # 解析输出 (假设输出包含 JSON 格式的结果)
        # 你需要根据实际输出格式调整
        
        return {
            'task': task,
            'k': k_value,
            'success_rate': None,  # 从输出中解析
            'inferences': None,    # 从输出中解析
            'status': 'success'
        }
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 实验失败: {e}")
        print(f"错误输出: {e.stderr}")
        return {
            'task': task,
            'k': k_value,
            'status': 'failed',
            'error': str(e)
        }


def run_all_experiments(args):
    """运行所有实验"""
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for task in args.tasks:
        for k in args.k_values:
            result = run_fixed_k_experiment(task, k, output_dir)
            results.append(result)
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"fixed_k_results_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"✅ 所有实验完成!")
    print(f"{'='*70}")
    print(f"结果已保存: {output_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='运行 Fixed-k 基线实验')
    parser.add_argument('--tasks', nargs='+', 
                       default=['square', 'transport'],
                       help='任务列表')
    parser.add_argument('--k_values', nargs='+', type=int,
                       default=[5, 10, 20, 30, 50],
                       help='k 值列表')
    parser.add_argument('--output_dir', type=str,
                       default='../experiments/fixed_k_results',
                       help='输出目录')
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"Fixed-k 基线实验运行器")
    print(f"{'='*70}")
    print(f"任务: {args.tasks}")
    print(f"k 值: {args.k_values}")
    print(f"输出目录: {args.output_dir}")
    print()
    
    print("✅ eval_offline_trajectory.py 已支持 --fixed_k 参数")
    print(f"   共 {len(args.tasks) * len(args.k_values)} 个实验将运行...")
    print()
    
    run_all_experiments(args)


if __name__ == "__main__":
    main()
