#!/usr/bin/env python3
"""
收集 Fixed-k 基线实验结果
从各个实验目录中提取结果并生成汇总表格
"""

import json
import os
from pathlib import Path
import argparse


def collect_results(base_dir):
    """收集所有Fixed-k实验结果"""
    
    results = []
    
    # 遍历所有子目录
    for task_dir in sorted(Path(base_dir).glob("*_k*")):
        if not task_dir.is_dir():
            continue
        
        # 解析task和k值
        dirname = task_dir.name
        parts = dirname.rsplit('_k', 1)
        if len(parts) != 2:
            continue
        
        task_name = parts[0]
        k_value = int(parts[1])
        
        # 查找summary文件
        summary_file = task_dir / "all_tasks_summary.json"
        detail_file = task_dir / f"{task_name}_detailed.json"
        
        if summary_file.exists():
            with open(summary_file) as f:
                data = json.load(f)
                if data:
                    item = data[0]  # 取第一个
                    results.append({
                        'task': task_name,
                        'k': k_value,
                        'success_rate': item['adastep_completion'],
                        'num_inferences': item['num_inferences'],
                        'status': 'success'
                    })
        elif detail_file.exists():
            with open(detail_file) as f:
                data = json.load(f)
                summary = data.get('summary', {})
                results.append({
                    'task': task_name,
                    'k': k_value,
                    'success_rate': summary.get('adastep_completion'),
                    'num_inferences': summary.get('num_inferences'),
                    'status': 'success'
                })
        else:
            print(f"⚠️  未找到结果文件: {task_dir}")
            results.append({
                'task': task_name,
                'k': k_value,
                'status': 'missing'
            })
    
    return results


def print_table(results):
    """打印结果表格"""
    
    # 按task和k排序
    results_sorted = sorted(results, key=lambda x: (x['task'], x['k']))
    
    print("\n" + "="*80)
    print(" Fixed-k 基线实验结果汇总")
    print("="*80)
    print()
    
    # 按任务分组
    tasks = {}
    for r in results_sorted:
        task = r['task']
        if task not in tasks:
            tasks[task] = []
        tasks[task].append(r)
    
    # 打印每个任务的结果
    for task_name, task_results in tasks.items():
        print(f"【{task_name.upper()}】")
        print(f"  {'k值':<8} {'成功率':<12} {'推理次数':<12} 状态")
        print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*10}")
        
        for r in task_results:
            k = r['k']
            if r['status'] == 'success':
                success = f"{r['success_rate']:.1f}%"
                inferences = f"{r['num_inferences']:.1f}"
                status = "✅"
            else:
                success = "N/A"
                inferences = "N/A"
                status = "❌"
            
            print(f"  {k:<8} {success:<12} {inferences:<12} {status}")
        
        print()
    
    print("="*80)


def generate_python_dict(results):
    """生成可用于更新generate_paper_figures.py的Python字典代码"""
    
    # 按任务分组
    tasks = {}
    for r in results:
        if r['status'] != 'success':
            continue
        task = r['task']
        if task not in tasks:
            tasks[task] = {}
        tasks[task][r['k']] = {
            'success': r['success_rate'],
            'inferences': r['num_inferences']
        }
    
    print("\n" + "="*80)
    print(" Python 字典代码 (用于 generate_paper_figures.py)")
    print("="*80)
    print()
    
    for task_name, task_data in tasks.items():
        print(f"# {task_name.upper()} task")
        print(f"fixed_k_data_{task_name} = {{")
        for k in sorted(task_data.keys()):
            d = task_data[k]
            print(f"    {k}: {{'success': {d['success']:.1f}, 'inferences': {d['inferences']:.1f}}},")
        print("}")
        print()


def main():
    parser = argparse.ArgumentParser(description='收集Fixed-k实验结果')
    parser.add_argument('--base_dir', type=str,
                       default='../experiments/fixed_k_baselines',
                       help='结果基础目录')
    parser.add_argument('--output', type=str,
                       default=None,
                       help='输出JSON文件路径')
    
    args = parser.parse_args()
    
    # 收集结果
    results = collect_results(args.base_dir)
    
    if not results:
        print("❌ 未找到任何结果")
        return
    
    # 打印表格
    print_table(results)
    
    # 生成Python代码
    generate_python_dict(results)
    
    # 保存JSON
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✅ 结果已保存: {args.output}\n")


if __name__ == "__main__":
    main()
