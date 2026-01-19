"""
整合现有实验结果
基于已完成的离线验证生成综合报告
"""

import json
import os
from pathlib import Path

def collect_experiment_results():
    """收集所有任务的实验结果"""
    
    base_dir = Path("/home/yhj/桌面/ACT/adastep_extension/experiments")
    
    tasks = {
        'transport': 'results_transport_mh',
        'can': 'results_can_mh',
        'lift': 'results_lift_optimized',
        'square': 'results_square_mh'
    }
    
    results = {}
    
    for task_name, result_dir in tasks.items():
        report_path = base_dir / result_dir / 'stage3_validation' / 'EXPERIMENT_REPORT.md'
        
        if report_path.exists():
            with open(report_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # 解析结果
            inference_saving = None
            avg_k = None
            accuracy = None
            
            for line in content.split('\n'):
                if '推理次数节省' in line:
                    inference_saving = float(line.split(':')[1].strip().rstrip('%'))
                elif '平均步长' in line and 'AdaStep' in line:
                    avg_k = float(line.split(':')[1].strip())
                elif '总体准确率' in line:
                    accuracy = float(line.split('**')[1].rstrip('%'))
            
            results[task_name] = {
                'inference_saving': inference_saving,
                'avg_k': avg_k,
                'accuracy': accuracy,
                'report_path': str(report_path)
            }
            
            print(f"✅ {task_name:12s}: 推理节省 {inference_saving:.2f}%, 平均k={avg_k:.1f}, 准确率={accuracy:.1f}%")
        else:
            print(f"❌ {task_name:12s}: 未找到实验报告")
            results[task_name] = None
    
    return results


def generate_summary(results):
    """生成结果总结"""
    
    print("\n" + "="*70)
    print("📊 AdaStep 实验结果总结")
    print("="*70)
    
    print("\n### 离线验证结果 (100%真实数据)")
    print("\n| 任务 | 推理节省率 | 平均k | MLP准确率 | 数据状态 |")
    print("|------|-----------|-------|----------|---------|")
    
    for task, data in results.items():
        if data:
            print(f"| {task:12s} | {data['inference_saving']:.2f}% | {data['avg_k']:.1f} | {data['accuracy']:.1f}% | ✅ 真实 |")
        else:
            print(f"| {task:12s} | N/A | N/A | N/A | ❌ 缺失 |")
    
    # 计算平均值
    valid_results = [r for r in results.values() if r is not None]
    if valid_results:
        avg_saving = sum(r['inference_saving'] for r in valid_results) / len(valid_results)
        avg_k_all = sum(r['avg_k'] for r in valid_results) / len(valid_results)
        
        print(f"\n**平均值**: 推理节省 {avg_saving:.2f}%, 平均k={avg_k_all:.1f}")
    
    print("\n" + "="*70)
    print("✅ 这些数据是基于测试集的真实统计，可直接用于论文")
    print("⚠️  任务成功率需要通过在线仿真获取（当前为估计值）")
    print("="*70)
    
    return results


def generate_simple_success_estimates(results):
    """基于k值生成简单的成功率估计"""
    
    print("\n### 成功率估计 (基于k-penalty模型)")
    print("\n| 任务 | 估计成功率 | 状态 |")
    print("|------|-----------|------|")
    
    # 简单估计：成功率 ≈ baseline * (1 - k_penalty)
    # baseline约95%, k_penalty = (avg_k - 5) * 0.01
    
    for task, data in results.items():
        if data and data['avg_k']:
            k_penalty = max(0, (data['avg_k'] - 5) * 0.01)
            estimated_success = 95 * (1 - k_penalty * 0.1)
            print(f"| {task:12s} | {estimated_success:.1f}% | 🔄 估计值 |")
        else:
            print(f"| {task:12s} | N/A | ❌ 缺失 |")
    
    print("\n⚠️  注意：这些是基于数学模型的估计值，非真实仿真结果")
    print("   如需真实数据，需要运行在线仿真（需要完整的ACT模型）")


if __name__ == "__main__":
    print("🔍 收集现有实验结果...")
    results = collect_experiment_results()
    
    print("\n" + "="*70)
    summary = generate_summary(results)
    generate_simple_success_estimates(results)
    
    # 保存结果
    output_file = "/home/yhj/桌面/ACT/adastep_extension/EXPERIMENT_RESULTS_SUMMARY.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ 结果已保存到: {output_file}")
