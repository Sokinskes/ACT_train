#!/bin/bash

# 设置环境
echo "🚀 AdaStep算法完整训练"
echo "训练所有5个Robomimic任务"
echo "========================================"

# 激活conda环境
source ~/anaconda3/etc/profile.d/conda.sh && conda activate act

# 设置变量
BASE_DIR="/home/yhj/桌面/ACT/adastep_extension"
EXPERIMENTS_DIR="$BASE_DIR/experiments"
DATA_DIR="$BASE_DIR/robomimic_data"
RESULTS_DIR="$BASE_DIR/results"

# 创建结果目录
mkdir -p $RESULTS_DIR

# 任务列表
TASKS=("square" "transport" "can" "lift")
DATA_TYPES=("mh" "mh" "mh" "mh")  # 都使用mh (Multi-Human)

echo "📋 训练计划:"
echo "  任务数: ${#TASKS[@]}"
echo "  数据类型: mh (Multi-Human)"
echo "  算法: AdaStep v2.0 (状态级自适应)"
echo ""

# 逐个训练任务
for i in "${!TASKS[@]}"; do
    TASK="${TASKS[$i]}"
    DATA_TYPE="${DATA_TYPES[$i]}"

    echo "========================================="
    echo "🎯 训练任务: $TASK ($DATA_TYPE)"
    echo "========================================="

    # 数据路径
    DATA_PATH="$DATA_DIR/${TASK}/${DATA_TYPE}/low_dim_v15.hdf5"

    if [ ! -f "$DATA_PATH" ]; then
        echo "❌ 数据文件不存在: $DATA_PATH"
        echo "   请先下载数据文件"
        continue
    fi

    # 结果目录
    TASK_RESULTS_DIR="$RESULTS_DIR/results_${TASK}_${DATA_TYPE}"
    mkdir -p $TASK_RESULTS_DIR

    echo "📂 数据路径: $DATA_PATH"
    echo "📁 结果目录: $TASK_RESULTS_DIR"

    # 运行训练
    echo "🏃 开始训练..."
    cd $BASE_DIR  # 切换到AdaStep扩展目录

    python train_adastep_improved.py \
        --task $TASK \
        --data_path $DATA_PATH \
        --results_dir $TASK_RESULTS_DIR \
        --num_epochs 50 \
        --batch_size 4 \
        --lr 1e-4 \
        --seed 42

    # 检查训练结果
    if [ $? -eq 0 ]; then
        echo "✅ $TASK 训练完成!"
    else
        echo "❌ $TASK 训练失败!"
    fi

    echo ""
done

echo "========================================="
echo "🎉 所有任务训练完成!"
echo "========================================="

# 生成汇总报告
cd $EXPERIMENTS_DIR

python -c "
import os
import json
from pathlib import Path

results_dir = Path('$RESULTS_DIR')
summary = {}

print('AdaStep训练汇总报告')
print('='*50)

for task_dir in results_dir.glob('results_*'):
    if task_dir.is_dir():
        task_name = task_dir.name.replace('results_', '')
        summary[task_name] = {}

        # 检查是否有训练结果
        train_results = task_dir / 'training_results.json'
        if train_results.exists():
            summary[task_name]['status'] = 'completed'
            print(f'✅ {task_name}: 训练完成')
        else:
            summary[task_name]['status'] = 'failed'
            print(f'❌ {task_name}: 训练失败')

# 保存汇总
with open(results_dir / 'training_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f'\\n📁 汇总报告已保存到: {results_dir}/training_summary.json')
"

echo ""
echo "🎯 下一步:"
echo "  1. 检查训练日志: $RESULTS_DIR/results_*/training_results.json"
echo "  2. 查看模型文件: $RESULTS_DIR/results_*/models/"
echo "  3. 运行评估: python evaluate.py --results_dir $RESULTS_DIR"
echo ""
echo "🚀 AdaStep算法训练完成!"