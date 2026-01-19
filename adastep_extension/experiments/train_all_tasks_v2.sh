#!/bin/bash
# 使用改进算法在5个真实任务上训练AdaStep模型

echo "=========================================="
echo "AdaStep改进算法 - 真实数据验证"
echo "=========================================="
echo ""
echo "数据集: Robomimic v1.5"
echo "任务数: 5个 (CAN, LIFT, SQUARE, TOOL_HANG, TRANSPORT)"
echo "算法: AdaStep v2.0 (K=10, 动态阈值, 线性偏离度)"
echo ""

# 激活环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate act

# 数据路径前缀
DATA_DIR="/home/yhj/桌面/ACT/adastep_extension/robomimic_data"

# 任务列表
TASKS=("can" "lift" "square" "tool_hang" "transport")

# 遍历每个任务
for task in "${TASKS[@]}"; do
    echo ""
    echo "=========================================="
    echo "训练任务: ${task^^}"
    echo "=========================================="
    
    # 使用mh (multi-human)数据
    data_path="${DATA_DIR}/${task}/mh/low_dim_v15.hdf5"
    output_dir="results_${task}_v2"
    
    if [ -f "$data_path" ]; then
        echo "数据文件: $data_path"
        echo "输出目录: $output_dir"
        echo ""
        
        # 运行训练
        python run_full_experiment.py \
            --data_path "$data_path" \
            --num_epochs 100 \
            --output_dir "$output_dir" \
            2>&1 | tee "train_${task}_v2.log"
        
        echo "✓ ${task} 训练完成"
        
        # 分析k值分布
        echo ""
        echo "分析k值分布..."
        python -c "
import pickle
import numpy as np
import os

result_dir = '$output_dir/stage1_clustering'
if os.path.exists(result_dir):
    with open(f'{result_dir}/cluster_analyzer.pkl', 'rb') as f:
        data = pickle.load(f)
        horizons = data['cluster_horizons']
        
    # 加载labels
    labels = np.load(f'{result_dir}/horizon_labels.npy')
    
    # 统计k值分布
    k_values = []
    for label in labels:
        if label < len(horizons):
            k_values.append(horizons[int(label)])
    
    unique_k, counts = np.unique(k_values, return_counts=True)
    
    print(f'\\n=== ${task^^} - k值分布 ===')
    print(f'聚类数: {data[\"num_clusters\"]}, 动态阈值: {data[\"error_threshold\"]}')
    for k, count in zip(unique_k, counts):
        pct = count / len(k_values) * 100
        print(f'  k={k:2d}: {count:5d}个样本 ({pct:5.1f}%)')
    print(f'  k值种类数: {len(unique_k)}')
    print(f'  k值标准差: {np.std(k_values):.2f}')
    print()
"
    else
        echo "⚠ 数据文件不存在: $data_path"
    fi
done

echo ""
echo "=========================================="
echo "所有任务训练完成！"
echo "=========================================="
echo ""
echo "查看结果:"
echo "  cd /home/yhj/桌面/ACT/adastep_extension/experiments"
echo "  ls -d results_*_v2"
