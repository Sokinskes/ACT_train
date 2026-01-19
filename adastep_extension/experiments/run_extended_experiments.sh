#!/bin/bash
# AdaStep扩展实验 - 多任务验证
# 使用MH数据集（多人类演示，多样性高）

set -e  # 遇到错误立即退出

# 激活conda环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate act

# 实验配置
ERROR_THRESHOLD=0.4  # 基于Lift优化的最佳阈值
MAX_EPISODES=50
NUM_EPOCHS=100

echo "================================================================"
echo "AdaStep 扩展实验 - Robomimic MH数据集"
echo "================================================================"
echo ""
echo "实验配置:"
echo "  - 误差阈值: ${ERROR_THRESHOLD}"
echo "  - 最大轨迹数: ${MAX_EPISODES}"
echo "  - 训练轮数: ${NUM_EPOCHS}"
echo ""
echo "================================================================"
echo ""

# 任务1: Can（开罐器操作 - 预期中等复杂度）
echo ""
echo "▶ 任务1/3: Can (开罐器操作)"
echo "================================================================"
python run_full_experiment_lift_optimized.py \
    --data_path ../robomimic_data/can/mh/low_dim_v15.hdf5 \
    --error_threshold ${ERROR_THRESHOLD} \
    --max_episodes ${MAX_EPISODES} \
    --num_epochs ${NUM_EPOCHS} \
    --output_dir results_can_mh \
    2>&1 | tee can_mh_experiment.log

echo ""
echo "✓ Can任务完成！"
echo ""

# 等待5秒
sleep 5

# 任务2: Transport（物体搬运 - 预期低复杂度）
echo ""
echo "▶ 任务2/3: Transport (物体搬运)"
echo "================================================================"

# 检查Transport数据集是否存在
if [ -f "../robomimic_data/transport/mh/low_dim_v15.hdf5" ]; then
    python run_full_experiment_lift_optimized.py \
        --data_path ../robomimic_data/transport/mh/low_dim_v15.hdf5 \
        --error_threshold ${ERROR_THRESHOLD} \
        --max_episodes ${MAX_EPISODES} \
        --num_epochs ${NUM_EPOCHS} \
        --output_dir results_transport_mh \
        2>&1 | tee transport_mh_experiment.log
    
    echo ""
    echo "✓ Transport任务完成！"
else
    echo "⚠ Transport数据集不存在，跳过此任务"
fi

echo ""
sleep 5

# 任务3: Tool_Hang（工具悬挂 - 预期高精度）
echo ""
echo "▶ 任务3/3: Tool_Hang (工具悬挂)"
echo "================================================================"

if [ -f "../robomimic_data/tool_hang/mh/low_dim_v15.hdf5" ]; then
    python run_full_experiment_lift_optimized.py \
        --data_path ../robomimic_data/tool_hang/mh/low_dim_v15.hdf5 \
        --error_threshold ${ERROR_THRESHOLD} \
        --max_episodes ${MAX_EPISODES} \
        --num_epochs ${NUM_EPOCHS} \
        --output_dir results_tool_hang_mh \
        2>&1 | tee tool_hang_mh_experiment.log
    
    echo ""
    echo "✓ Tool_Hang任务完成！"
else
    echo "⚠ Tool_Hang数据集不存在，跳过此任务"
fi

echo ""
echo "================================================================"
echo "✅ 所有扩展实验完成！"
echo "================================================================"
echo ""
echo "结果目录:"
echo "  - Can: results_can_mh/stage3_validation/"
echo "  - Transport: results_transport_mh/stage3_validation/"
echo "  - Tool_Hang: results_tool_hang_mh/stage3_validation/"
echo ""
echo "下一步: 运行 generate_extended_comparison.py 生成对比报告"
echo ""
