#!/bin/bash
#
# 批量运行 Fixed-k 基线实验
# 用法: bash run_all_fixed_k.sh
#

set -e  # 遇到错误立即退出

cd "$(dirname "$0")/../experiments"

TASKS="square transport"
K_VALUES="5 10 20 30 50"
OUTPUT_DIR="fixed_k_baselines"

echo "========================================================================"
echo " Fixed-k 基线实验批处理"
echo "========================================================================"
echo "任务: $TASKS"
echo "k值: $K_VALUES"
echo "输出目录: $OUTPUT_DIR"
echo ""

mkdir -p "$OUTPUT_DIR"

# 创建结果汇总文件
SUMMARY_FILE="$OUTPUT_DIR/fixed_k_summary.txt"
echo "Fixed-k Baseline Results" > "$SUMMARY_FILE"
echo "=========================" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

# 运行所有实验
TOTAL_EXPERIMENTS=$(echo "$TASKS" | wc -w)
TOTAL_EXPERIMENTS=$((TOTAL_EXPERIMENTS * $(echo "$K_VALUES" | wc -w)))
CURRENT=0

for task in $TASKS; do
    for k in $K_VALUES; do
        CURRENT=$((CURRENT + 1))
        echo ""
        echo "###################################################################"
        echo "# [$CURRENT/$TOTAL_EXPERIMENTS] Task: $task, k=$k"
        echo "###################################################################"
        echo ""
        
        # 创建输出目录
        TASK_OUTPUT_DIR="$OUTPUT_DIR/${task}_k${k}"
        mkdir -p "$TASK_OUTPUT_DIR"
        
        # 运行实验
        conda run -n act --no-capture-output python eval_offline_trajectory.py \
            --task "$task" \
            --fixed_k "$k" \
            --device cuda \
            --output_dir "$TASK_OUTPUT_DIR" \
            2>&1 | tee "$TASK_OUTPUT_DIR/run.log"
        
        # 提取结果到汇总文件
        echo "Task: $task, k=$k" >> "$SUMMARY_FILE"
        grep -E "(成功率|推理次数)" "$TASK_OUTPUT_DIR/run.log" | head -2 >> "$SUMMARY_FILE" || echo "  (解析失败)" >> "$SUMMARY_FILE"
        echo "" >> "$SUMMARY_FILE"
    done
done

echo ""
echo "========================================================================"
echo " ✅ 所有实验完成!"
echo "========================================================================"
echo "结果保存在: $OUTPUT_DIR/"
echo "汇总文件: $SUMMARY_FILE"
echo ""

# 打印汇总
cat "$SUMMARY_FILE"
