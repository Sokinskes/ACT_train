#!/bin/bash
# AdaStep 快速部署脚本
# 用法: ./scripts/quick_start_adastep.sh <task_name> <ckpt_dir>

set -e

TASK_NAME=${1:-"sim_transfer_cube_scripted"}
CKPT_DIR=${2:-"checkpoints/transfer_cube"}

echo "======================================================================="
echo "  AdaStep 快速部署脚本"
echo "======================================================================="
echo "任务: $TASK_NAME"
echo "检查点目录: $CKPT_DIR"
echo ""

# 检查环境
echo "🔍 检查环境..."
if ! command -v conda &> /dev/null; then
    echo "❌ 未找到 conda，请先安装 Anaconda/Miniconda"
    exit 1
fi

if ! conda env list | grep -q "^act "; then
    echo "❌ 未找到 act conda 环境，请先创建"
    exit 1
fi

echo "✅ 环境检查通过"

# 检查数据集
echo ""
echo "🔍 检查数据集..."
if [ ! -f "$CKPT_DIR/policy_last.ckpt" ]; then
    echo "❌ 未找到策略检查点: $CKPT_DIR/policy_last.ckpt"
    echo "请先训练 ACT 策略"
    exit 1
fi

if [ ! -f "$CKPT_DIR/dataset_stats.pkl" ]; then
    echo "❌ 未找到数据集统计: $CKPT_DIR/dataset_stats.pkl"
    exit 1
fi

echo "✅ 数据集检查通过"

# 运行烟雾测试
echo ""
echo "🧪 运行集成测试..."
conda run -n act python test_adastep_integration.py
if [ $? -ne 0 ]; then
    echo "❌ 集成测试失败"
    exit 1
fi

echo ""
echo "======================================================================="
echo "  🚀 开始训练 AdaStep"
echo "======================================================================="

# 训练 AdaStep
conda run -n act python train_adastep.py \
    --dataset_dir ./data \
    --ckpt_dir "$CKPT_DIR" \
    --k_min 5 \
    --k_max 50 \
    --num_clusters 10 \
    --lambda_param 1.0 \
    --epochs 100 \
    --batch_size 256 \
    --lr 1e-3

if [ $? -ne 0 ]; then
    echo "❌ AdaStep 训练失败"
    exit 1
fi

echo ""
echo "======================================================================="
echo "  📊 开始评估"
echo "======================================================================="

# 基线评估
echo ""
echo "1️⃣  固定 horizon 基线评估..."
conda run -n act python eval_adastep.py \
    --task_name "$TASK_NAME" \
    --ckpt_dir "$CKPT_DIR" \
    --num_rollouts 50 \
    | tee "$CKPT_DIR/baseline_results.txt"

# AdaStep 评估
echo ""
echo "2️⃣  AdaStep 自适应评估..."
conda run -n act python eval_adastep.py \
    --task_name "$TASK_NAME" \
    --ckpt_dir "$CKPT_DIR" \
    --predictor_ckpt "$CKPT_DIR/horizon_predictor_best.pth" \
    --use_adastep \
    --k_min 5 \
    --k_max 50 \
    --num_rollouts 50 \
    | tee "$CKPT_DIR/adastep_results.txt"

echo ""
echo "======================================================================="
echo "  ✅ 部署完成！"
echo "======================================================================="
echo ""
echo "📁 生成的文件:"
echo "  - $CKPT_DIR/horizon_predictor_best.pth   (预测器权重)"
echo "  - $CKPT_DIR/cluster_analyzer.pkl         (聚类状态)"
echo "  - $CKPT_DIR/horizon_distribution.png     (Horizon 分布图)"
echo "  - $CKPT_DIR/training_curves.png          (训练曲线)"
echo "  - $CKPT_DIR/baseline_results.txt         (基线结果)"
echo "  - $CKPT_DIR/adastep_results.txt          (AdaStep 结果)"
echo ""
echo "📊 下一步:"
echo "  1. 查看训练曲线: eog $CKPT_DIR/training_curves.png"
echo "  2. 对比结果: diff $CKPT_DIR/baseline_results.txt $CKPT_DIR/adastep_results.txt"
echo "  3. 部署到真机: python eval_adastep.py --use_adastep --device cuda"
echo ""
