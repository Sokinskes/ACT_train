#!/bin/bash
# AdaStep 实验启动脚本（激活conda环境）

echo "======================================================================"
echo "🚀 AdaStep 实验启动器"
echo "======================================================================"
echo ""

# 激活conda环境
echo "📦 激活conda环境: act"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate act

# 验证环境
echo "✓ Python: $(which python)"
echo "✓ Version: $(python --version)"
echo ""

# 检查数据集路径
DATA_PATH="robomimic_data/square/mh/low_dim_v15.hdf5"

if [ ! -f "$DATA_PATH" ]; then
    echo "❌ 错误：未找到数据集文件！"
    echo "   期望路径: $DATA_PATH"
    exit 1
fi

echo "✓ 找到数据集: $DATA_PATH"
echo ""

# 进入实验目录并运行
cd experiments || exit 1

echo "======================================================================"
echo "📊 开始运行AdaStep完整实验"
echo "======================================================================"
echo ""
echo "配置:"
echo "  - 数据集: ../$DATA_PATH"
echo "  - 最大轨迹数: 50"
echo "  - 训练轮数: 100"
echo "  - 误差阈值: 0.15 (已优化)"
echo ""
echo "⏱️  预计运行时间: ~30分钟"
echo "💡 可以随时按 Ctrl+C 停止"
echo ""

# 运行实验
python run_full_experiment.py \
  --data_path "../$DATA_PATH" \
  --max_episodes 50 \
  --num_epochs 100

# 检查运行状态
if [ $? -eq 0 ]; then
    echo ""
    echo "======================================================================"
    echo "✅ 实验完成！"
    echo "======================================================================"
    echo ""
    echo "生成的文件:"
    echo "  - results/stage1_clustering/cluster_analyzer.pkl"
    echo "  - results/stage2_training/best_predictor.pth"
    echo "  - results/stage3_validation/*.png (4张图表)"
    echo ""
    echo "🌟 重点查看: results/stage3_validation/validation_2_temporal_curve.png"
    echo "   这是论文第三章的核心图（凹字形曲线）"
    echo ""
else
    echo ""
    echo "❌ 实验失败！请检查上面的错误信息。"
    echo ""
fi
