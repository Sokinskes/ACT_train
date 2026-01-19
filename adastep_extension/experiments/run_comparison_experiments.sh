#!/bin/bash
# AdaStep 对比实验脚本
# 运行Square和Lift两个任务的实验

echo "=================================================================="
echo "🚀 AdaStep 对比实验启动"
echo "=================================================================="
echo ""

# 激活环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate act

cd /home/yhj/桌面/ACT/adastep_extension/experiments

# ============================================================
# 实验1: Square任务（高精度操作）
# ============================================================
echo "=================================================================="
echo "📊 实验1: Square任务（高精度插孔操作）"
echo "=================================================================="
echo ""

python run_full_experiment.py \
  --data_path ../robomimic_data/square/mh/low_dim_v15.hdf5 \
  --max_episodes 50 \
  --num_epochs 50 \
  --output_dir ./results_square \
  2>&1 | tee square_experiment.log

echo ""
echo "✓ Square实验完成！"
echo ""
sleep 2

# ============================================================
# 实验2: Lift任务（混合复杂度）- 放宽阈值
# ============================================================
echo "=================================================================="
echo "📊 实验2: Lift任务（抓取提升操作）- 优化参数"
echo "=================================================================="
echo ""

# 修改配置：放宽误差阈值
python run_full_experiment_lift.py \
  --data_path ../robomimic_data/lift/mh/low_dim_v15.hdf5 \
  --max_episodes 50 \
  --num_epochs 50 \
  --error_threshold 0.3 \
  --output_dir ./results_lift_optimized \
  2>&1 | tee lift_optimized_experiment.log

echo ""
echo "✓ Lift优化实验完成！"
echo ""

# ============================================================
# 生成对比报告
# ============================================================
echo "=================================================================="
echo "📝 生成对比报告"
echo "=================================================================="
echo ""

python generate_comparison_report.py

echo ""
echo "=================================================================="
echo "✅ 所有实验完成！"
echo "=================================================================="
echo ""
echo "结果位置:"
echo "  - Square: ./results_square/"
echo "  - Lift原始: ./results_lift/"
echo "  - Lift优化: ./results_lift_optimized/"
echo "  - 对比报告: ./COMPARISON_REPORT.md"
echo ""
