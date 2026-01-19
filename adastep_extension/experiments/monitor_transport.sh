#!/bin/bash
# Transport任务进度监控脚本

echo "=================================================="
echo "Transport任务实时监控"
echo "=================================================="
echo ""
echo "任务特点:"
echo "  - 超长轨迹: 701.9步 (是Lift的9.3倍!)"
echo "  - 双臂协作: 动作维度14 (双臂各7维)"
echo "  - 预期复杂度: 低-中 (长距离搬运为主)"
echo ""
echo "预期结果:"
echo "  - 聚类k分布: [35, 45, 50]"
echo "  - 平均预测步长: 40-45"
echo "  - 推理节省: 85-90%"
echo ""
echo "=================================================="
echo ""

LOG_FILE="/home/yhj/桌面/ACT/adastep_extension/experiments/transport_mh_experiment.log"

while true; do
    clear
    echo "=================================================="
    echo "Transport任务实时监控 - $(date +%H:%M:%S)"
    echo "=================================================="
    echo ""
    
    # 检查当前阶段
    if grep -q "阶段1: 状态聚类" "$LOG_FILE" 2>/dev/null; then
        echo "✓ 阶段1: 状态聚类与帕累托分析"
        
        if grep -q "K-Means聚类完成" "$LOG_FILE"; then
            echo "  ✓ K-Means聚类完成"
        fi
        
        if grep -q "帕累托分析完成" "$LOG_FILE"; then
            echo "  ✓ 帕累托分析完成"
            echo ""
            echo "  聚类结果:"
            grep -A 3 "帕累托分析完成" "$LOG_FILE" | tail -3
        fi
    fi
    
    echo ""
    
    if grep -q "阶段2: 训练HorizonPredictor" "$LOG_FILE" 2>/dev/null; then
        echo "✓ 阶段2: 训练HorizonPredictor"
        
        # 获取最新训练进度
        LATEST_EPOCH=$(grep "Epoch" "$LOG_FILE" | tail -1)
        echo "  当前: $LATEST_EPOCH"
        
        if grep -q "早停触发" "$LOG_FILE"; then
            echo "  ✓ 早停触发（节省训练时间）"
        fi
        
        if grep -q "训练完成" "$LOG_FILE"; then
            echo "  ✓ 训练完成"
            grep "最佳验证损失" "$LOG_FILE" | tail -1
        fi
    fi
    
    echo ""
    
    if grep -q "阶段3: 离线验证" "$LOG_FILE" 2>/dev/null; then
        echo "✓ 阶段3: 离线验证实验"
        
        if grep -q "验证1: 预测准确率" "$LOG_FILE"; then
            echo "  ✓ 验证1: 预测准确率测试"
            ACCURACY=$(grep "总体准确率" "$LOG_FILE" | tail -1)
            if [ ! -z "$ACCURACY" ]; then
                echo "    $ACCURACY"
            fi
        fi
        
        if grep -q "验证2: 步长时序曲线" "$LOG_FILE"; then
            echo "  ✓ 验证2: 步长时序曲线分析"
        fi
        
        if grep -q "验证3: 动作预测误差" "$LOG_FILE"; then
            echo "  ✓ 验证3: 推理节省分析"
            SAVING=$(grep "推理次数节省" "$LOG_FILE" | tail -1)
            if [ ! -z "$SAVING" ]; then
                echo "    $SAVING"
            fi
        fi
    fi
    
    echo ""
    echo "=================================================="
    
    # 检查是否完成
    if grep -q "实验完成" "$LOG_FILE" 2>/dev/null; then
        echo ""
        echo "🎉 Transport任务实验完成！"
        echo ""
        echo "最终结果:"
        grep -A 5 "推理次数节省" "$LOG_FILE" | tail -6
        echo ""
        break
    fi
    
    sleep 5
done

echo "=================================================="
echo "✅ 监控结束"
echo "=================================================="
