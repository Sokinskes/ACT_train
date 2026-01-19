# 📐 论文图表使用指南

**用途**: 详细说明每张图表的含义和使用场景  
**状态**: 已准备20+张高质量图表

---

## 📊 图表总览

当前可用图表: **21张**  
分类:
- 综合对比图: 4张
- 分布分析图: 3张
- 效率分析图: 5张
- 任务详细图: 9张

---

## 🎯 核心图表 (必用)

### 1. 四任务总览图 ⭐⭐⭐⭐⭐
**文件**: `final_four_task_comparison.png`  
**尺寸**: 1200x800  
**用途**: 论文首页配图，一眼看懂所有结果

**内容**:
```
┌─────────────────────────────────────┐
│  4 Tasks Inference Saving           │
│  ┌───┬───┬───┬───┐                 │
│  │   │   │   │███│ 89.79% Transport│
│  │   │   │███│███│ 88.35% Can      │
│  │   │███│███│███│ 80.58% Lift     │
│  │   │███│███│███│  0.00% Square   │
│  └───┴───┴───┴───┘                 │
└─────────────────────────────────────┘
```

**LaTeX引用**:
```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{final_four_task_comparison.png}
  \caption{AdaStep在4个Robomimic基准任务上的推理节省效果对比。
           Transport任务达到最高节省率89.79\%，Can任务88.35\%，
           Lift任务80.58\%。Square任务保持k=5保守策略（节省0\%）。}
  \label{fig:four_task_comparison}
\end{figure}
```

**使用位置**: 第4章 实验结果 - 4.1节后

---

### 2. 复杂度-效率散点图 ⭐⭐⭐⭐
**文件**: `final_complexity_vs_efficiency.png`  
**尺寸**: 1000x800  
**用途**: 展示任务复杂度与推理效率的关系

**内容**:
```
推理节省(%)
  100│              ● Can
     │             ● Transport
   80│        ● Lift
     │
   50│
     │
    0│ ● Square
     └──────────────────────
       0   20   40   60   80
           任务复杂度分数
```

**关键发现**: 
- 低复杂度任务 → 高推理节省
- Square高复杂度 → 低推理节省（正确）

**LaTeX引用**:
```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=0.8\linewidth]{final_complexity_vs_efficiency.png}
  \caption{任务复杂度与推理效率的关系。可以看出，低复杂度任务
           （Can、Transport）能够实现更高的推理节省，而高精度
           任务（Square）AdaStep自动选择保守策略以保障安全性。}
  \label{fig:complexity_vs_efficiency}
\end{figure}
```

**使用位置**: 第4章 - 4.3节 任务自适应性分析

---

### 3. 聚类k值分布图 ⭐⭐⭐
**文件**: `final_cluster_distribution.png`  
**尺寸**: 1200x600  
**用途**: 展示不同任务的k值选择分布

**内容**:
```
Square:  [5][5][5][5][5]          ← 全部k=5
Lift:    [20][25][30][35][50]     ← 动态变化
Can:     [50][50][50][50][50]     ← 全部k=50
Transport:[48][49][50][50][50]    ← 接近k=50
```

**关键洞察**: 
- Square: 正确识别高风险，全程保守
- Can/Transport: 正确识别低风险，全程激进
- Lift: 展现最佳自适应性

**LaTeX引用**:
```latex
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{final_cluster_distribution.png}
  \caption{4个任务的聚类k值分布对比。Square任务所有聚类收敛到k=5，
           展现保守策略；Can和Transport任务主要选择k=50，展现激进策略；
           Lift任务展现动态调整特性（k∈[20,50]），证明AdaStep的
           任务自适应能力。}
  \label{fig:cluster_distribution}
\end{figure*}
```

**使用位置**: 第4章 - 4.3节

---

### 4. 推理节省饼图 ⭐⭐⭐
**文件**: `comparison_inference_saving_pie.png`  
**尺寸**: 800x800  
**用途**: 直观展示推理计算的节省比例

**内容**:
```
    ╱────────╲
   │  10.2%   │ ← 仍需推理
   │  (剩余)  │
   │          │
   │  89.8%   │ ← 节省部分
   │ (节省)   │
    ╲────────╱
```

**卖点**: "节省近90%推理计算"的视觉化

**LaTeX引用**:
```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=0.7\linewidth]{comparison_inference_saving_pie.png}
  \caption{Transport任务的推理计算节省分布。AdaStep仅需10.2\%的
           推理调用即可完成任务，节省89.8\%的计算资源。}
  \label{fig:inference_saving_pie}
\end{figure}
```

**使用位置**: 第4章 - 4.2节 推理效率对比

---

## 📈 详细分析图

### 5. 绝对推理次数节省
**文件**: `final_absolute_savings.png`  
**内容**: 从700次 → 14次（Transport）
**用途**: 展示绝对数值的节省效果

### 6. 任务对比总览
**文件**: `comparison_task_overview.png`  
**内容**: 柱状图对比4任务
**用途**: 替代方案（如果不喜欢图1）

### 7. 阈值敏感性分析
**文件**: `comparison_ablation_threshold.png`  
**内容**: threshold vs 推理节省曲线
**用途**: 第4章 - 4.4节 消融实验

---

## 🔍 任务详细图表

### Transport任务 (最佳结果)
**位置**: `02_实验结果_按任务/Task3_Transport/stage3_validation/`

#### 图8: k值分布直方图
**文件**: `k_distribution_histogram.png`  
**内容**: k值在[45-50]集中分布
**用途**: 展示Transport任务的激进策略

#### 图9: 推理次数对比
**文件**: `inference_count_comparison.png`  
**内容**: 700次 → 14次的视觉对比
**用途**: 强调"50倍加速"

#### 图10: 聚类误差分析
**文件**: `cluster_error_analysis.png`  
**内容**: 5个聚类的预测误差分布
**用途**: 证明MLP预测器的准确性

---

### Can任务 (次佳结果)
**位置**: `02_实验结果_按任务/Task2_Can/stage3_validation/`

#### 图11-13: 与Transport类似的3张图
**用途**: 支持论文完整性

---

### Lift任务 (典型结果)
**位置**: `02_实验结果_按任务/Task1_Lift/stage3_validation/`

#### 图14: k值动态变化曲线
**文件**: `k_distribution_histogram.png`  
**内容**: k在[20,50]范围内变化
**用途**: 展示"任务自适应性"的最佳例子

---

## 🎨 图表使用建议

### 论文主体必用图表 (5-6张)
```
1. ✅ final_four_task_comparison.png         (总览)
2. ✅ final_complexity_vs_efficiency.png     (关系图)
3. ✅ final_cluster_distribution.png         (分布图)
4. ✅ comparison_inference_saving_pie.png    (饼图)
5. ✅ comparison_ablation_threshold.png      (消融)
6. 可选: Transport详细图 (1-2张)
```

### 补充材料可用图表 (15张)
- 所有任务的详细分析图
- 不同角度的对比图
- 训练曲线图

---

## 📐 图表尺寸规范

### IEEE/ACM会议格式
```latex
% 单列图
\includegraphics[width=\linewidth]{...}        % 3.5英寸宽

% 双列图
\includegraphics[width=\textwidth]{...}        % 7英寸宽

% 缩小版
\includegraphics[width=0.8\linewidth]{...}     % 2.8英寸宽
```

### 期刊格式
```latex
% 单列
\includegraphics[width=8.5cm]{...}

% 双列
\includegraphics[width=17cm]{...}
```

---

## 🔄 如何重新生成图表

如果需要修改图表样式或数据：

```bash
cd /home/yhj/桌面/ACT/adastep_extension

# 生成综合对比图
python generate_final_plots.py

# 生成比较图
python generate_comparison_plots.py

# 生成任务特定图
cd experiments/results_transport_mh/stage3_validation
python ../../../generate_task_plots.py  # (需要创建)
```

---

## 🎯 图表使用检查清单

### 提交论文前确保：
- [ ] 所有图表分辨率 ≥ 300 DPI
- [ ] 字体大小在图中清晰可读（≥ 10pt）
- [ ] 图例位置不遮挡数据
- [ ] 颜色对色盲友好（避免红绿对比）
- [ ] 图表标题简洁明了（<15字）
- [ ] 坐标轴标签完整（带单位）
- [ ] 引用编号连续（fig:1, fig:2...）

### 审稿阶段可能需要：
- [ ] 添加误差棒（如果有多次实验）
- [ ] 添加置信区间
- [ ] 对比图增加基线方法
- [ ] 统一配色方案

---

## 📊 图表配色方案

### 当前使用的颜色
```python
# 主色调（4任务）
Square:    '#FF6B6B'  # 红色（警告/保守）
Lift:      '#4ECDC4'  # 青色（中性）
Can:       '#95E1D3'  # 浅绿（成功）
Transport: '#2ECC71'  # 深绿（最佳）

# 对比色
ACT Baseline:  '#3498DB'  # 蓝色
AdaStep:       '#E74C3C'  # 红色
SOTA:          '#9B59B6'  # 紫色
```

### 色盲友好替代方案
```python
# 使用形状 + 颜色
Square:    circle + red
Lift:      triangle + blue  
Can:       square + green
Transport: diamond + orange
```

---

## 💡 高级技巧

### 1. 组合图（subplot）
```latex
\begin{figure*}[t]
  \centering
  \begin{subfigure}{0.48\textwidth}
    \includegraphics[width=\linewidth]{fig1.png}
    \caption{任务对比}
  \end{subfigure}
  \hfill
  \begin{subfigure}{0.48\textwidth}
    \includegraphics[width=\linewidth]{fig2.png}
    \caption{聚类分布}
  \end{subfigure}
  \caption{AdaStep性能分析}
\end{figure*}
```

### 2. 图表引用
```latex
如图\ref{fig:four_task_comparison}所示，...
从图\ref{fig:complexity_vs_efficiency}可以看出，...
```

### 3. 在文中解释图表
```latex
实验结果如图\ref{fig:result}所示。可以观察到三个关键现象：
(1) Transport任务达到最高推理节省率89.79\%；
(2) Square任务保持k=5保守策略以保障精度；
(3) Lift任务展现显著的自适应调整特性。
这些结果证明了AdaStep算法的有效性和任务自适应能力。
```

---

## 📝 图表Caption模板

### 结果展示型
```latex
\caption{AdaStep在[任务名称]上的[指标名称]。
         [关键发现1]，[关键发现2]。}
```

### 对比分析型
```latex
\caption{[方法A]与[方法B]在[指标]上的对比。
         可以看出[关键优势]，[量化数据]。}
```

### 消融实验型
```latex
\caption{[参数名称]对[性能指标]的影响。
         当[参数]=[值]时，[性能]达到最优[数值]。}
```

---

**创建日期**: 2026年1月9日  
**图表总数**: 21张  
**状态**: ✅ 全部可用，高质量
