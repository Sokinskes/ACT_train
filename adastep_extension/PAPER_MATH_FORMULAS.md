# AdaStep 论文核心数学公式速查表

**用途:** 写论文/回复审稿人时快速查找关键公式

---

## 📐 Section 3.1: Problem Formulation

### 1. 标准 Action Chunking 执行流程
```latex
a_t, a_{t+1}, \ldots, a_{t+k-1} \sim \pi_\theta(a_{t:t+T} | s_t), \quad s_{t+k} \leftarrow \text{env.step}(a_{t:t+k-1})
```

**含义:** 在 $t$ 时刻预测 $T$ 个动作,执行前 $k$ 个,然后在 $t+k$ 时刻重新推理

---

### 2. k-Step 累积误差函数 (Cumulative Divergence Function)
```latex
\mathcal{E}(s_t, k) = \sum_{i=1}^{k} \| \hat{a}_{t+i-1} - a^*_{t+i-1} \|_2
```

**变量说明:**
- $\hat{a}_{t+i-1}$: 模型预测的第 $i$ 步动作 (开环)
- $a^*_{t+i-1}$: 专家在真实状态下的动作 (闭环)
- $\mathcal{E}(s_t, k)$: 累积轨迹偏差

**物理意义:** 
- 简单状态 (自由空间): $L_k \approx 0.01$ → 误差亚线性增长 → 可用大 $k$
- 复杂状态 (接触): $L_k \approx 0.15$ → 误差超线性增长 → 需小 $k$

---

### 3. 约束优化目标 (Constrained Optimization)
```latex
k^*(s_t) = \arg\max_{k \in [k_{\min}, k_{\max}]} \quad k
```
```latex
\text{subject to} \quad \mathcal{E}(s_t, k) \leq \delta_{safe}
```

**含义:** 在满足误差安全约束的前提下,选择最大的执行步长 $k$ (Pareto 最优)

**两个竞争目标:**
1. **计算效率:** $\mathcal{C}(k) \propto \frac{1}{k}$ (步长越大,推理越少)
2. **执行安全:** $\mathcal{R}(s_t, k) = \mathcal{E}(s_t, k)$ (步长越大,误差越大)

**超参数:**
- $k_{\min} = 1$: 最小步长 (最安全)
- $k_{\max} = 50$: 最大步长 (ACT 预测视野)
- $\delta_{safe} = 0.02$: 安全阈值 (通过交叉验证确定)

---

### 4. 自适应视野控制流程
```latex
k_t = h_\phi(s_t), \quad a_{t:t+k_t-1} \sim \pi_\theta(a_{t:t+T} | s_t), \quad s_{t+k_t} \leftarrow \text{env.step}(a_{t:t+k_t-1})
```

**含义:** 每一步动态预测最优 $k_t$,而非使用固定值

---

## 📊 Section 3.2: Pareto-Optimal Horizon Assignment

### Stage 1: State Manifold Clustering

#### 5. K-Means 聚类目标
```latex
\min_{\{\mu_j\}} \sum_{j=1}^K \sum_{z_i \in \mathcal{C}_j} \| z_i - \mu_j \|^2
```

**变量说明:**
- $z_i = E_{vision}(s_i)$: 从 ACT 编码器提取的视觉特征
- $\mu_j$: 第 $j$ 个簇的质心
- $\mathcal{C}_j$: 第 $j$ 个簇的样本集合
- $K = 3$: 簇数量 (经验值)

**Manifold Hypothesis:** 同一簇的状态具有相似的 Lipschitz 常数 $L_k^{(j)}$

---

### Stage 2: Pareto Frontier Labeling

#### 6. 聚类误差度量
```latex
\bar{E}_j(k) = \frac{1}{|\mathcal{C}_j|} \sum_{s_i \in \mathcal{C}_j} \| \Delta a_{i:i+k} \|_2
```
```latex
\sigma_j(k) = \text{std}(\{ \| \Delta a_{i:i+k} \|_2 \mid s_i \in \mathcal{C}_j \})
```

**含义:**
- $\bar{E}_j(k)$: 簇 $j$ 内所有样本的平均 $k$-step 动作方差
- $\sigma_j(k)$: 方差的标准差 (衡量簇内差异性)
- $\Delta a = a_{i+1} - a_i$: 动作速度变化

**直觉:** 高 $\bar{E}_j(k)$ 表示动作剧烈变化 → 需要频繁重规划 → 小 $k$

---

#### 7. 最优视野标签 (Pareto-Optimal Label)
```latex
k_j^* = \max \{ k \in [k_{\min}, k_{\max}] \mid \bar{E}_j(k) + \lambda \cdot \sigma_j(k) < \delta_{safe} \}
```

**变量说明:**
- $k_j^*$: 簇 $j$ 的最优步长
- $\lambda = 1.0$: 安全系数 (覆盖 1 个标准差内的样本)
- $\delta_{safe} = 0.02$: 误差容忍阈值

**Pareto 最优性:** 在所有满足 $\bar{E}_j(k) + \lambda \sigma_j(k) < \delta_{safe}$ 的 $k$ 中,选择最大值 → 最小化计算量

**标签广播:** 簇内所有样本共享相同标签 $k_i^* = k_j^*$ if $s_i \in \mathcal{C}_j$

---

## 🧠 Section 3.3: Online Horizon Predictor Learning

#### 8. 网络架构
```latex
h_\phi(z_t) = \sigma(\mathbf{W}_3 \cdot \text{ReLU}(\mathbf{W}_2 \cdot \text{ReLU}(\mathbf{W}_1 \cdot z_t)))
```

**架构细节:**
- 输入: $z_t \in \mathbb{R}^{512}$ (ACT 特征编码器输出,frozen)
- 隐藏层: [512 → 256 → 128]
- 激活函数: ReLU
- 输出层: Sigmoid $\sigma(\cdot) \in [0, 1]$

**参数量:** <1M (轻量级)
**推理延迟:** 0.8ms (RTX 3090)

---

#### 9. 归一化目标
```latex
\tilde{k}_i = \frac{k_i^* - k_{\min}}{k_{\max} - k_{\min}} \in [0, 1]
```

**反归一化 (推理时):**
```latex
k_t = \lfloor h_\phi(z_t) \cdot (k_{\max} - k_{\min}) + k_{\min} \rfloor
```

---

#### 10. 损失函数 (回归问题)
```latex
\mathcal{L}_{horizon}(\phi) = \frac{1}{N} \sum_{i=1}^N (\hat{k}_i - \tilde{k}_i)^2 + \lambda \|\phi\|_2^2
```

**变量说明:**
- $\hat{k}_i = h_\phi(z_i)$: 预测的归一化 $k$ 值
- $\tilde{k}_i$: 真实的归一化标签
- $\lambda = 1 \times 10^{-5}$: L2 正则化系数

---

#### 11. 联合训练 (可选)
```latex
\mathcal{L}_{total} = \mathcal{L}_{ACT}(\theta) + \alpha \cdot \mathcal{L}_{horizon}(\phi)
```

**变量说明:**
- $\mathcal{L}_{ACT}(\theta)$: ACT 策略的原始损失 (CVAE + 模仿学习)
- $\alpha = 0.1$: 视野预测器的权重系数

**优势:** 端到端优化,特征编码器可微调 (如果不 freeze)

---

## 📈 实验结果关键指标

### 12. 推理节省率 (Inference Reduction Rate)
```latex
\text{Reduction Rate} = \frac{T_{baseline} - T_{AdaStep}}{T_{baseline}} \times 100\%
```

其中:
- $T_{baseline}$: 固定步长 $k=1$ 的推理次数 (Episode 长度)
- $T_{AdaStep}$: AdaStep 的推理次数 ($\sum_{t} \frac{1}{k_t}$)

**实验结果:**
- Transport: 97.9% reduction ($k=50$)
- Can: 97.8% reduction ($k=50$)
- Lift: 96.8% reduction ($k=35$)
- Square: 94.1% reduction ($k=17$, range 6-30)
- **Average: 96.7% reduction (38× speedup)**

---

### 13. 成功率 (Success Rate)
```latex
\text{Success Rate} = \frac{\text{# Successful Episodes}}{\text{Total Episodes}} \times 100\%
```

**定义 (Robomimic):**
- Transport: 物体放置在目标区域内
- Can: 罐子被放入垃圾桶
- Lift: 方块被抓取并抬高到指定高度
- Square: 方形钉插入方形孔

**实验结果:** 所有 4 个任务均达到 **100% 成功率** (50 测试轨迹)

---

## 🎯 关键理论主张 (用于回复审稿人)

### 为什么是 Pareto 最优?

**Claim 1:** 我们的方法在计算-精度 Pareto 曲线上找到最优点

**证明逻辑:**
1. 固定 $k$ 形成一条 Trade-off 曲线 (Figure 2)
2. AdaStep 通过自适应 $k$ 突破了这条曲线 (位于左上角)
3. 数学上等价于求解: $\max k$ s.t. $\mathcal{E}(s_t, k) \leq \delta_{safe}$

---

### 为什么误差动力学有效?

**Claim 2:** Lipschitz 常数在不同状态类型之间差异显著

**实验支撑:**
- Free-space: $L_k \approx 0.01$ → $k=50$ 安全
- Contact: $L_k \approx 0.15$ → $k \leq 10$ 必要
- 差异 15× → 证明自适应的必要性

---

### 为什么聚类有效?

**Claim 3:** 状态流形聚类能够捕捉误差动力学的相似性

**Manifold Hypothesis:** 高维状态空间在低维流形上,相似特征 $z_i$ 对应相似 $L_k$

**实验验证:**
- Cluster 1 (自由空间): 平均 $k=50$
- Cluster 2 (接近): 平均 $k=35$
- Cluster 3 (接触): 平均 $k \leq 10$
- 簇间差异显著 → 证明聚类有效分离复杂度

---

## 🔄 与相关工作的数学对比

### vs. 固定步长 (Fixed Chunking)
- 他们: $k = \text{constant}$ (如 ACT 的 $k=10$)
- 我们: $k_t = h_\phi(s_t)$ (状态自适应)
- 优势: 消除 Trade-off

### vs. MPC (Model Predictive Control)
- 他们: 每步优化 $\min_{a_{t:t+H}} \sum \text{cost}(s, a)$ (计算密集)
- 我们: 轻量级 MLP $h_\phi$ (0.8ms vs 数秒)
- 优势: 实时性

### vs. 自适应控制 (Adaptive Control)
- 他们: 在线调整控制器参数 (需要系统模型)
- 我们: 离线学习 + 在线快速推理 (无模型)
- 优势: 通用性

---

## ✅ 公式完整性检查表

论文必须包含的公式 (按出现顺序):

- [ ] Eq. 1: 标准 Action Chunking 流程
- [ ] Eq. 2: k-step 累积误差 $\mathcal{E}(s_t, k)$
- [ ] Eq. 3: 约束优化目标 $k^* = \arg\max k$
- [ ] Eq. 4: 约束条件 $\mathcal{E}(s_t, k) \leq \delta_{safe}$
- [ ] Eq. 5: K-Means 聚类目标
- [ ] Eq. 6: 聚类误差度量 $\bar{E}_j(k)$ 和 $\sigma_j(k)$
- [ ] Eq. 7: Pareto 最优标签 $k_j^*$
- [ ] Eq. 8: MLP 架构 $h_\phi(z_t)$
- [ ] Eq. 9: 损失函数 $\mathcal{L}_{horizon}(\phi)$
- [ ] Eq. 10 (可选): 联合训练 $\mathcal{L}_{total}$

---

## 📝 LaTeX 技巧

### 公式编号示例
```latex
\begin{equation}
\mathcal{E}(s_t, k) = \sum_{i=1}^{k} \| \hat{a}_{t+i-1} - a^*_{t+i-1} \|_2
\label{eq:error_dynamics}
\end{equation}
```

### 引用方式
```latex
As defined in Eq.~\ref{eq:error_dynamics}, the cumulative error...
```

### 数学符号一致性
- 状态: $s_t$ 或 $\mathcal{S}$ (集合)
- 动作: $a_t$ 或 $\mathcal{A}$ (集合)
- 策略: $\pi_\theta$ (参数化)
- 预测器: $h_\phi$ (MLP)
- 簇: $\mathcal{C}_j$ (花体 C)
- 误差: $\mathcal{E}$ (花体 E)
- 损失: $\mathcal{L}$ (花体 L)

---

**提示:** 将此文档放在 Overleaf 项目旁边,写论文时随时查阅!
