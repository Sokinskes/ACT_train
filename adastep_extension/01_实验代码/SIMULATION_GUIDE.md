# AdaStep 真实仿真评估指南

> **关键学术问题**: 将"离线推理节省率"转化为"在线任务成功率"

## 📌 为什么需要真实仿真？

### 当前数据状态

✅ **已有数据（100%真实）**:
- 推理节省率: 80-90% (基于测试集统计)
- MLP准确率: 88-100% (真实预测性能)
- K值分布: 真实聚类结果

⚠️ **缺少数据（估计值）**:
- 任务成功率: 目前使用数学模型估计
- 与基线对比: 未在真实环境中验证
- 风险意识验证: 未在Square任务中实测

### 学术严谨性要求

**✅ 正确表述** (目前):
> "Given the time constraints, the success rates reported in Table X are **estimated based on offline metrics**, not online simulation."

**❌ 错误表述** (会被质疑):
> "We evaluated in MuJoCo simulation and achieved 90% success rate." ← 这是学术不端！

**✅ 理想表述** (完成仿真后):
> "We evaluated AdaStep in MuJoCo simulation across 200 rollouts (50 per task), achieving an average success rate of 92.7% with 85% inference reduction."

---

## 🚀 快速开始

### 步骤1: 环境诊断

运行诊断脚本，确保环境配置正确：

```bash
cd /home/yhj/桌面/ACT/adastep_extension/01_实验代码

# 检查Transport任务
python diagnostic_simulation.py --task transport

# 检查Square任务
python diagnostic_simulation.py --task square
```

**预期输出**:
```
✅ Robomimic 已安装
✅ MuJoCo 已安装
✅ 数据集格式正确
✅ 模型加载成功
✅ 环境创建成功
```

如果有任何❌，请先解决问题。

### 步骤2: 运行单任务仿真

```bash
# Transport任务 (验证高节省率场景)
python run_real_simulation.py \
    --task transport \
    --ckpt ../checkpoints/transport_mh/policy_best.ckpt \
    --data ../robomimic_data/transport/mh/low_dim_v141.hdf5 \
    --num_rollouts 50 \
    --save_results results_transport_real.pkl
```

**预期运行时间**: 约20-30分钟 (50条轨迹)

**预期输出**:
```
📊 仿真评估结果
任务: Transport
成功率: 92.3% (46/50)
平均推理次数: 12.3
平均步长 k: 48.5
推理节省率: 89.2%
```

### 步骤3: 运行所有任务

```bash
# 一键运行4个任务
python run_all_simulations.py
```

这会依次评估:
1. Transport (高效率场景)
2. Can (高效率场景)
3. Lift (中等效率场景)
4. Square (保守策略场景)

**总耗时**: 约1.5-2小时

---

## 📊 核心逻辑说明

### AdaStep在仿真中的运行机制

```python
# 伪代码展示核心逻辑
action_queue = []

while not done:
    # 如果队列空了，进行推理
    if len(action_queue) == 0:
        # 1. 准备输入
        qpos, image = prepare_observation(obs)
        
        # 2. AdaStep推理
        k_pred, action_sequence = policy.predict(qpos, image)
        
        # 3. 截取前k步
        action_queue = action_sequence[:k_pred]
        
        # 记录推理次数
        inference_count += 1
    
    # 从队列取动作执行
    action = action_queue.pop(0)
    obs, reward, done, info = env.step(action)
    env_step_count += 1
```

**关键点**:
- 每次推理产生k个动作
- 连续执行k步，期间不推理
- 推理节省 = 1 - (inference_count / env_step_count)

---

## 🎯 关键观察点

### Transport任务 (验证效率)

**预期行为**:
- 远离物体时: k ≈ 45-50 (大步快走)
- 接近物体时: k ≈ 20-30 (减速)
- 抓取/放置时: k ≈ 10-15 (精细控制)

**失败模式**:
- 如果成功率<85%: k太大，累积误差过多
- 如果推理节省<70%: k太小，策略过于保守

### Square任务 (验证安全性)

**预期行为**:
- 全程保持小k (5-10)
- 成功率应与ACT baseline相当 (>85%)

**失败模式**:
- 如果k>20: 算法未正确识别高风险
- 如果成功率<70%: 插孔精度不足

---

## 📁 输出文件说明

### 1. results_{task}_real.pkl

Python pickle格式，包含:
```python
{
    'task': 'transport',
    'success_rate': 92.3,
    'avg_inference_count': 12.3,
    'avg_k': 48.5,
    'rollouts': [...]  # 每条轨迹的详细数据
}
```

### 2. 使用结果

```python
import pickle

# 加载结果
with open('results_transport_real.pkl', 'rb') as f:
    results = pickle.load(f)

print(f"成功率: {results['success_rate']:.1f}%")
print(f"推理节省: {results['avg_inference_saving']:.1f}%")
```

---

## ⚠️ 常见问题

### Q1: ModuleNotFoundError: No module named 'robomimic'

**解决**:
```bash
pip install robomimic
```

### Q2: MuJoCo not found

**解决**:
```bash
# 安装MuJoCo
pip install mujoco-py  # 或
pip install mujoco
```

### Q3: 模型加载失败: KeyError

**原因**: 模型配置不匹配

**解决**:
1. 检查checkpoint中是否有'config'键
2. 手动指定配置参数（在run_real_simulation.py中）

### Q4: 环境创建失败

**解决**:
1. 确认HDF5文件包含'env_args'
2. 使用diagnostic_simulation.py诊断

### Q5: 仿真成功率过低（<50%）

**可能原因**:
1. 模型训练不充分
2. 观测预处理不匹配（图像归一化、qpos维度）
3. k值预测错误

**调试**:
```python
# 在run_rollout中添加调试信息
print(f"Step {step}: k={k_pred}, queue_len={len(action_queue)}")
```

---

## 📈 预期实验结果

### 理想情况 (论文级)

| 任务 | 成功率 | 推理节省 | 平均k |
|------|--------|---------|-------|
| Transport | 92% | 89% | 48.5 |
| Can | 93% | 88% | 47.2 |
| Lift | 91% | 81% | 26.3 |
| Square | 87% | 12% | 7.8 |

### 可接受情况 (会议级)

| 任务 | 成功率 | 推理节省 | 平均k |
|------|--------|---------|-------|
| Transport | >85% | >70% | >30 |
| Can | >85% | >70% | >30 |
| Lift | >80% | >60% | >20 |
| Square | >75% | <30% | <15 |

### 需要重新训练 (不可接受)

- 任何任务成功率<70%
- Transport/Can推理节省<50%
- Square的k>25 (说明未识别风险)

---

## 📝 将结果写入论文

### 替换估计值

**修改前** (CRITICAL_ISSUES_ANALYSIS.md):
```latex
\caption{Task Success Rate Comparison (Estimated)}
AdaStep & 93\% (est.) & ...
```

**修改后**:
```latex
\caption{Task Success Rate Comparison}
AdaStep & 92.3\% & ...
\footnote{Based on 50 rollouts in MuJoCo simulation.}
```

### 添加实验描述

**Section 4.2: Success Rate Validation**
> We evaluated AdaStep in the Robomimic simulation environment using MuJoCo physics engine. For each task, we conducted 50 rollouts starting from random initial states. The success criterion follows Robomimic's standard: the task is considered successful if the object reaches the target within tolerance.

> Results show that AdaStep achieves an average success rate of 92.7% across four tasks, with only 2.3% degradation compared to the ACT baseline (95%). Notably, in the high-precision Square task, AdaStep automatically adopts a conservative strategy (avg. k=7.8), maintaining 87% success rate while sacrificing efficiency gain.

---

## 🎯 时间规划

### 最快路径 (半天)

1. **1小时**: 环境诊断 + 单任务测试
2. **2小时**: 运行4个任务 (各50轨迹)
3. **1小时**: 分析结果 + 更新论文

### 完整路径 (1天)

1. **2小时**: 环境配置 + 代码调试
2. **3小时**: 运行4个任务 (各100轨迹)
3. **2小时**: 可视化 + 结果分析
4. **1小时**: 论文更新 + 表格制作

---

## ✅ 完成检查清单

- [ ] 诊断脚本全部通过
- [ ] Transport成功率 >85%
- [ ] Square的k <15 (证明风险意识)
- [ ] 4个任务平均成功率 >85%
- [ ] 推理节省率与离线数据一致 (±5%)
- [ ] 结果已保存为.pkl文件
- [ ] 论文表格已更新
- [ ] 删除所有"estimated"标注

---

## 📚 相关文件

- `run_real_simulation.py`: 主评估脚本
- `run_all_simulations.py`: 批量评估脚本
- `diagnostic_simulation.py`: 环境诊断工具
- `../CRITICAL_ISSUES_ANALYSIS.md`: 问题分析文档

---

**最后提醒**: 这一步是将论文从"优秀"提升到"卓越"的关键！一旦完成，你的工作就是完整的、可复现的、学术严谨的研究。

**预祝成功！** 🚀
