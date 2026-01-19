# AdaStep论文 - Introduction & Method完整草稿

> **撰写日期**: 2026-01-13  
> **论文标题**: AdaStep: Adaptive Action Chunking for Efficient Robot Control  
> **目标会议**: RSS / CoRL / ICRA

---

## 1. Introduction

Recent advances in vision-language-action (VLA) models have demonstrated remarkable success in robot manipulation [cite: RT-1, RT-2, OpenVLA]. However, these large-scale models incur substantial computational overhead—requiring 20-100ms per inference on high-end GPUs. For real-time robot control at 10-30Hz, this poses a critical bottleneck, particularly for deployment on resource-constrained platforms such as mobile manipulators or humanoid robots.

**Action chunking** has emerged as a promising solution: instead of predicting one action per timestep, modern policies generate multi-step action sequences (typically 10-100 steps) and execute them open-loop before re-planning [cite: ACT, Diffusion Policy]. This amortizes the inference cost over multiple timesteps, achieving 10-100× speedups. However, existing methods use **fixed chunking horizons** across all states and tasks, leading to a fundamental trade-off:
- **Small horizons** ($k \approx 1-5$): Safe for precision tasks but computationally expensive  
- **Large horizons** ($k \approx 50-100$): Efficient for simple motions but risky for contact-rich manipulation

We ask: *Can we adaptively adjust the chunking horizon based on state complexity?*

**Key Insight.** Not all robot states require the same planning frequency. During long-distance reaching, the robot can safely execute 50-step chunks without visual feedback. But when inserting a peg into a hole, frequent re-planning (every 5-10 steps) becomes essential. A **state-aware adaptive horizon** could provide "the best of both worlds"—maximum efficiency during simple motions, automatic safety during precision operations.

**Our Approach: AdaStep.** We propose a lightweight horizon predictor that learns to output $k_t$ (the number of steps to execute before next inference) conditioned on the current state. The core challenge is obtaining training labels: ground-truth $k$ values are not available in standard robot datasets. We address this via **Pareto-optimal clustering**:

1. Apply K-Means to cluster states by complexity (e.g., "far from object", "approaching", "in contact")  
2. For each cluster, perform Pareto analysis: find the maximum $k$ that keeps cumulative action error below a threshold $\epsilon$
3. Use cluster-assigned $k$ values as pseudo-labels to train a small MLP ($<$1M parameters)

At deployment, this MLP adds negligible overhead (0.8ms) while enabling **dynamic horizon selection** from $k=5$ (precision) to $k=50$ (efficiency).

**Contributions:**
1. A principled framework for adaptive action chunking via Pareto-optimal clustering  
2. Empirical validation on four manipulation tasks: 94-98\% inference reduction with 100\% success rate
3. Demonstration that a 3-layer MLP can learn task-aware horizon policies without explicit supervision
4. Preliminary real-world validation showing automatic horizon adaptation on a UR5 robot

---

## 2. Related Work

### 2.1 Action Chunking in Robot Learning

**Fixed-horizon approaches.** ACT [cite] predicts 100-step action sequences using a CVAE, executing them with a fixed stride of $k=10$. Diffusion Policy [cite] uses 16-step chunks with overlapping execution. 3D Diffusion Policy [cite] extends this to 3D representations. While these methods achieve significant speedups (10-100×), they do not adapt $k$ to task difficulty.

**Receding horizon control.** Model Predictive Control (MPC) [cite] dynamically adjusts planning horizons but relies on differentiable dynamics models. Recent work combines MPC with learned models [cite: MPPI], but computational cost remains high (10-100× slower than open-loop execution). Our approach learns a direct state→horizon mapping, avoiding online optimization.

### 2.2 State Complexity Estimation

**Uncertainty-based methods.** Ensemble models [cite] or Bayesian NNs [cite] estimate epistemic uncertainty, which can inform re-planning frequency. However, these require training multiple networks or costly sampling. We use deterministic clustering with offline Pareto analysis, making deployment lightweight.

**Manual heuristics.** Some works [cite] manually define "stages" (approach → grasp → lift) and assign different $k$ per stage. This requires domain knowledge and does not generalize. AdaStep discovers complexity tiers automatically from data.

### 2.3 Efficient Robot Inference

**Model distillation** [cite: RT-X] compresses large VLA models into smaller networks. **Quantization** [cite] reduces precision. **Neural architecture search** [cite] finds efficient architectures. These are orthogonal to our approach—AdaStep reduces inference *frequency* rather than per-step cost, and can be combined with any policy architecture.

---

## 3. Method

### 3.1 Problem Formulation

We consider a Markov Decision Process (MDP) with states $s_t \in \mathcal{S}$, actions $a_t \in \mathcal{A}$, and a learned action-chunking policy $\pi_\theta(a_{t:t+T} | s_t)$ that predicts $T$ future actions (e.g., $T=100$ for ACT). Standard practice executes these actions open-loop with a fixed stride $k$:

$$
a_t, a_{t+1}, \ldots, a_{t+k-1} \sim \pi_\theta(a_{t:t+T} | s_t), \quad s_{t+k} \leftarrow \text{env.step}(a_{t:t+k-1})
$$

Then inference is triggered again at $t+k$. Fixed $k$ wastes computation in simple states (where $k=50$ would suffice) and risks failures in complex states (where $k=5$ is safer).

#### Error Dynamics Modeling

We formalize this trade-off through **error dynamics analysis**. Define the **k-step cumulative divergence function** $\mathcal{E}(s_t, k)$ as the trajectory deviation when executing $k$ steps open-loop:

$$
\mathcal{E}(s_t, k) = \sum_{i=1}^{k} \| \hat{a}_{t+i-1} - a^*_{t+i-1} \|_2
$$

where $\hat{a}_{t+i-1}$ is the predicted action and $a^*_{t+i-1}$ is the expert action at the actual state reached after $i-1$ steps. This function quantifies how quickly open-loop execution diverges from the expert trajectory.

**Physical Interpretation:** The divergence rate $\partial \mathcal{E} / \partial k$ varies dramatically across state types:
- **Free-space motion:** Error grows sub-linearly (small Lipschitz constant $L_k \approx 0.01$), enabling safe long horizons ($k=50$)
- **Contact-rich states:** Error grows super-linearly or exponentially (large Lipschitz constant $L_k \approx 0.15$), requiring frequent replanning ($k=6-10$)

#### Constrained Optimization Objective

We formulate adaptive action horizon selection as a **constrained optimization problem** on the state manifold:

$$
k^*(s_t) = \arg\max_{k \in [k_{\min}, k_{\max}]} \quad k
$$
$$
\text{subject to} \quad \mathcal{E}(s_t, k) \leq \delta_{safe}
$$

where $\delta_{safe}$ is the maximum tolerable trajectory error (safety threshold). This formulation explicitly seeks the **Pareto frontier** between two competing objectives:
1. **Computational efficiency:** $\mathcal{C}(k) \propto 1/k$ (fewer inferences)
2. **Execution safety:** $\mathcal{R}(s_t, k) = \mathcal{E}(s_t, k)$ (bounded trajectory error)

**Intuition:** For each state, we select the largest horizon $k$ that keeps trajectory error within safety bounds, minimizing inference frequency while preserving task success. This is inherently a Pareto-optimal solution on the efficiency-accuracy trade-off curve.

**Adaptive horizon formulation.** We augment the policy with a horizon predictor $h_\phi: \mathcal{S} \to [k_{\min}, k_{\max}]$ that learns to approximate $k^*(s_t)$ from offline data. The modified control loop becomes:

$$
k_t = h_\phi(s_t), \quad a_{t:t+k_t-1} \sim \pi_\theta(a_{t:t+T} | s_t), \quad s_{t+k_t} \leftarrow \text{env.step}(a_{t:t+k_t-1})
$$

**Design goals:**  
1. $h_\phi$ must be lightweight (inference $<$ 1ms)  
2. $h_\phi$ should be learned from offline data (no additional environment interaction)  
3. $h_\phi$ should generalize across tasks without manual tuning

---

### 3.2 Pareto-Optimal Horizon Assignment

The key challenge is obtaining labels $\{(s_i, k_i^*)\}$ for training $h_\phi$. We propose a two-stage approach:

#### Stage 1: State Manifold Clustering

Since the state space $\mathcal{S}$ is continuous and high-dimensional, directly computing $k^*(s_t)$ for every state is intractable. We leverage the **manifold hypothesis**: states with similar dynamics share similar error characteristics.

Given a dataset $\mathcal{D} = \{(s_i, a_{i:i+T})\}_{i=1}^N$ of state-action trajectories, we extract visual features $z_i = E_{vision}(s_i)$ using the pre-trained ACT encoder, then perform K-Means clustering:

$$
\min_{\{\mu_j\}} \sum_{j=1}^K \sum_{z_i \in \mathcal{C}_j} \| z_i - \mu_j \|^2
$$

where $\mu_j$ is the centroid of cluster $\mathcal{C}_j$. Each cluster represents a "complexity tier" with homogeneous error dynamics (e.g., Cluster 1 = "free-space", Cluster 2 = "approaching", Cluster 3 = "contact"). Empirically, $K=3$ provides good balance.

**Key Assumption:** States in the same cluster $\mathcal{C}_j$ exhibit similar Lipschitz constants $L_k^{(j)}$, thus they can share the same optimal horizon $k_j^*$.

#### Stage 2: Pareto Frontier Labeling

For each cluster $\mathcal{C}_j$, we solve the constrained optimization problem to find the *maximum safe horizon* $k_j^*$. We approximate $\mathcal{E}(s_t, k)$ using the **k-step action variance** as a proxy for trajectory complexity:

$$
\bar{E}_j(k) = \frac{1}{|\mathcal{C}_j|} \sum_{s_i \in \mathcal{C}_j} \| \Delta a_{i:i+k} \|_2, \quad \sigma_j(k) = \text{std}(\{ \| \Delta a_{i:i+k} \|_2 \mid s_i \in \mathcal{C}_j \})
$$

where $\Delta a = a_{i+1} - a_i$ measures action velocity changes. High $\bar{E}_j(k)$ indicates complex, rapidly-varying actions that require frequent replanning.

The optimal horizon $k_j^*$ is defined as:

$$
k_j^* = \max \{ k \in [k_{\min}, k_{\max}] \mid \bar{E}_j(k) + \lambda \cdot \sigma_j(k) < \delta_{safe} \}
$$

where $\lambda$ is a safety coefficient (we use $\lambda = 1.0$), and $\delta_{safe} = 0.02$ is the error tolerance threshold determined by cross-validation. This formulation incorporates variance $\sigma_j(k)$ to ensure robustness across the cluster.

**Pareto Optimality:** Among all horizons $k$ satisfying the error constraint $\bar{E}_j(k) + \lambda \sigma_j(k) < \delta_{safe}$, we select the maximum $k_j^*$ to minimize computational cost. This solution lies on the Pareto frontier of the efficiency-safety trade-off.

**Output:** Cluster-level horizon assignments $\{(\mathcal{C}_j, k_j^*)\}_{j=1}^K$, which are then broadcasted to all states in each cluster: $k_i^* = k_j^*$ if $s_i \in \mathcal{C}_j$.

---

### 3.3 Online Horizon Predictor Learning

With pseudo-labels $\{(s_i, k_i^*)\}$ obtained from Pareto frontier analysis, we train a lightweight neural network $h_\phi: \mathcal{S} \to [k_{\min}, k_{\max}]$ to approximate the optimal horizon function $k^*(s_t)$.

**Architecture:** We use a 3-layer MLP that processes visual features $z_t = E_{vision}(s_t)$ from the frozen ACT encoder:

$$
h_\phi(z_t) = \sigma(\mathbf{W}_3 \cdot \text{ReLU}(\mathbf{W}_2 \cdot \text{ReLU}(\mathbf{W}_1 \cdot z_t)))
$$

with hidden dimensions [512, 256, 128], ReLU activations, and Sigmoid output layer $\sigma(\cdot)$ that produces normalized predictions $\hat{k}_i \in [0, 1]$.

**Loss function:** We formulate this as a regression problem with normalized targets $\tilde{k}_i = (k_i^* - k_{\min})/(k_{\max} - k_{\min})$:

$$
\mathcal{L}_{horizon}(\phi) = \frac{1}{N} \sum_{i=1}^N (\hat{k}_i - \tilde{k}_i)^2 + \lambda \|\phi\|_2^2
$$

Where the L2 regularization term (with $\lambda = 1 \times 10^{-5}$) prevents overfitting.

**Joint Training (Optional):** For end-to-end optimization, the horizon predictor can be trained jointly with the ACT policy:

$$
\mathcal{L}_{total} = \mathcal{L}_{ACT}(\theta) + \alpha \cdot \mathcal{L}_{horizon}(\phi)
$$

where $\alpha = 0.1$ is the weighting coefficient. Since $h_\phi$ reuses the pre-trained feature encoder $E_{vision}$, the additional inference overhead is only $\mathcal{O}(1)$ with 0.8ms latency.

**Training details:** Adam optimizer, learning rate $1 \times 10^{-4}$, batch size 64, 100 epochs with early stopping (patience=10). Training converges in ~5 minutes on a single RTX 3090 GPU.

At inference, we denormalize the output:

$$
k_t = \lfloor h_\phi(s_t) \cdot (k_{\max} - k_{\min}) + k_{\min} \rfloor
$$

---

### 3.4 Integration with Existing Policies

AdaStep is **policy-agnostic** and can be integrated with any action-chunking method:

**With ACT [cite]:**
```python
state = get_current_state()
k = horizon_predictor(state)  # 0.8ms
actions = act_policy(state)   # 31.0ms (encoder + decoder)
execute_actions(actions[:k])   # Open-loop for k steps
```

**With Diffusion Policy [cite]:**
```python
state = get_current_state()
k = horizon_predictor(state)
actions = diffusion_policy.sample(state)  # ~100ms
execute_actions(actions[:k])
```

The key advantage is that $h_\phi$ adds negligible overhead (0.8ms) compared to the base policy (31-100ms), so the speedup is approximately:

$$
\text{Speedup} = \frac{T_{\text{baseline}}}{T_{\text{AdaStep}}} \approx \frac{T_{\text{traj}} / k_{\text{fixed}}}{T_{\text{traj}} / \bar{k}} = \frac{\bar{k}}{k_{\text{fixed}}}
$$

where $\bar{k}$ is the average horizon under AdaStep (typically 30-50), and $k_{\text{fixed}}$ is the baseline stride (typically 1-10).

---

### 3.5 Theoretical Justification

**Why does this work?** The key insight is that state clustering implicitly discovers a **hierarchical task structure**:
- **Cluster 1 (Simple):** Long-distance motions with smooth dynamics → Assign large $k$ (e.g., 50)  
- **Cluster 2 (Moderate):** Approaching objects, requires moderate precision → Assign medium $k$ (e.g., 20-30)  
- **Cluster 3 (Complex):** Contact-rich manipulation, high sensitivity → Assign small $k$ (e.g., 5-10)

The MLP $h_\phi$ learns a *soft* version of this clustering by regressing continuous $k$ values. Empirically, we find that $h_\phi$ generalizes well to unseen states (96.3\% test accuracy), suggesting that complexity is a smooth function of state features.

**Pareto optimality.** Our horizon assignment is Pareto-optimal in the sense that for each complexity tier, we select the *largest* safe horizon. This avoids overly conservative strategies (e.g., always using $k=5$) that waste computation.

---

### 3.6 Practical Considerations

**Handling action sequence boundaries.** If the policy predicts $T=100$ steps but we've already executed 95 steps, we cap $k_t = \min(k_t, T - t_{\text{current}})$ to avoid out-of-bounds indexing.

**Warm-up period.** For the first 5 steps of each episode, we use a conservative $k=k_{\min}$ to allow the policy to "settle" from the initial state.

**Safety fallback.** If the predicted horizon exceeds a user-defined maximum (e.g., $k_{\max}^{\text{safe}} = 30$ for precision tasks), we clip it. This provides an additional safety layer for deployment.

---

## Pseudo-code: AdaStep Training Pipeline

```python
# Stage 1: Clustering and Pareto Analysis
states, actions = load_dataset()
clusters = KMeans(n_clusters=3).fit(states)

optimal_k = {}
for cluster_id in range(3):
    cluster_states = states[clusters.labels_ == cluster_id]
    cluster_actions = actions[clusters.labels_ == cluster_id]
    
    # Pareto analysis
    k_opt = find_max_k_below_error_threshold(
        cluster_actions, error_threshold=0.02
    )
    optimal_k[cluster_id] = k_opt

# Assign labels
k_labels = [optimal_k[label] for label in clusters.labels_]

# Stage 2: Train MLP
horizon_predictor = MLP(input_dim=state_dim, hidden_dims=[512, 256, 128])
train(horizon_predictor, states, k_labels, epochs=100)

# Deployment
def robot_control_loop():
    state = get_state()
    k = horizon_predictor(state)
    actions = act_policy(state)
    execute_open_loop(actions[:k])
```

---

## Summary of Method Contributions

1. **Pareto-optimal clustering** provides a principled way to assign horizons without manual labeling  
2. **Lightweight MLP** ($<$1M params) enables real-time inference on edge devices  
3. **Policy-agnostic** design works with any action-chunking method (ACT, Diffusion Policy, etc.)  
4. **Automatic task adaptation** emerges from data-driven clustering, no task-specific tuning needed

---

**Next**: See `PAPER_DRAFT_RESULTS_SECTION.md` for experimental validation

---

## Word Count

- Introduction: ~600 words  
- Related Work: ~350 words  
- Method: ~1,200 words  
- **Total**: ~2,150 words

Combined with Results (2,100 words), the paper is approximately **4,250 words**, suitable for RSS (8 pages), CoRL (8 pages), or ICRA (6-8 pages with supplementary material).

---

## Writing Style Notes

- ✅ Clear motivation (efficiency vs. safety trade-off)  
- ✅ Concrete running example (peg-in-hole vs. reaching)  
- ✅ Mathematical formulation without excessive complexity  
- ✅ Pseudo-code for reproducibility  
- ✅ Practical deployment considerations  
- ✅ Comparison with fixed-horizon baselines in Related Work

---

**Status**: ✅ Complete Introduction + Method draft  
**Next Step**: Combine with Results and format in LaTeX
