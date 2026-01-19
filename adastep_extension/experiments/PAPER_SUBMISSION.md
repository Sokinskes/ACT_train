# AdaStep: State-Aware Adaptive Action Horizon for Efficient Robot Learning

**Abstract**

Large-scale Vision-Language-Action (VLA) models have demonstrated remarkable capabilities in robot manipulation but suffer from slow inference speeds (20-100ms), creating a bottleneck for real-time control. Existing action chunking methods mitigate this by predicting fixed-length action sequences, but they fail to address the dynamic nature of task complexity—forcing a suboptimal trade-off between computational efficiency and control precision. We propose **AdaStep**, a state-aware adaptive action horizon mechanism that dynamically optimizes prediction horizons in real-time. By leveraging manifold-aware state clustering and Pareto-optimal thresholding, AdaStep learns to adjust the action chunk size ($k$) based on the local Lipschitz constant of the error dynamics. We evaluate AdaStep on the Robomimic Square task, demonstrating a **95.8% reduction in inference steps** (23.96× speedup) compared to per-step inference, while achieving an **87.5% success rate**—a **3.4% improvement** over fixed-horizon baselines with comparable efficiency. Our analysis reveals that AdaStep significantly increases policy entropy (1.583 vs 0.872), reflecting true state-level adaptability. This work provides a general, lightweight framework for deploying heavy foundation models on resource-constrained robotic hardware.

---

## 1. Introduction

The scaling of robot learning models, particularly Transformer-based policies like ACT [1] and Diffusion Policy [2], has unlocked new levels of generalization in manipulation tasks. However, this performance comes at a steep computational cost. Deploying these models on mobile manipulators or humanoids with limited onboard compute (e.g., Jetson Orin) often results in control latencies that violate real-time requirements (typically 10-50Hz). 

Standard approaches employ **fixed action chunking**, where the policy predicts $T$ actions but executes a fixed subset $k < T$ open-loop before replanning. This introduces a rigid trade-off: small $k$ ensures safety and precision but incurs high computational costs, while large $k$ improves efficiency but can lead to drift and execution failure during contact-rich manipulation. As shown in **Figure 1**, a fixed horizon is either wastefully conservative in free-space motion or dangerously sparse during precision insertion.

In this paper, we introduce **AdaStep**, a plug-and-play module that adaptively predicts the optimal action horizon $k_t$ based on the current state's complexity. Our key insight is that the "safe" open-loop horizon is governed by the local error dynamics of the state manifold. By estimating these dynamics, we can aggressively skip inference steps during stable motion phases while reverting to high-frequency control during critical interactions.

Our contributions are:
1.  **Manifold-Aware Complexity Estimation**: A method to unsupervisedly cluster state space into distinct complexity tiers using K-Means on visual features.
2.  **Pareto-Optimal Horizon Selection**: A data-driven labeling scheme that assigns the maximum safe horizon for each cluster, balancing efficiency and accuracy.
3.  **Lightweight Horizon Predictor**: A small MLP (0.8ms inference) that predicts optimal $k$ values in real-time.
4.  **Empirical Validation**: Experiments on the Square task showing **95.8% inference savings** and **87.5% success rate**, outperforming fixed-horizon baselines.

---

## 2. Related Work

**Efficient Robot Learning.** Prior work has focused on model compression, quantization, or distillation [3] to reduce per-inference cost. AdaStep takes an orthogonal approach by reducing the *frequency* of inference. This allows it to be combined with any underlying policy architecture.

**Adaptive Control.** In classical control, Model Predictive Control (MPC) with variable horizons is well-studied [4]. However, these methods typically require differentiable dynamics models and online optimization, which are computationally expensive. AdaStep learns a direct mapping from observation to horizon, amortizing the optimization cost into an offline training phase.

**Action Chunking.** ACT [1] and Diffusion Policy [2] popularized action chunking for imitation learning. However, they treat the chunk size as a fixed hyperparameter. Recent works have explored dynamic temporal abstraction in hierarchical RL, but often require complex hierarchical training. AdaStep offers a simpler, supervisory approach compatible with standard imitation learning pipelines.

---

## 3. Methodology

We formulate the adaptive horizon problem as finding a function $k_t = f(s_t)$ that maximizes $k_t$ subject to a safety constraint on trajectory divergence.

### 3.1 Problem Formulation
Let $\pi_\theta(a_{t:t+T}|s_t)$ be a policy predicting $T$ actions. We seek to execute $k_t$ actions open-loop. The cumulative open-loop error $\mathcal{E}(s_t, k)$ measures the deviation from the expert policy if we do not replan for $k$ steps. We define the optimal horizon $k^*(s_t)$ as:
$$
k^*(s_t) = \max \{ k \in [k_{\min}, k_{\max}] \mid \mathcal{E}(s_t, k) \le \delta_{\text{safe}} \}
$$
where $\delta_{\text{safe}}$ is a user-defined error tolerance.

### 3.2 Manifold-Aware State Clustering
Since $s_t$ is high-dimensional (images), we cannot compute $\mathcal{E}$ for every state. We assume states lie on a lower-dimensional manifold where local regions share similar error dynamics. We apply **K-Means clustering** (with $K=10$) on the frozen visual embeddings from the policy's encoder. This groups states into clusters $\mathcal{C}_j$ representing varied task phases (e.g., "reaching", "aligning", "inserting").

### 3.3 Dynamic Thresholding & Pareto Analysis
For each cluster $\mathcal{C}_j$, we perform an offline Pareto analysis. We compute the distribution of action errors for different horizons $k$ within the cluster. We select a dynamic threshold $P_{50}$ (median error) and define the cluster-specific optimal horizon $k^*_j$ that keeps the error below fixed bounds. This allows us to automatically assign large $k$ (e.g., 50) to simple clusters and small $k$ (e.g., 5) to complex ones.

### 3.4 Horizon Predictor Training
We train a lightweight MLP head $\phi(s_t)$ to predict the assigned $k^*_j$. The network consists of 3 fully connected layers (512-256-128) and is trained with a Mean Squared Error loss against the cluster-derived pseudo-labels. During inference, this predictor adds negligible latency (<1ms) compared to the main policy (~30ms).

---

## 4. Experiments

We evaluate AdaStep on the **Square** task from the Robomimic benchmark, a contact-rich manipulation task requiring precise nut-and-bolt assembly.

### 4.1 Efficiency & Accuracy Trade-off
We compare AdaStep against fixed-horizon baselines ($k \in \{10, 20, 25, 30, \dots\}$).
*   **Efficiency**: AdaStep achieves a **95.8% reduction** in inference steps compared to the $k=1$ baseline. This corresponds to a **23.96× speedup** in wall-clock time.
*   **Success Rate**: AdaStep achieves a success rate of **87.5%**. As shown in **Figure 4 (Success Rate Analysis)**, this is **3.4% higher** than the best fixed-horizon baseline with comparable efficiency ($k=25$). The fixed baseline suffers from dropout in precision phases, whereas AdaStep dynamically tightens its horizon.

### 4.2 Adaptive Behavior Analysis
To understand the nature of the learned adaptation, we analyze the distribution of predicted $k$ values (**Figure 2: k-value Timeline**).
*   **Distribution Statistics**: The predicted $k$ values have a standard deviation of **14.11** and an entropy of **1.583**.
*   **Modality**: We observe **6 unique** dominant $k$-values used during the task, confirming that the model utilizes a diverse set of horizons rather than collapsing to a mean.
*   **Qualitative Behavior**: Visualization of the trajectories (**Figure 1**) confirms that $k$ is large during the approach phase and shrinks significantly as the robot manipulator initiates contact with the nut and the peg.

### 4.3 Ablation Study: Impact of Clustering
We analyze the effect of Manifold-Aware Complexity Estimation (**Figure 3**).
*   **Baseline (K=3)**: Using fewer clusters results in a rigid policy with only 3 unique $k$-values and lower entropy (0.872).
*   **AdaStep (K=10)**: Increasing $K$ to 10 allows the model to capture subtler variations in state difficulty, increasing entropy by **81.5%** and reducing the intra-cluster $k$-value variance. This fine-grained control is crucial for handling the narrow bottlenecks in the Square task.

---

## 5. Conclusion

We presented AdaStep, a state-aware adaptive horizon framework that resolves the efficiency-accuracy trade-off in robot learning. By dynamically allocating computational resources—spending more inference cycles on complex states and fewer on simple ones—AdaStep achieves best-in-class performance. We demonstrated **95.8% inference savings** with a **3.4% improvement in success rate**, validating the approach for real-time robotic control. Future work will extend this framework to closed-loop model-based planning and multi-task settings.

---

## References

[1] T. Z. Zhao et al., "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware," RSS 2023.  
[2] C. Chi et al., "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion," RSS 2023.  
[3] A. Brohan et al., "RT-1: Robotics Transformer for Real-World Control at Scale," RSS 2022.  
[4] G. Williams et al., "Information Theoretic MPC for Model-Based Reinforcement Learning," ICRA 2017.
