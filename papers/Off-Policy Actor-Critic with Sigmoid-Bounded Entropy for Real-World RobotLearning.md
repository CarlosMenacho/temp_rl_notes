## 1. Overview & Motivation

This paper addresses a persistent gap in robot learning: deploying reinforcement learning (RL) directly in the real world remains costly, data-hungry, and unstable. While prior work mitigates these issues via large offline datasets or pretrained VLA (Vision-Language-Action) models, both approaches impose heavy infrastructure requirements. SigEnt-SAC targets a more constrained and practical regime — learning from scratch using **only a single expert trajectory**, raw visual observations from a single camera, and sparse binary rewards.

**Core claim:** Two targeted algorithmic modifications to the standard Soft Actor-Critic (SAC) framework are sufficient to enable stable, sample-efficient policy learning in real-world robotics across diverse robot morphologies, without any large-scale pretraining.

---

## 2. Problem Statement

### 2.1 Identified Failure Modes in Existing Methods

**Negative Entropy Collapse in Conservative Q-Learning:**  
Standard SAC uses entropy as a regularizer: $\hat{Q}(s,a) - \alpha \log \pi(a|s)$. When the policy becomes confident (low variance), $\log \pi(a|s)$ becomes very negative, causing the entropy term to dominate the Q-value estimate. This creates a misleading Q-landscape where out-of-distribution (OOD) actions appear to have higher Q-values than in-distribution ones, pushing policy improvement toward the action boundary — precisely where the robot has no experience.

**Q-Function Oscillations in Conservative Methods:**  
Conservative Q-learning (CQL, Cal-QL) penalizes OOD actions but is susceptible to oscillation, particularly in one-shot or few-shot settings where the replay buffer is sparsely populated. These oscillations translate to erratic hardware behavior and increased mechanical wear.

**Dependency on Large Datasets:**  
Offline-to-online methods (IQL, AWAC, Cal-QL) are designed for large, high-quality offline datasets. In the one-shot setting they either fail entirely or exhibit severe performance degradation.

---

## 3. Method: SigEnt-SAC

SigEnt-SAC modifies standard SAC with two complementary mechanisms:

### 3.1 Sigmoid-Bounded Entropy (Core Contribution)

**Motivation:** Replace the standard entropy term (which can be negative and unbounded) with a bounded, strictly positive alternative.

**Definition:**  
For a tanh-squashed Gaussian policy $a = \tanh(x)$, where $x = \mu_\theta(s) + \sigma_\theta(s) \odot \epsilon$, define the per-dimension surprisal as:

$$s_i = -\log \pi_{\theta,i}(a_i|s)$$

The bounded entropy contribution per dimension is:

$$h_i(s_i) = h_{\max} \cdot \sigma\left(\frac{s_i - m}{t}\right)$$

The total sigmoid-bounded entropy is:

$$H_{\text{sig}}(s, a) = \sum_{i=1}^{d} h_i(s_i)$$

**Parameters:**

- $h_{\max} > 0$: maximum per-dimension entropy score
- $m$: center offset
- $t > 0$: temperature controlling transition steepness
- By construction: $H_{\text{sig}}(s, a) \in (0, d \cdot h_{\max})$

**Modified Q-update:**

$$\hat{Q}_{k+1}(s,a) = (1-\alpha)\hat{Q}_k(s,a) + \alpha\left[r(s,a) + \gamma \mathbb{E}_{a'\sim\pi}\left[\hat{Q}_k(s',a') + H_{\text{sig}}(s',a')\right]\right]$$

**Effect:** The bounded entropy creates a "bowl-shaped" Q-landscape with a clear high-value region near the state-visit distribution, eliminating the pathological boundary-maximization behavior of standard entropy.

---

### 3.2 Gated Behavior Cloning (GBC)

**Motivation:** Use the single expert trajectory as a corrective signal, but only when the policy has drifted significantly — avoiding over-constraint that would prevent improvement beyond the demonstration.

**Gating mask:**

$$p_{\text{gate}}(s, a_{\text{exp}}) = \mathbb{I}\left[|a_{\text{mean}}(s) - a_{\text{exp}}|_2 > \varepsilon\right]$$

where $a_{\text{mean}}(s) = \tanh(\mu_\theta(s))$ is the deterministic policy mean.

**Unified policy objective:**

$$J(\pi_\theta) = \mathbb{E}_{s\sim\mathcal{D}, a\sim\pi_\theta(\cdot|s)}\left[Q^k(s,a) + \alpha H_{\text{sig}}(s,a)\right] - \lambda \mathbb{E}_{(s,a_{\text{exp}})\sim\mathcal{D}_{\text{exp}}}\left[p_{\text{gate}}(s,a_{\text{exp}}) \cdot |a_{\text{mean}}(s) - a_{\text{exp}}|_2^2\right]$$

**Behavior:** The BC term activates only when policy deviation exceeds threshold $\varepsilon$. At convergence (or when the policy improves beyond the demonstration), the gate closes, allowing the policy to adapt freely.

---

### 3.3 Critic Loss

**Entropy-augmented Bellman target:**

$$y = r + (1-d)\gamma\left[\min_{i=1,2} Q'_i(s',a') + \alpha H_{\text{sig}}(s',a')\right]$$

**Base TD loss:**

$$\mathcal{L}^{(i)}_{\text{TD}} = \mathbb{E}_{\mathcal{D}_{\text{buf}}}\left[(Q_i(s,a) - y)^2\right]$$

**CQL-style OOD regularizer:**

$$\mathcal{L}^{(i)}_{\text{CQL}} = \mathbb{E}_{(s,a)\sim\mathcal{D}_{\text{buf}}}\left[\beta \log \sum_{\tilde{a} \in {a} \cup \mathcal{A}_{\text{ood}}(s)} \exp\left(\frac{1}{\beta}Q_i(s,\tilde{a})\right) - Q_i(s,a)\right]$$

**Final critic objective:**

$$\mathcal{L}_{Q_i} = \mathcal{L}^{(i)}_{\text{TD}} + \lambda_{\text{ood}} \mathcal{L}^{(i)}_{\text{CQL}}$$

---

### 3.4 Algorithm (SigEnt-SAC Training)

```
Input: Replay buffer D_buf, expert buffer D_exp, steps N, batch size B
Initialize: Policy ϕ, critics θ1/θ2, target networks θ'1/θ'2

Phase 1: Collect one expert trajectory → D_exp

Phase 2: for k = 1 to N:
  - Collect (s, a, r, s') → D_buf
  - if |D_buf| ≥ B:
    - Sample batches from D_buf and D_exp
    - Compute H_sig for next-state actions
    - Update critics using Eq. 2 (sigmoid entropy Bellman backup)
    - Update policy maximizing Eq. 4 (gated BC + sigmoid entropy)
    - Soft update target networks
```

---

## 4. Experimental Setup

### 4.1 Baselines

- **Cal-QL / CQL**: Conservative Q-learning methods with offline pre-training
- **RLPD**: Large Q-ensemble + LayerNorm for online fine-tuning
- **AWAC**: Q-weighted behavior cloning
- **IQL**: Implicit Q-learning with expectile regression
- **SAC (w/ Prior)**: Standard SAC initialized with expert data

### 4.2 Benchmarks

**Simulation:** D4RL Adroit (door, hammer, pen) and Kitchen tasks in the one-shot setting (1 expert trajectory, 1M environment steps).

**Real-world (4 tasks across 4 embodiments):**

|Task|Platform|Challenge|
|---|---|---|
|Push-Cube|RealMan RM65-B manipulator|Spatial randomization, sparse reward|
|Ball-to-Goal|AgileX LIMO Pro / TurtleBot3|Dynamic obstacles, movable objects|
|Slalom|Unitree Go2 (quadruped)|High-speed locomotion, obstacle avoidance|
|Slalom|Unitree G1 (humanoid)|Whole-body balance, dynamic obstacles|

**Observation:** All tasks use a single-view grayscale image from an onboard camera — no external state estimation, no multi-view setups.

### 4.3 Computational Overhead

|Method|Q-Functions|Params (M)|Train Time (ms/update)|
|---|---|---|---|
|AWAC|2|0.215|0.61|
|IQL|2|0.083|1.31|
|Cal-QL|2|0.090|1.51|
|**SigEnt-SAC**|**2**|**0.091**|**3.13**|
|RLPD|10|0.090|0.78|

SigEnt-SAC does not increase inference cost (same 2 Q-functions, similar parameter count). The slightly higher training time (~3ms vs ~1.5ms) comes from the gated BC computation.

---

## 5. Results

### 5.1 Simulation (D4RL One-Shot Setting)

- SigEnt-SAC is the **only method** to reach 100% success rate across all tested tasks
- Converges faster than all baselines (door, hammer, pen, kitchen tasks)
- No performance drop post-convergence — other methods exhibit degradation after peaking
- Substantially lower OOD action ratio compared to Cal-QL variants

### 5.2 Robustness to Demonstration Quality

Three degraded demonstration conditions tested:

- 50% random transition dropout
- Action noise ($\sigma = 0.2$ Gaussian)
- State noise ($\sigma = 0.2$ Gaussian)

SigEnt-SAC maintains strong performance under all three degradations. Cal-QL shows significant performance collapse, attributed to its high target entropy ($H_0 = 0$) increasing sensitivity to guidance quality.

### 5.3 Real-World Results

|Task|BC (30 demos)|VLM (zero-shot)|SigEnt-SAC (1 demo)|
|---|---|---|---|
|Push-Cube|20%|10%|**100%**|
|Ball-to-Goal|10%|10%|**100%**|
|Slalom (Go2)|0%|0%|**100%**|
|Slalom (G1)|0%|0%|**100%**|

All results over 10 evaluation trials per task.

### 5.4 Policy Improvement Beyond Demonstration

SigEnt-SAC's learned policy surpasses the expert demonstration in task efficiency:

|Task|Demo Steps|Learned Steps|Reduction|
|---|---|---|---|
|Push-Cube|21|18|14%|
|Ball-to-Goal|26|3|**88%**|
|Quadruped Slalom|38|23|39%|
|Humanoid Slalom|33|26|21%|

**Average reduction: 40.9% in task completion time.**

Notably, the Ball-to-Goal policy discovered an entirely different strategy (lateral sway instead of frontal push) that is both faster and more accurate.

---

## 6. Hyperparameter Sensitivity Analysis

### GBC Weight $\lambda$

- **Too small:** Policy relies mainly on online RL; oscillatory behavior without expert correction
- **Moderate:** Optimal — provides corrective gradients when needed, accelerates convergence
- **Too large:** Over-regularizes toward demonstration, slows exploration and adaptation

### Gating Threshold $\varepsilon$

- **Very small:** Gate always open → degenerates to unconditional BC; slow improvement
- **Very large:** Gate rarely opens → minimal expert guidance in early exploration; susceptible to Q-oscillations
- **Broad stable region observed across both hyperparameters**, indicating ease of tuning

---

## 7. Limitations

1. **Control interface:** Limited to high-level velocity commands; no joint-level full-body control demonstrated
2. **Task complexity:** Validated primarily on short-horizon, coarse tasks; no long-horizon or fine-grained manipulation benchmarks
3. **Embodiment diversity:** While cross-morphology is demonstrated, all tasks are locomotion or coarse pushing; no dexterous manipulation

---

## 8. Key Takeaways for Research

### Algorithmic Insights

- **Negative entropy is a fundamental problem** in tanh-squashed Gaussian policies under conservative Q-learning — not just a minor instability. The sigmoid reparameterization is a clean, lightweight fix.
- **Conditionality of behavior cloning matters** — unconditional BC constrains the policy unnecessarily; gating allows both stabilization and improvement beyond the demonstration.
- **Conservative regularization + bounded entropy are complementary** — CQL handles OOD Q-values; sigmoid entropy handles OOD action selection during policy improvement.

### Practical Implications

- A single expert trajectory is sufficient for real-world policy learning given the right algorithmic design — large datasets are not always necessary.
- Single-view, single-camera observations are viable for RL in dynamic environments if the method is noise-tolerant.
- Cross-embodiment generalization (manipulators, wheeled, quadrupeds, humanoids) is achievable with a unified framework without task-specific architecture changes.

### Positioning vs. Related Work

|Axis|Offline-to-Online (Cal-QL, RLPD)|VLA Fine-Tuning (GR-RL)|SigEnt-SAC|
|---|---|---|---|
|Data requirement|Large dataset|Large pretrained model + dataset|1 trajectory|
|Compute|Moderate|High (GPU for VLA)|Low|
|Embodiment generality|Limited|Limited|Broad|
|Visual setup|Multi-view common|Multi-view common|Single-view|
|Real-world validation|Rare|Tabletop manipulation|4 diverse platforms|

---

## 9. Connections to Agricultural Robotics

For robotics applications involving sparse-reward environments with few available demonstrations (e.g., tomato harvesting, fruit occlusion discovery), SigEnt-SAC offers several relevant properties:

- **One-shot efficiency** is directly applicable when collecting expert harvesting demonstrations is costly or task-specific
- **Robustness to noisy observations** mirrors challenges in outdoor agricultural settings (variable lighting, occlusion, plant deformation)
- **Policy improvement beyond demonstration** aligns with the goal of active perception methods that must discover novel viewpoints or strategies not present in the initial demonstration
- The **gated BC mechanism** is conceptually related to uncertainty-aware reward shaping — it activates corrective signals only when the policy is in uncertain territory, a principle applicable to occlusion-aware exploration

---

## 10. Citation

```bibtex
@article{wu2026sigentsac,
  title={Off-Policy Actor-Critic with Sigmoid-Bounded Entropy for Real-World Robot Learning},
  author={Wu, Xiefeng and Hu, Mingyu and Zhang, Shu},
  journal={arXiv preprint arXiv:2601.15761},
  year={2026}
}
```