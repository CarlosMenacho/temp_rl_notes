## 1. Overview & Motivation

Global path planning sits at the top of the autonomous robot navigation stack, providing a reference path to the lower-level local trajectory planner. Most conventional methods (A*, Dijkstra, RRT) optimize for path _length_ alone, often producing routes dangerously close to obstacles — increasing the burden on local planners and risking failure. Existing RL methods suffer from slow convergence and poor safety guarantees.

This paper proposes **PF-SAC**, which integrates Artificial Potential Fields (APF) into the Soft Actor-Critic (SAC) framework as a supervised learning signal for an auxiliary critic network. A second variant, **PF-SAC-Risk**, adds a Monte Carlo risk assessment module to the state space, enabling adaptation to environmental changes without full retraining.

**Core claim:** Encoding domain knowledge (potential fields) directly into the critic training signal improves both convergence speed and path safety compared to purely data-driven RL approaches.

---

## 2. Problem Statement

### 2.1 Limitations of Conventional Methods

| Method           | Strength                       | Weakness                                               |
| ---------------- | ------------------------------ | ------------------------------------------------------ |
| A* / Dijkstra    | Optimal path length            | Slow in large/complex maps; ignores safety margins     |
| RRT              | Efficient sampling             | Non-smooth paths; no optimality guarantee              |
| APF (standalone) | Safety-aware                   | Gets trapped in local minima                           |
| DDPG             | Continuous action space        | Limited exploration; fails to converge in complex maps |
| SAC              | Better exploration via entropy | High path potential (unsafe proximity to obstacles)    |

### 2.2 Formal Problem Definition

Global path planning is cast as a constrained optimal control problem:
$$
arg⁡max⁡π(s)Eπ[∑t=0Tγtr(st)]\arg\max_{\pi(s)} \mathbb{E}_\pi \left[\sum_{t=0}^{T} \gamma^t r(s_t)\right]argπ(s)max​Eπ​[t=0∑T​γtr(st​)]
$$
Subject to:

- **Kinematic constraint:** $st=st−1+π(st)Δts_t = s_{t-1} + \pi(s_t)\Delta t st​=st−1​+π(st​)Δt$
- **Goal reaching:** ∥st−sg∥≤ϵ\|s_t - s_g\| \leq \epsilon ∥st​−sg​∥≤ϵ
- **Collision avoidance:** min⁡dt≥d0\min d_t \geq d_0 mindt​≥d0​ (minimum safe distance from obstacles)

---

## 3. Potential Field Formulation

### 3.1 Repulsive Field (Obstacles)

Pobs(st)={12κ(1ρ(st,ot)−1ρ0)2if ρ(st,ot)≤ρ00if ρ(st,ot)>ρ0P_{\text{obs}}(s_t) = \begin{cases} \frac{1}{2}\kappa\left(\frac{1}{\rho(s_t, o_t)} - \frac{1}{\rho_0}\right)^2 & \text{if } \rho(s_t, o_t) \leq \rho_0 \\ 0 & \text{if } \rho(s_t, o_t) > \rho_0 \end{cases}Pobs​(st​)=⎩⎨⎧​21​κ(ρ(st​,ot​)1​−ρ0​1​)20​if ρ(st​,ot​)≤ρ0​if ρ(st​,ot​)>ρ0​​

where κ\kappa κ is the repulsion factor, ρ0\rho_0 ρ0​ is the influence radius, and ρ(st,ot)\rho(s_t, o_t) ρ(st​,ot​) is the Euclidean distance to the obstacle.

### 3.2 Attractive Field (Goal)

Ptar(st)=12γ⋅ρ(st,gt)P_{\text{tar}}(s_t) = \frac{1}{2}\gamma \cdot \rho(s_t, g_t)Ptar​(st​)=21​γ⋅ρ(st​,gt​)

where γ\gamma γ is the gravitational factor and ρ(st,gt)\rho(s_t, g_t) ρ(st​,gt​) is the distance to the goal.

### 3.3 Combined & Normalized Field

$$
P(st)=Ptar(st)+Pobs(st)max⁡(Ps(st))∈[0,1]P(s_t) = \frac{P_{\text{tar}}(s_t) + P_{\text{obs}}(s_t)}{\max(P_s(s_t))} \in [0, 1]P(st​)=max(Ps​(st​))Ptar​(st​)+Pobs​(st​)​∈[0,1]
$$
Normalization stabilizes training by preventing large potential values from destabilizing gradients.

---

## 4. Method: PF-SAC

### 4.1 RL Formulation

- **State space:** S∈R2S \in \mathbb{R}^2 S∈R2, where s(t)=(x,y)s(t) = (x, y) s(t)=(x,y) (2D robot position)
- **Action space:** A∈R2A \in \mathbb{R}^2 A∈R2, where a(t)=(vx,vy)∈[−1,1]a(t) = (v_x, v_y) \in [-1, 1] a(t)=(vx​,vy​)∈[−1,1] (lateral/longitudinal velocity)
- **Reward function (sparse):**

R={+10if goal reached−2if collisionR = \begin{cases} +10 & \text{if goal reached} \\ -2 & \text{if collision} \end{cases}R={+10−2​if goal reachedif collision​

### 4.2 PF-Critic Network

Two additional fully connected networks (dual architecture, similar to SAC's dual Q-critics) are trained to _predict_ the potential field value at each state-action pair. Using the minimum of the two predictions prevents overestimation:

$$
JP(τ)=Est,at,st+1∼D[12(Pτ(st,at)−P^(st,at))2]J_P(\tau) = \mathbb{E}_{s_t, a_t, s_{t+1} \sim \mathcal{D}}\left[\frac{1}{2}\left(P_\tau(s_t, a_t) - \hat{P}(s_t, a_t)\right)^2\right]JP​(τ)=Est​,at​,st+1​∼D​[21​(Pτ​(st​,at​)−P^(st​,at​))2]
$$


where $P^(st,at)\hat{P}(s_t, a_t) P^(st​,at​)$ is the ground-truth potential value from the APF model (used as supervision), and $Pτ(st,at)P_\tau(s_t, a_t) Pτ​(st​,at​)$ is the network prediction.

**Key insight:** The potential field provides a dense, analytically computed supervision signal for free — no additional environment interactions required.

### 4.3 PF-Actor Network

The actor's policy objective is extended to include the predicted potential value as an additional minimization term:

$$
Jπ(ϕ)=Est∼D[αlog⁡πϕ(at∣st)−Qϕ(st,at)+λpPτ(st,at)]J_\pi(\phi) = \mathbb{E}_{s_t \sim \mathcal{D}}\left[\alpha \log \pi_\phi(a_t | s_t) - Q_\phi(s_t, a_t) + \lambda_p P_\tau(s_t, a_t)\right]Jπ​(ϕ)=Est​∼D​[αlogπϕ​(at​∣st​)−Qϕ​(st​,at​)+λp​Pτ​(st​,at​)]
$$
where $λp\lambda_p λp$​ is a weighting coefficient. This three-component objective simultaneously:

1. Maximizes entropy (encourages exploration)
2. Maximizes Q-value (optimizes reward)
3. **Minimizes predicted potential** (steers away from obstacles)

### 4.4 Network Architecture

|Component|Architecture|Hidden Units|Activation|
|---|---|---|---|
|Q-networks (×2)|3-layer FC|256|ReLU|
|Target Q-networks (×2)|3-layer FC|256|ReLU|
|PF-critic networks (×2)|3-layer FC|**512**|ReLU|
|PF-actor network|4-layer FC|256|ReLU + tanh|

The PF-critic uses larger hidden layers (512 vs 256) to capture the smoother potential field landscape.

### 4.5 PF-SAC Algorithm

```
Initialize: replay buffer D, policy φ, Q-params θ, PF-params τ

for each iteration:
  for each timestep:
    Sample action a_t from π(s_t)
    Observe r_t, s_{t+1}
    Compute potential field value p_t from P(s_t)
    Store (s_t, a_t, r_t, p_t, s_{t+1}) in D
  
  for each gradient step:
    Update Q-network:  θ ← θ - λ_Q ∇J_Q(θ)
    Update PF-critic:  τ ← τ - λ_p ∇J_P(τ)
    Update actor:      φ ← φ - λ_π ∇J_π(φ)
```

---

## 5. Extension: PF-SAC-Risk

### 5.1 Motivation

PF-SAC requires retraining when the environment changes. PF-SAC-Risk addresses this by encoding local collision risk directly into the state, allowing the policy to adapt to new obstacle configurations without retraining.

### 5.2 Monte Carlo Risk Assessment

A collision indicator variable is defined:
$$
C(x_i, y_i) =
\begin{cases}
1 & \text{if collision at } (x_i, y_i) \\
0 & \text{otherwise}
\end{cases}
$$
The risk value at position (x,y)(x, y) (x,y) is the empirical collision probability via Monte Carlo sampling:

$$
risk(x,y)=∑i=1NC(xi,yi)N\text{risk}(x, y) = \frac{\sum_{i=1}^{N} C(x_i, y_i)}{N}risk(x,y)=N∑i=1N​C(xi​,yi​)​
$$
where NN N samples are drawn from N((x,y),σ)\mathcal{N}((x, y), \sigma) N((x,y),σ) — a Gaussian centered at the robot's position.

### 5.3 Extended State Space

$$
sr(t)=(x(t), y(t), risk(x,y))s_r(t) = (x(t),\ y(t),\ \text{risk}(x, y))sr​(t)=(x(t), y(t), risk(x,y))
$$

The risk value acts as a local safety indicator injected at each timestep, enabling the policy to reason about obstacle proximity even in unseen environments.

### 5.4 Dense Reward Function

PF-SAC-Risk uses a denser reward to accelerate convergence:

$$
R =
\begin{cases}
+10 & \text{if goal reached} \\
-2 & \text{if collision} \\
-\sqrt{(x_t - x_o)^2 + (y_t - y_o)^2} & \text{otherwise}
\end{cases}​
$$
The continuous distance penalty provides gradient signal at every step, avoiding the sparse reward problem present in PF-SAC.

---

## 6. Experimental Setup

### 6.1 Test Environments

Four 2D maps used for PF-SAC evaluation (simulation only, 2D coordinates as input):

- **Map F (Flytrip):** 23m × 23m four-room maze with central obstacle
- **Map I:** Single wall between start and goal
- **Map N:** N-shaped corridor
- **Map U:** U-shaped corridor

For Gazebo simulation: 18m × 10m indoor environment with multiple rooms; 10 random target locations; gmapping for 2D map construction.

### 6.2 Training Parameters

|Parameter|Value|
|---|---|
|Potential field weight λp\lambda_p λp​|10|
|Safe distance ρ0\rho_0 ρ0​|2m|
|Repulsive factor κ\kappa κ|5|
|Gravitational factor γ\gamma γ|1|
|Risk sample size NN N|500|
|Sample variance σ\sigma σ|[0.5, 0.5]|
|Batch size|256|
|Episodes|1000|
|Max trajectory length|100|
|Discount factor|0.99|

### 6.3 Baselines

- **Conventional:** A*, Dijkstra, RRT, APF
- **RL-based:** DDPG, SAC, Risk-Conditioned-SAC

---

## 7. Results

### 7.1 PF-SAC vs. Conventional Methods (Map F & N)

|Method|Reach Target|Time Cost (ms)|Path Potential|
|---|---|---|---|
|A*|✓|2.401 ± 0.549|0.057 ± 0.000|
|Dijkstra|✓|2.201 ± 0.447|0.056 ± 0.000|
|APF|✗ (local minima)|—|—|
|RRT|✓|62.517 ± 10.396|**0.054 ± 0.003**|
|DDPG|✗|—|—|
|SAC|✓|**1.680 ± 0.406**|0.191 ± 0.000|
|**PF-SAC**|✓|**1.784 ± 0.403**|**0.018 ± 0.000**|

Key results across all maps:

- **38.64% faster** computation time than A*
- **95.72% lower path potential** than RRT (Map N)
- **90.58% lower path potential** than SAC (Map F)
- **40.48% faster convergence** than SAC (converges ~250 epochs vs ~420 for SAC)
- DDPG fails to converge even after 1000 episodes

### 7.2 Path Quality

PF-SAC consistently generates paths that pass through the center of traversable regions (maximizing clearance from obstacles), reflecting the repulsive field effect. SAC-generated paths frequently hug obstacles, while PF-SAC paths exhibit Bellman optimal properties — multiple starting points converge to overlapping paths in the latter half, indicating a well-defined optimal value function.

### 7.3 PF-SAC-Risk Generalization

Trained on a one-wall environment, evaluated on a two-wall (structurally different) environment:

|Method|Reaches Target (One-Wall)|Reaches Target (Two-Wall)|
|---|---|---|
|SAC|✓|✗|
|Risk-Conditioned-SAC|✓|✗|
|**PF-SAC-Risk**|✓|✓|

PF-SAC-Risk is the **only method** that successfully transfers to the unseen environment. The Monte Carlo risk value effectively encodes local obstacle context at inference time, compensating for the shifted obstacle layout.

---

## 8. Limitations

1. **2D only:** State space is limited to 2D positions; no elevation, dynamic obstacles, or 3D environments
2. **Requires map:** The APF requires a known, static occupancy map — unsuitable for unknown environments without mapping
3. **Environment shift:** When the evaluation environment differs substantially from training (beyond minor obstacle changes), PF-SAC-Risk still fails and retraining is needed
4. **No local planning:** The method addresses only global path planning; local trajectory tracking and dynamic obstacle avoidance are left to downstream modules
5. **Sparse reward sensitivity:** PF-SAC's sparse reward can slow initial convergence; PF-SAC-Risk mitigates this with the distance-based dense reward

---

## 9. Key Takeaways for Research

### Algorithmic Insights

**Supervised auxiliary heads are a powerful hybridization strategy.** Rather than integrating APF as a reward shaping term (which can create complex credit assignment issues), the authors use APF as direct supervision for a separate network head. This provides clean, dense gradient signal without corrupting the main RL reward structure.

**Risk as state vs. risk as reward.** PF-SAC-Risk encodes risk in the _state_ rather than the reward. This is architecturally significant — it allows the same policy to respond differently to different local environments at inference time, without any parameter update. This is essentially a form of context-conditioned generalization.

**Convergence acceleration via domain knowledge.** The 40.48% faster convergence vs. SAC is achieved by initializing the critic's "spatial intuition" through supervised potential field learning, reducing the amount of exploration needed to discover safe regions.

### Comparison with Related Work

|Axis|PF-SAC|SigEnt-SAC (Wu et al., 2026)|
|---|---|---|
|Setting|Global path planning (2D sim)|Real-world task learning|
|State input|2D coordinates|Raw visual images|
|Prior knowledge|APF (analytical)|Single expert trajectory|
|Safety encoding|Potential field critic|Gated BC + bounded entropy|
|Generalization|Limited (requires retraining)|Cross-embodiment|
|Deployment|Simulation + Gazebo|Real physical robots|

### Relevance to Agricultural Robotics

For autonomous navigation in agricultural environments (e.g., row crops, greenhouse aisles), PF-SAC offers several transferable ideas:

- **Repulsive field modeling** can encode plant rows and structures as soft obstacles, guiding the robot through the center of crop rows — directly analogous to its behavior in maze environments
- **Monte Carlo risk assessment** is particularly interesting for deformable crop environments where obstacle boundaries are probabilistic (e.g., hanging branches, tall plants), as the Gaussian sampling naturally models spatial uncertainty
- The **potential field critic as supervised auxiliary task** is a design pattern worth adapting: in harvesting contexts, ripeness maps, occlusion maps, or reachability maps could serve as alternative supervised signals for auxiliary critic heads
- The limitation of requiring a known map aligns with the practical constraint in agricultural robotics of needing a prior map — though semantic maps from SLAM or aerial surveys could substitute