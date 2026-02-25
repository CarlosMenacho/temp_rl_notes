## 1. Introduction

### Motivation

- Transformers have achieved remarkable success in NLP (GPT series, BERT) and computer vision (ViT)
- The sequential nature of RL decision-making makes transformers a natural architectural choice
- Offline RL's data-driven approach aligns well with transformers' ability to learn from large static datasets

### Key Challenges Addressed

1. **Traditional deep RL limitations**: Bootstrapping issues and the "deadly triad" problem
2. **Data efficiency**: RL environments often have costly interactions (robotics, healthcare, autonomous driving)
3. **Partial observability**: Need for effective memory mechanisms in POMDPs

### Main Contributions

The survey provides:

- Comprehensive taxonomy of TRL methods
- Analysis of architecture enhancement vs. trajectory optimization approaches
- Review of real-world applications
- Discussion of limitations and future research directions

---

## 2. Preliminaries

### 2.1 Reinforcement Learning Foundations

**MDP Formulation:**

- Defined as 6-tuple: M = (S, A, T, ρ₀, R, γ)
- Goal: Find optimal policy π* that maximizes expected cumulative reward

**RL Categories:**

1. **Model-Based vs. Model-Free**
    - Model-based: Learns transition dynamics p(s_{t+1}, r_t | s_t, a_t)
    - Model-free: Learns policy directly from experience
2. **On-Policy vs. Off-Policy**
    - On-policy: Updates policy using current policy's data
    - Off-policy: Uses separate behavior and target policies
3. **Online vs. Offline**
    - Online: Interacts with environment during training
    - Offline: Learns from pre-collected static datasets

### 2.2 Transformer Architecture

**Core Components:**

1. **Self-Attention Mechanism:**

$$
   Attention(Q, K, V) = softmax(QK^T / √d_k)V
$$

- Models pairwise relations between input tokens
- Q (query), K (key), V (value) vectors derived from linear projections

2. **Multi-Head Attention:**
    - Allows model to jointly attend to information from different representation subspaces
    - h parallel attention heads with dimension d_model/h each
3. **Feed-Forward Networks:**
    - Position-wise fully connected layers
    - Two linear transformations with nonlinear activation (GELU/ReLU)
4. **Positional Encoding:**
    - Sinusoidal encoding to capture position information
    - Enables attention to relative positions

---

## 3. Transformer-Based RL Methods

### 3.1 Architecture Enhancement

This approach applies transformers within traditional RL frameworks to improve representation learning.

#### 3.1.1 Feature Representation

**Key Challenge:** POMDPs require historical observations for optimal decision-making.

**Notable Methods:**

1. **Working Memory Graph (WMG)** [Loynd et al., 2020]
    - Processes variable number of factor vectors
    - Introduces "Memos" for historical information storage
    - Novel shortcut recurrence structure
2. **Gated Transformer-XL (GTrXL)** [Parisotto et al., 2019]
    - Addresses instability in canonical transformer for RL
    - Key modifications:
        - Replaces residual connections with gating mechanisms
        - Layer normalization relocated to "skip" stream (identity map reordering)
        - Options: highway connections, SigTanh gates, GRU gates
    - Demonstrated superior performance on DMLab-30 benchmark
3. **CoBERL** [Banino et al., 2021]
    - Combines LSTM with transformer architecture
    - LSTM captures recent dependencies
    - Transformer handles long-range dependencies
    - Optimizes memory utilization
4. **Compressive Transformer** [Rae et al., 2021]
    - Retains old memories rather than discarding them
    - Effective LSTM replacement in IMPALA framework
5. **Recurrent Fast Weight Programmers (RFWP)** [Irie et al., 2022]
    - Recognizes equivalence between linear transformers and FWPs
    - Incorporates recurrent connections
    - Significant improvements on Atari 2600 domain

#### 3.1.2 Environment Representation

**Focus:** Using transformers for dynamics and reward modeling in model-based RL.

**Notable Methods:**

1. **TransDreamer** [Chen et al., 2022]
    - Integrates transformer into Dreamer agent
    - Introduces Transformer State-Space Model (TSSM)
    - Replaces RNN in RSSM with transformer
    - Superior in tasks requiring long-term memory
2. **IRIS (Imagination with auto-regression over inner speech)** [Micheli et al., 2022]
    - Operates in imaginative realm with discrete auto-encoder
    - GPT-like autoregressive transformer simulates environment dynamics
    - Translates dynamics learning into sequence generation
    - Remarkable efficiency on Atari 100k benchmark
3. **MINECLIP** [Fan et al., 2033]
    - Autonomously generates reward signals from video demonstrations
    - Evaluates congruence between linguistic goals and visual observations
    - Uses InfoNCE objective for optimization
    - Demonstrates adaptable reward function modeling

### 3.2 Trajectory Optimization

This approach treats RL as a conditional sequence modeling problem, fundamentally changing the paradigm.

#### 3.2.1 Conditioned Behavior Cloning

**Decision Transformer (DT)** [Chen et al., 2021]

Core innovation: Models trajectories as sequences of (return-to-go, state, action) tuples:

```
τ = (R₁, s₁, a₁, R₂, s₂, a₂, ..., R_T, s_T, a_T)
```

Where RTG R_t = Σ_{t'=t}^T r_{t'} represents cumulative future rewards.

**Training:**

- Auto-regressive modeling on trajectory sequences
- Cross-entropy loss for discrete actions, MSE for continuous

**Inference:**

- Condition on desired target return
- Generate actions sequentially
- Update RTG by subtracting observed rewards

**Advantages:**

- Avoids bootstrapping issues
- Handles sparse reward environments
- More stable than traditional value-based methods

**Trajectory Transformer (TT)** [Janner et al., 2021]

- Models state and reward transitions explicitly
- Discretizes each dimension independently:

```
  τ = (s¹_t, s²_t, ..., s^N_t, a¹_t, a²_t, ..., a^M_t, r_t)^T_{t=1}
```

- Uses beam search for planning with RTG signal
- More aligned with model-based methods

**Limitations and Optimality Conditions:**

**ESPER** [Paster et al., 2034]

- Addresses DT's failure in stochastic environments
- Conditions on **expected returns** instead of realized returns
- Clusters trajectories into discrete representations
- Predictions independent of environmental stochasticity

**Theoretical Analysis** [Brandfonbrener et al., 2022]

- DT requires stricter assumptions than classical DP methods:
    - Nearly deterministic dynamics
    - Known target return
    - Conditioning value consistent with dataset distribution

#### 3.2.2 Canonical RL Integration

**Goal:** Combine sequence modeling benefits with traditional RL algorithms.

**Online Decision Transformer (ODT)** [Zheng et al., 2022]

- Integrates online fine-tuning with DT pretraining
- Addresses exploration-exploitation via max-ent RL framework
- Explicitly imposes lower bound on policy entropy:

$$
  min_θ J(θ) = E[−log π_θ(a|s, R̂)]
  subject to H^τ_θ[a|s, R̂] ≥ β
$$

- Sequence-level entropy differs from transition-level (SAC)

**StARformer** [Shang et al., 2022]

- Incorporates Markovian-like inductive bias
- Two components:
    - **Step transformer**: Local Markovian representations within (s,a,r) triples
    - **Sequence transformer**: Long-term dependencies
- Processes step-wise rewards without RTG design

**Graph-DT** [Hu et al., 2023]

- Uses dependency graph and graph transformer
- Explores Markovian bias with graph structure
- Reduced parameters and computation vs. StARformer

**Bootstrapped Transformer (BooT)** [Wang et al., 2022]

- Generates synthetic data through model itself
- Uses generated data for further refinement
- Confidence-based trajectory selection:

```
  c(τ) = 1/(T(N+M+2)) Σ log P_θ(τ_t|τ_{<t})
```

- Mitigates overfitting, enhances data coverage

**Stitching Ability Improvements:**

Methods to synthesize optimal policies from sub-optimal trajectories:

1. **Q-learning DT (QDT)** [Yamagata et al., 2022]
    - Integrates Q-learning to relabel RTG tokens
    - Refines training data quality
2. **DoC** [Yang & Nachum, 2022]
    - Conditions on latent future trajectory representation
    - Reduces mutual information to anticipate outcomes
3. **EDT, CGDT** [Wu et al., 2023; Wang et al., 2023]
    - Dynamically filter optimal trajectories
    - Use learned value estimator
    - Aggregate estimated returns across diverse trajectories

#### 3.2.3 Pretraining

**Leveraging External Data:**

**Language Pretraining** [Reid et al., 2022]

- Uses natural language pretraining for offline RL
- Demonstrates improved convergence and performance
- Similarity-based objective aligns language embeddings with trajectory inputs:

```
  L_cos = −Σ max_j C(I_i, E_j)
```

- Unified view of sequence modeling domain

**Pretrained Language Models** [Li et al., 2022]

- Leverages LMs to scaffold learning
- No need to translate states/actions to natural language
- Strategic encoding of sequential input is key

**Self-Supervised Pretraining:**

**MaskDP (Masked Decision Prediction)** [Liu et al., 2022]

- Random masking of state and action tokens
- Enhances generalizability through unsupervised data
- Critical role of mask ratios due to temporal correlations

**Multi-Pattern Masking** [Wu et al., 2023]

- Training with varied masking patterns
- Models gain versatile capabilities
- Adaptable for different roles via mask selection

**Optimal Pretraining Objectives** [Yang et al., 2031]

- Explored three downstream categories:
    - Limited-data imitation learning
    - Offline RL (using BRAC)
    - Online RL (using SAC)
- Finding: Optimal objective varies by downstream task
- No universally superior pretraining objective

#### 3.2.4 Generalist Agents

**Goal:** Single agent capable of mastering diverse tasks across domains.

**Multi-Game Decision Transformer** [Lee et al., 2022]

- Extends DT to 46 Atari games simultaneously
- Addresses mixed-quality dataset challenge
- Inference-time methodology using binary classifier:

```
  P(R_t|expert_t, ...) ∝ P_θ(R_t|...)P(expert_t|R_t, ...)
```

- Expert likelihood proportional to future returns
- Demonstrates scaling law: performance improves with model size

**Gato** [Reed et al., 2022]

- Transformer-based general agent
- Multi-modal, multi-task, multi-embodiment domains
- Standardizes diverse data into uniform token sequences
- Training objective:

```
  L(θ) = −Σ_b Σ_l m(b,l) log p_θ(s^(b)_l | s^(b)_1, ..., s^(b)_{l-1})
```

- Trained exclusively on near-optimal experience
- Requires expert trajectory prompts during evaluation

**Generalized Decision Transformer (GDT)** [Furuta et al., 2021]

- Leverages diverse hindsight information
- Formalizes as Information Matching (IM) problem:

```
  min_π E[D(I^Φ(τ), z)]
```

- Flexible framework with adjustable:
    - Feature function Φ
    - Anti-causal aggregator
- Generalizes across unseen/synthetic multi-modal distributions

**Prompt-Decision Transformer (Prompt-DT)** [Xu et al., 2022]

- Inspired by NLP prompt-based frameworks
- Uses trajectory segments as prompts (not text descriptions)
- Agent imitates demonstrations without fine-tuning
- Effective in meta-RL environments (Cheetah-dir, Ant-dir)
- Generalizes to out-of-distribution tasks

**Additional Generalization Approaches:**

1. **AnyMorph** [Trabucco et al., 2022]: Adapts to novel agent morphologies
2. **Attention-Neuron** [Tang & Ha, 2021]: Handles sudden input reordering
3. **Transfer-DT** [Boustati et al., 2021]: Robust to dynamics shifts via causal reasoning
4. **SwitchTT** [Lin et al., 2022]: Sparsely activated model mitigates parameter sharing issues

#### 3.2.5 Multi-Agent Extension

**Multi-Agent Decision Transformer (MADT)** [Meng et al., 2021]

**Framework:**

- Offline pretraining + online fine-tuning for MARL
- Large-scale dataset from MAPPO on SMAC tasks
- Trajectory representation:

```
  τⁱ = (x₁, ..., x_t, ..., x_T) where x_t = (s_t, oⁱ_t, aⁱ_t)
```

- s_t: global state
- oⁱ_t: local observation for agent i
- aⁱ_t: action for agent i

**Training:**

- Pretraining: Cross-entropy loss to align output with ground-truth actions
- Online: Integrate pretrained transformer into PPO actor-critic

**Results:**

- Superior sample efficiency
- Significant generalizability improvements on SMAC

**Multi-Agent Transformer (MAT)** [Wen et al., 2048]

**Architecture:** Encoder-decoder transformer structure

**Key Innovation:**

- Addresses complexity of agent interactions
- Sequential policy optimization formulation:

```
  A^{i₁:n}_π(o, a^{i₁:n}) = Π^n_{m=1} A^{i_m}_π(o, a^{i₁:m-1}, a^{i_m})
```

- Monotonically improving performance guarantee

**Differences from MADT:**

- Trained only via online trial-and-error (PPO-like objective)
- No offline imitation learning phase
- Superior performance, data efficiency, few-shot learning

**Limitations:**

- Requires intensive agent-to-agent communication
- Challenges in resource-constrained environments

**CommFormer** [Hu et al., 2024]

- Optimizes communication graph with architecture parameters
- Bi-level optimization strategy
- Continuous relaxation for graph representation
- Maintains performance with reduced communication

**Other MARL Approaches:**

1. **ATM** [Yang et al., 2022]: Entity-bound action layer for semantic inductive biases
2. **UPDeT** [Hu et al., 2021]: Decouples policy distribution from observations
3. **MaskMA** [Liu et al., 2024]: Masks units for zero-shot capabilities

---

## 4. Applications of TRL

### 4.1 Robotic Manipulation

**One-Shot Imitation Learning:**

**T-OSIL** [Dasari & Gupta, 2021]

- Transformers capture relational features from demonstrations
- Unsupervised inverse dynamics loss
- Models dynamic interactions in multi-agent environments

**MOSAIC** [Zhao et al., 2022]

- Temporal contrastive loss distinguishes adjacent/non-adjacent frames
- Refines temporal representation coherence

**Language-Based Interfaces:**

**Trajectory Modeling from Natural Language** [Bucker et al., 2022]

- Language-based interface for trajectory adaptation
- Frames trajectory generation as sequence modeling
- Multi-modal attention aligns instructions with geometry

**LATTE** [Bucker et al., 2022]

- Incorporates CLIP image encoder for 3D space
- Applicable to aerial and legged robots
- Components:
    - Language and contextual encoder
    - Geometry encoder
    - Multi-modal transformer decoder

**Task-Specific Adaptation:**

**TTP (Prompt-Situation Architecture)** [Jain et al., 2022]

- Adapts to user preferences with single demonstration
- Optimization objective:

```
  min L_CE(a, π(S, ψ(τ)))
```

- Demonstrated in simulated kitchens and Franka Panda arm

**Additional Applications:**

- Gripper manipulation [Han et al., 2021; Jangir et al., 2022]
- Legged locomotion [Yang et al., 2021]
- Dual-arm robots [Kim et al., 2021]

### 4.2 Text-Based Games

**Challenges:**

- Expansive action spaces (>1 billion possible actions)
- Partial observability
- Sparse rewards
- Game length and verbosity

**Key Frameworks:** TextWorld, LIGHT, Jericho, TextWorld with QA

**Notable Methods:**

**Q*BERT** [Ammanabrolu et al., 2020]

- Pretrained transformer for QA tasks in TBGs
- Generates questions for each observation
- Answers update knowledge graph for action selection
- Faster asymptotic performance via QA system

**GATA (Graph-Aided Transformer Agent)** [Adhikari et al., 2020]

- Data-driven approach to construct/update belief graphs
- Two components:
    - **Graph updater**: Observation reconstruction as seq2seq task
    - **Action selector**: Encodes belief graph and observations
- Bidirectional attention for refinement
- Superior performance with graph-structured representations

**LTL-Enhanced GATA** [Tuli et al., 2022]

- Augments GATA with Linear Temporal Logic
- Monitors progress toward instruction fulfillment
- Directs actions toward defined goals

**OOTD (Object-Oriented Text Dynamics)** [Liu et al., 2056]

- Model-based planning approach
- Graph representations for objects
- Separate transition layers for belief state prediction
- Transformer-based models for:
    - State transition predictions
    - Reward correlations
- Object- and self-supervised training
- Significantly improved sample efficiency on TextWorld

### 4.3 Navigation

**Vision-Language Navigation (VLN):**

**Problem:** Agents interpret language instructions, perceive environment, execute actions to reach goals.

**Modeled as:** Partially observable MDP requiring memory for partial instruction integration.

**Datasets/Simulators:** R2R, R2RIE, Touchdown, CVDN, VNLA, HANNA, REVERIE

**Pretraining Strategies:**

**PREVALENT** [Hao et al., 2020]

- Aligns language instruction representations with visual states
- Two pretraining tasks:
    1. **Image-attended MLM**: Predicts masked words using context + visual states
    2. **Action Prediction (AP)**: Forecasts actions from instructions + visual inputs
- Combined objective optimizes both tasks

**Similar Approaches:**

- **PRESS** [Hong et al., 2021]: Leverages BERT for instruction comprehension
- **VLN-BERT** [Hong et al., 2021]: Fine-tunes ViLBERT on instruction-trajectory pairs
- **BEVBert** [An et al., 2023]: Pretraining with unified map
- **Recurrent BERT** [Hong et al., 2021]: Integrates recurrent function

**Long-Range Dependencies:**

**HAMT (History-Aware Multi-Modal Transformer)** [Chen et al., 2021]

- Cross-modal transformer identifies long-range dependencies
- Hierarchical structure for computational efficiency
- Pretraining: AP and MLM tasks
- Fine-tuning: RL + imitation learning objective
- Robust performance across varied environments

**Topological Navigation** [Chen et al., 2022]

- Topological map tracks visited/navigable locations
- Enables efficient long-term planning

**E.T. (Episode Transformer)** [Pashevich et al., 2021]

- Encodes complete episode histories
- Synthetic instructions simplify learning
- Enhanced generalization

**Audio-Vision-Language:**

**AVLEN** [Paul et al., 2022]

- Transformer architecture for audio-vision-language environment
- Localizes audio sources in realistic visual settings
- Hierarchical RL policy structure
- Two levels for multi-modal information processing

### 4.4 Autonomous Driving

**Problem Definition:** Point-to-point navigation in urban settings while:

- Maintaining safe distance from dynamic agents
- Following traffic rules

**Approaches:** Supervised learning vs. Reinforcement learning

**Simulators/Datasets:** CARLA, NoCrash, Nuscenes

**Multi-Modal Sensor Fusion:**

**TransFuser** [Prakash et al., 2021]

- Self-attention transformer integrates modalities (camera images, LiDAR)
- Auto-regressive waypoint prediction via GRU
- Imitation learning to replicate expert trajectories
- Minimizes expected loss between predicted/actual waypoints

**InterFuser** [Shao et al., 2061]

- Addresses TransFuser's scalability limitations
- Interpretable sensor fusion transformer
- Multi-modal multi-view sensor integration
- Superior navigation in complex/adversarial conditions

**Stochastic Environment Adaptation:**

**SPLT (Separated Latent Trajectory Transformer)** [Villaflor et al., 2062]

- Dual-model approach:
    - **World model**: Reconstructs returns, rewards, states
    - **Policy model**: Generates action sequences
- Disentangles policy effects from world dynamics
- Mitigates risks in stochastic, safety-critical domains
- Superior performance on CARLA vs. traditional models

**Multi-Modal Expert Data:**

**BeT (Behavior Transformer)** [Shafiullah et al., 2198]

- Learns from inherently multi-modal, sub-optimal data
- Segments continuous actions into:
    - "Action centers"
    - "Residual actions"
- Maps observations to discrete action distributions
- Comprehensive coverage of data modes
- Demonstrated on CARLA benchmark

---

## 5. Discussion: Limitations and Future Directions

### 5.1 Limitations and Challenges

#### Local Context Sensitivity

- **Issue:** Transformers excel at global context but struggle with local details
- Self-attention focuses on point-wise comparisons across entire sequences
- **Solutions:** Local window-based attention (inspired by CNN)
- Examples: StARformer, Graph-DT, Lin et al. 2023

#### RL-Specific Transformer Architectures

- **Current State:** Most TRL adapts NLP-focused models
- **Needs:** Transformers tailored specifically for RL
- **Existing Modifications:**
    - GTrXL: Gating mechanisms for stability
    - StARformer: Markovian-like inductive bias
    - SPLT: Disentangled policy/world dynamics
- **Gap:** No universally applicable RL-centric transformer

#### Stochastic Environment Adaptation

- **Challenge:** DT-like methods fail in stochastic environments
- **Requirements for Optimal Policy** [Brandfonbrener et al.]:
    - Deterministic dynamics
    - Known target return
    - Consistent conditioning value with dataset distribution
- **Partial Solutions:**
    - ESPER: Conditioning on expected returns (still inferior to traditional RL)
    - Visual goal conditioning remains problematic
- **Open Problem:** Overcoming preconditions and limitations

#### Efficiency in Model Design

- **Issue:** Computational demands vs. lightweight traditional RL networks
- **Challenges:**
    - Hardware limitations
    - Constraints on model scalability
    - Limited capacity for extended input sequences
- **Attempts:** Compression, distillation (still computationally expensive)
- **Need:** Computationally efficient transformer models

### 5.2 Future Prospects

#### Integration with Traditional RL Algorithms

**Key Research Directions:**

1. **Computational Optimization**
    - Reduce computational demands for broader RL application
    - Enable deployment in resource-constrained environments
2. **Adaptive Memory Modules**
    - Alternative to LSTM units in POMDPs
    - Explore additional roles beyond memory
    - Investigate capabilities beyond current applications
3. **Universal RL Algorithm Emulation**
    - Transformers can potentially emulate all RL algorithms [Laskin et al., 2022]
    - Develop versatile models that:
        - Dynamically select appropriate traditional RL algorithm
        - Implement based on specific task requirements
    - **Concept yet to be thoroughly investigated**

#### Enhancing Sequence Modeling with RL Advantages

**Return-to-Go (RTG) Optimization:**

- **Current Limitation:** DT-like approaches rely on RTG tokens but struggle with optimal setting
- **Research Needs:**
    - Identify optimal RTG conditions
    - Develop functional approaches for RTG setting
    - Address inadequate RTG conditions across environments

**Versatile Policy Framework:**

- Process various modalities (images, videos, texts, speech)
- Standardized processing blocks
- **Open Questions:**
    - Can agents efficiently generalize from extensive datasets?
    - Methods for training agents on unseen tasks without heavy assumptions

**Generalized World Model:**

- Assess transformers' capacity for world model across tasks/scenarios
- Represents pivotal shift toward generalist AI systems

#### Practical Applications

**Decision Network Integration:**

- Beyond feature extraction to actual decision-making
- **Critical Needs:**
    - Understanding error sources
    - Identifying potential adverse impacts
    - Ensuring security and reliability
    - Comprehensive analysis of operational intricacies

#### Reverse Direction: RL for Transformers

**Emerging Paradigm:**

- Using RL to enhance transformer training (less explored)
- **Current Approaches:**
    - Offline RL for language/dialogue generation [Snell et al., 2022]
    - Relabeling strategies [Snell et al., 2022]
    - Value functions [Verma et al., 2022]
    - **RLHF (Reinforcement Learning from Human Feedback):** Aligns language models with human intent [Ouyang et al., 2022]

**Future Potential:**

- RL as pivotal mechanism for enhancing transformers across domains

---

## 6. Conclusion

This comprehensive survey provides:

1. **Systematic Overview:**
    - Integration of transformers in RL
    - Two main methodologies: Architecture Enhancement and Trajectory Optimization
    - Real-world applications across diverse domains
2. **Key Insights:**
    - **Architecture Enhancement:** Applies transformers within traditional RL frameworks for better representation
    - **Trajectory Optimization:** Reimagines RL as sequence modeling, leveraging transformers' long-sequence capabilities
    - Both approaches show significant promise but face distinct challenges
3. **Application Domains:**
    - **Robotic Manipulation:** One-shot imitation, language-based interfaces
    - **Text-Based Games:** Knowledge graphs, QA systems, model-based planning
    - **Navigation:** VLN with pretraining, multi-modal fusion
    - **Autonomous Driving:** Sensor fusion, stochastic environment handling
4. **Research Gaps:**
    - Need for RL-specific transformer architectures
    - Stochastic environment adaptation
    - Computational efficiency
    - Local context sensitivity
5. **Future Directions:**
    - Integration with traditional RL advantages
    - Generalist agents and world models
    - Bidirectional enhancement (RL↔Transformers)
    - Practical deployment in safety-critical domains