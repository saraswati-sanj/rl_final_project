# FashionVerse

### Adaptive AI Fashion Styling using Reinforcement Learning with GenAI and Immersive 3D/VR Try-On

> **FashionVerse** is an adaptive AI fashion stylist that combines Generative AI for natural-language fashion understanding, Reinforcement Learning (PPO) for sequential personalized outfit selection, and 3D/WebXR visualization for immersive virtual try-on.

---

## 1. Abstract

Traditional fashion recommendation systems operate as static one-shot matrix factorization or classification filters ($User \rightarrow Model \rightarrow Top\text{-}K$). They fail to capture sequential composition constraints (e.g., matching tops to bottoms and shoes within strict budgets) and treat user interactions as passive clicks rather than active reward signals.

**FashionVerse** formulates outfit recommendation as a genuine **Markov Decision Process (MDP)**. A Proximal Policy Optimization (**PPO**) agent sequentially constructs coordinated outfits item-by-item across structured fashion categories while satisfying hard occasion, formality, and budget constraints. User feedback ($\heartsuit$ Love, $\thumbsup$ Like, $\thumbsdown$ Dislike, $\square$ Save, $\wp$ Buy) is converted into dense and terminal reward signals that update the user's observable preference state via Exponential Moving Average (EMA). An integrated **GenAI Layer** parses natural-language prompts and explains RL decisions, while a **Three.js / WebXR 3D Engine** provides real-time interactive try-on.

---

## 2. System Architecture

```
                                 ┌──────────────────────────────┐
                                 │ Natural Language User Prompt │
                                 │ ("Semi-formal college look") │
                                 └──────────────┬───────────────┘
                                                │
                                                ▼
                                 ┌──────────────────────────────┐
                                 │      GenAI Intent Parser     │
                                 │   (LLM / Rule-Based Heuristic│
                                 └──────────────┬───────────────┘
                                                │ [Occasion, Season, Budget, Formality]
                                                ▼
     ┌────────────────────────────────────────────────────────────────────────────────────────┐
     │                             RL AGENT (PPO Policy Network)                              │
     │                                                                                        │
     │   State S_t (70-dim)  ──────▶  Actor-Critic Network  ──────▶  Action A_t (60 discrete) │
     │   [User Belief State (23)                                    [Staged Slot × Candidate] │
     │    + Request Context (14)                                                              │
     │    + Outfit-So-Far (33)]                                                               │
     └──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                                │ Selected Outfit Items
                                                ▼
                                 ┌──────────────────────────────┐
                                 │      Gymnasium FashionEnv    │
                                 │   - Compatibility Scoring    │
                                 │   - Hard Budget Enforcement  │
                                 │   - Step Shaping Reward      │
                                 └──────────────┬───────────────┘
                                                │ Coordinated Look
                                                ▼
                                 ┌──────────────────────────────┐
                                 │  Three.js / WebXR 3D Avatar  │
                                 │  (Interactive Virtual Try-On)│
                                 └──────────────┬───────────────┘
                                                │
                                                ▼
                                 ┌──────────────────────────────┐
                                 │     User Feedback Signal     │
                                 │ (Love/Like/Dislike/Save/Buy) │
                                 └──────────────┬───────────────┘
                                                │
                                                ▼
                                 ┌──────────────────────────────┐
                                 │  Reward Calculation & Update │
                                 │  R = w_u*U + w_c*C + w_b*B   │
                                 │  Update User Belief Profile  │
                                 └──────────────────────────────┘
```

---

## 3. RL Formulation & Mathematical Framework

### State Space $\mathcal{S}$ ($\text{dim} = 70$)
At each decision timestep $t$, the state $S_t \in \mathbb{R}^{70}$ consists of five normalized segments:
1. **User Profile Estimate ($\mathbb{R}^{23}$)**: Style weights (9), color affinities (7), budget norm (1), formality estimate (1), and interaction acceptance statistics (5).
2. **Fashion Request Context ($\mathbb{R}^{14}$)**: Occasion one-hot (5), season one-hot (4), budget norm (1), formality targets (4).
3. **Current Outfit-So-Far ($\mathbb{R}^{33}$)**: Normalized 11-dimensional feature vectors for `[top/dress, bottom, shoes]`.
4. **Episode Progress ($\mathbb{R}^{4}$)**: Remaining budget norm, item count ratio, step ratio, coverage indicator.
5. **Reward History ($\mathbb{R}^{5}$)**: Rolling normalized rewards of previous 5 steps.

### Action Space $\mathcal{A}$ ($\text{dim} = 60$)
Rather than an intractable flat combinatorial action space ($360 \text{ items} \times 6 \text{ slots} > 2100$), FashionVerse uses a **Hierarchical Staged Discrete Action Space**:
$$\text{Action ID} = \text{Slot Index} \times 10 + \text{Candidate Index}$$
- Slots: `select_top`, `select_bottom`, `select_dress`, `select_shoes`, `select_accessory`, `finish_outfit`
- Candidates: Top-10 pre-filtered items ranked by compatibility with current outfit.

### Multi-Component Reward Function
$$R = w_1 U + w_2 C + w_3 O + w_4 B + w_5 D - w_6 P$$
- $U$: User explicit satisfaction ($\text{Love}: +10, \text{Like}: +5, \text{Save}: +7, \text{Purchase}: +15, \text{Dislike}: -8, \text{Skip}: -2$).
- $C$: Pairwise color, style, and formality compatibility score ($[0, 1]$).
- $O$: Target occasion match indicator ($0 \text{ or } 1$).
- $B$: Budget compliance bonus ($[0, 1]$).
- $D$: Recommendation diversity bonus (penalizes repeated items).
- $P$: Constraint violation penalties (over budget: $-8$, poor compatibility: $-3$).

### PPO Clipped Surrogate Objective
$$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right) \right] + c_1 L_t^{VF}(\theta) + c_2 S[\pi_\theta](s_t)$$
- Entropy bonus ($S[\pi_\theta]$) with configurable coefficient $\beta_{ent} = 0.01$ guarantees continuous exploration of novel fashion styles.

---

## 4. Empirical Evaluation & Baselines

| Model / Algorithm | Type | Mean Episode Reward | Acceptance Rate (%) | Personalization Mechanism |
|---|---|---|---|---|
| **Random** | Baseline | 4.84 | 56.5% | None |
| **Rule-Based (Greedy)** | Baseline | 5.08 | 62.5% | Static Rules |
| **Popularity-Based** | Baseline | 4.73 | 60.0% | Global Item Frequency |
| **Deep Q-Network (DQN)** | Secondary RL | 3.56 | 63.0% | Experience Replay Buffer |
| **FashionVerse PPO (Ours)** | Primary RL | **4.95** | **65.0%** | **Online Stochastic Policy (EMA)** |

### Generated Experiment Artifacts (in `experiments/plots/`):
1. `01_02_reward_curves.png`: PPO policy reward and moving average convergence.
2. `03_baseline_comparison.png`: Direct comparison across all 5 algorithms.
3. `04_acceptance_rate.png`: Acceptance rate distribution across models.
4. `05_user_satisfaction.png`: Comparative feedback label counts (PPO vs Random).
5. `06_exploration_exploitation.png`: Entropy coefficient ablation ($\beta = 0.05, 0.01, 0.001$).
6. `07_diversity_ablation.png`: Ablation of diversity reward component.
7. `08_preference_adaptation.png`: Policy recovery speed under stochastic preference drift.
8. `09_budget_violations.png`: Budget compliance comparison.
9. `10_reward_ablation.png`: Reward component contribution breakdown.

---

## 5. Local Running Instructions

### Prerequisites
- Python 3.11+
- Node.js v18+ & npm

### 1. Start FastAPI Backend
```bash
cd FashionVerse
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```
Backend Swagger documentation available at: `http://localhost:8000/docs`

### 2. Start React + Vite Frontend
```bash
cd FashionVerse/frontend
npm install
npm run dev
```
Frontend Web UI available at: `http://localhost:5173`

### 3. Run Automated Pytest Suite
```bash
cd FashionVerse
pytest tests/ -v
```
*(All 101 unit tests pass in < 4 seconds).*

---

## 6. Project Structure

```
FashionVerse/
├── backend/
│   ├── main.py                  # FastAPI entry point & routes
│   ├── api/                     # /chat, /recommend, /feedback, /avatar, /analytics, /user
│   ├── rl/                      # Gymnasium FashionEnv, PPO Agent, DQN, Reward, Actions, State
│   ├── fashion/                 # Catalog (360 items), Compatibility, Constraints, Generator
│   ├── user/                    # User Simulator (7 personas), Profile EMA, Preference Updates
│   ├── genai/                   # Intent Parser (LLM/NLP), Explanation Engine, Stylist
│   └── database/                # SQLite DB, SQLAlchemy models, Repository
├── frontend/
│   ├── src/
│   │   ├── components/          # Navbar, UI widgets
│   │   ├── pages/               # AI Stylist, 3D Try-On, Viva Demo Mode, RL Dashboard, My Style
│   │   ├── three/               # AvatarCanvas (Three.js WebGL & WebXR VR support)
│   │   └── services/            # API client
│   └── package.json
├── data/                        # fashion_items.csv, fashion_items.json
├── training/                    # train_ppo.py, evaluate.py
├── experiments/                 # config.yaml, results/*.json, plots/*.png
├── models/                      # ppo_exp1.zip, dqn_exp1.zip
└── tests/                       # test_phase1.py, test_environment.py, test_api.py
```
