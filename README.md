# Tetris Reinforcement Learning

Reinforcement Learning approaches for playing Tetris using the `tetris-gymnasium` environment.

## Project Structure

```
tetris/
├── tetris_material/     # Baseline policies (provided)
├── linear_features/     # Linear Q-learning approach
├── train_lin_grouped_dqn.py  # DQN with grouped actions
├── evaluate_grouped_dqn.py   # Evaluation script
└── notebooks/           # Debug and analysis notebooks
```

---

## 1. Baseline Policies (`tetris_material/`)

Standard policies for comparison:

| Policy | Description | Script |
|--------|-------------|--------|
| **Down** | Always hard drop | `view_episode_policy_down.py` |
| **Random** | Random actions | `view_episode_policy_random.py` |
| **Greedy** | Evaluates 80 sequences with heuristics | `view_episode_policy_greedy.py` |

Key file: `policies.py` - Contains `get_possible_next_states()` which simulates all possible placements using heuristics (height, holes, lines cleared).

---

## 2. Linear Q-Learning (`linear_features/`)

Hand-crafted feature-based approach:

- **Features**: Height, holes, bumpiness, lines cleared
- **Model**: Linear combination of features with learned weights
- **Training**: Q-learning with experience replay

Files:
- `policy_linear_Q.py` - Agent definition with feature extraction
- `train_tetris_linear_Q.py` - Training loop
- `play_linear_Q.py` - Play with trained weights

---

## 3. DQN with Grouped Actions (`train_lin_grouped_dqn.py`)

Deep Q-Network using environment wrappers for simplified action space.

### Wrappers
- **GroupedActionsObservations**: Each action = complete piece placement (~40 valid positions)
- **FeatureVectorObservation**: Board → 13-dim feature vector (heights, holes, bumpiness)

### Architecture
```
Input (13) → Linear(64) → ReLU → Linear(64) → ReLU → Linear(1) → Value
```

### Training
- **Algorithm**: DQN with replay buffer (30k) + target network
- **Exploration**: ε-greedy (1.0 → 0.001)
- **Steps**: 500,000

### Run
```bash
# Train
python train_lin_grouped_dqn.py

# Evaluate
python evaluate_grouped_dqn.py --model-path runs/<run_name>/model.cleanrl_model

# Watch trained agent
python run_trained_grouped_dqn.py
```


---

## Requirements

```bash
pip install tetris-gymnasium torch gymnasium stable-baselines3 tyro wandb
```
