import gymnasium as gym
import cv2
import numpy as np
from tetris_gymnasium.envs import Tetris
from policy_linear_Q import LinearQLearningAgent
from policies import get_possible_next_states

# ---- CONFIG ----
WEIGHTS = np.array([0.5, -0.01, -0.05, -0.05])  # example learned weights
RENDER = True

# ---- INIT ----
agent = LinearQLearningAgent()
agent.weights = WEIGHTS
agent.epsilon = 0.0  # pure greedy

env = gym.make("tetris_gymnasium/Tetris", render_mode="human", render_upscale=40)
env.reset(seed=42)

terminated = False
total_reward = 0

while not terminated:
    if RENDER:
        env.render()
        cv2.waitKey(1)

    candidates = get_possible_next_states(env)
    if not candidates:
        break

    # pick best candidate using the linear model
    best_score = -float("inf")
    best_candidate = None
    for cand in candidates:
        lines = cand.get("lines_cleared", 0)
        feats = agent.get_features(cand["board"], lines)
        score = agent.evaluate_board(feats)
        if score > best_score:
            best_score = score
            best_candidate = cand

    if best_candidate is None:
        break

    # execute chosen sequence
    for action in best_candidate["sequence"]:
        _, reward, terminated, _, _ = env.step(action)
        total_reward += reward
        if terminated:
            break
    print("Score:",total_reward,end='\r', flush=True)
print("Final Score:", total_reward)
