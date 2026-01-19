# import gymnasium as gym
# from policy_linear_Q import LinearQLearningAgent
# import numpy as np
# from policies import get_possible_next_states
# from tetris_gymnasium.envs import Tetris
# import matplotlib.pyplot as plt

# # Initialize
# agent = LinearQLearningAgent()
# env = gym.make("tetris_gymnasium/Tetris", render_mode=None) # faster without render

# print("Initial Weights:", agent.weights) # Should be zeros

# episodes_rewards = []

# for episode in range(100):
#     env.reset()
#     terminated = False

#     episode_reward = 0
#     plays = 0
#     while not terminated:
#         plays += 1
#         best_candidate = agent.choose_action(env)
#         if best_candidate is None:
#             # Give a harsh penalty so the agent learns to avoid this state
#             if prev_features is not None:
#                 agent.learn(prev_features, -10, 0)
#             break

#         # Execute chosen sequence in real env
#         current_reward = 0
#         for action in best_candidate["sequence"]:
#             _, r, terminated, _, _ = env.step(action)
#             current_reward += r
#             if terminated:
#                 break

#         episode_reward += current_reward

#         # Features for the action you JUST took (afterstate features)
#         chosen_features = agent.get_features(
#             best_candidate["board"],
#             best_candidate.get("lines_cleared", 0),
#         )

#         # Best next Q (from the new state)
#         next_candidates = get_possible_next_states(env)
#         if next_candidates:
#             next_qs = [
#                 agent.evaluate_board(
#                     agent.get_features(c["board"], c.get("lines_cleared", 0))
#                 )
#                 for c in next_candidates
#             ]
#             best_next_q = max(next_qs)
#         else:
#             best_next_q = 0

#         # Correct Q-learning update
#         agent.learn(chosen_features, current_reward, best_next_q)

#         if terminated:
#             # We died during execution
#             # Reward = Death Penalty (-10) + any lines we barely cleared
#             final_reward = -10 + episode_reward
#             agent.learn(chosen_features, final_reward, 0)
#             break
#         if plays > 100:
#             final_reward = 100 + episode_reward
#             agent.learn(chosen_features, final_reward, 0)
#             break

#     episodes_rewards.append(episode_reward)
#     # Epsilon decay
#     agent.epsilon = max(agent.min_epsilon, agent.epsilon * agent.epsilon_decay)

#     print(f"Ep {episode} | Weights: {np.round(agent.weights, 5)} | Epsilon: {agent.epsilon:.2f} | Score: {final_reward} | Died in {plays} plays")

# print(f"Average reward: {np.mean(episodes_rewards)}")
# plt.figure(figsize=(10, 5))
# plt.plot(range(len(episodes_rewards)), episodes_rewards)
# plt.title("Reward per episode")
# plt.xlabel("Episode")
# plt.ylabel("Reward")
# plt.savefig("reward_per_episode.png")


import gymnasium as gym
from policy_linear_Q import LinearQLearningAgent
from policies import get_possible_next_states
import numpy as np
import random
from collections import deque
from tetris_gymnasium.envs import Tetris
import cv2


# CONFIG
BATCH_SIZE = 32
MEMORY_SIZE = 2000

RENDER = True

# Initialize
agent = LinearQLearningAgent(learning_rate=0.001, epsilon=1.0)
# env = gym.make("tetris_gymnasium/Tetris", render_mode=None)
env = gym.make("tetris_gymnasium/Tetris", render_mode="human", render_upscale=40)

# env = gym.make("tetris_gymnasium/Tetris", render_mode=None) # faster without render


# 1. The "Memory Bank"
replay_buffer = deque(maxlen=MEMORY_SIZE)

print("Initial Weights:", agent.weights)

for episode in range(500):
    env.reset()
    terminated = False
    episode_reward = 0
    
    prev_features = None
    
    plays = 0
    while not terminated:
        plays += 1
        if RENDER:
            env.render()
            cv2.waitKey(1)
        # 1. Choose Action
        best_candidate = agent.choose_action(env)
        
        # DEATH CHECK (No moves possible)
        if best_candidate is None:
            if prev_features is not None:
                # Store the death memory: (Features, Reward, Next_Best_Q)
                replay_buffer.append((prev_features, -10, 0))
            break

        # 2. Execute Action
        current_reward = 0
        for action in best_candidate["sequence"]:
            _, r, terminated, _, _ = env.step(action)
            current_reward += r
            if terminated: break
        
        episode_reward += current_reward
        
        # 3. Get Data for Learning
        chosen_features = agent.get_features(
            best_candidate["board"], 
            best_candidate.get("lines_cleared", 0)
        )
        
        # Calculate Next Q (for the Bellman target)
        if not terminated:
            next_candidates = get_possible_next_states(env)
            if next_candidates:
                next_qs = [
                    agent.evaluate_board(
                        agent.get_features(c["board"], c.get("lines_cleared", 0))
                    ) for c in next_candidates
                ]
                best_next_q = max(next_qs)
            else:
                best_next_q = 0
        else:
            best_next_q = 0
            # Apply Death Penalty to current reward
            current_reward -= 10 
        if plays > 100:
            final_reward = 100 + current_reward
            agent.learn(chosen_features, final_reward, 0)
            break
        # 4. STORE instead of Learning immediately
        replay_buffer.append((chosen_features, current_reward, best_next_q))
        prev_features = chosen_features

        # 5. BATCH LEARN (The "Validation" Step)
        # Only learn if we have enough memories to form an opinion
        if len(replay_buffer) > BATCH_SIZE:
            # Pick 32 random moments from history
            mini_batch = random.sample(replay_buffer, BATCH_SIZE)
            
            # Learn from all of them
            for (feat, rew, next_q) in mini_batch:
                agent.learn(feat, rew, next_q)

    # End of Episode Handling
    agent.epsilon = max(agent.min_epsilon, agent.epsilon * agent.epsilon_decay)
    

    print(f"Ep {episode} | Weights: {np.round(agent.weights, 3)} | Score: {episode_reward} | Mem: {len(replay_buffer)}")