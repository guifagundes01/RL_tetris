"""
Training script for Linear Q-Learning with GroupedActionsObservations.
Much faster than deepcopy-based simulation.
"""
import numpy as np
import random
from collections import deque
import gymnasium as gym

from policy_linear_Q_grouped import LinearQAgentGrouped, create_grouped_env


# ============== CONFIG ==============
NUM_EPISODES = 500
BATCH_SIZE = 16
MEMORY_SIZE = 2000
LEARNING_RATE = 0.01
RENDER = False
SAVE_WEIGHTS = True
# ====================================


def compute_best_next_q(agent, obs, action_mask):
    """Compute max Q-value over all valid next actions."""
    valid_actions = np.where(action_mask == 1)[0]
    if len(valid_actions) == 0:
        return 0.0
    
    best_q = -float('inf')
    for action in valid_actions:
        features = agent.get_features_from_obs(obs[action], lines_cleared=0)
        q = agent.evaluate(features)
        if q > best_q:
            best_q = q
    
    return best_q


def main():
    # Initialize
    agent = LinearQAgentGrouped(learning_rate=LEARNING_RATE, epsilon=1.0)
    env = create_grouped_env(render_mode="human" if RENDER else None, render_upscale=40)
    
    # Experience replay buffer
    replay_buffer = deque(maxlen=MEMORY_SIZE)
    
    print(f"Initial Weights: {agent.weights}")
    print(f"Training for {NUM_EPISODES} episodes...")
    print("-" * 60)
    
    episode_rewards = []
    episode_lines = []
    
    for episode in range(NUM_EPISODES):
        obs, info = env.reset()
        action_mask = info.get("action_mask", np.ones(obs.shape[0], dtype=np.int8))
        
        terminated = False
        truncated = False
        episode_reward = 0
        total_lines = 0
        steps = 0
        
        prev_features = None
        
        while not (terminated or truncated):
            steps += 1
            
            # Choose action
            action, chosen_features = agent.choose_action(obs, action_mask)
            
            if action is None:
                # No valid moves - game over
                if prev_features is not None:
                    replay_buffer.append((prev_features, -10, 0))
                break
            
            # Execute action
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_action_mask = info.get("action_mask", np.ones(next_obs.shape[0], dtype=np.int8))
            lines_cleared = info.get("lines_cleared", 0)
            
            episode_reward += reward
            total_lines += lines_cleared
            
            # Update features with actual lines cleared
            chosen_features = agent.get_features_from_obs(obs[action], lines_cleared)
            
            # Compute next best Q
            if not terminated:
                best_next_q = compute_best_next_q(agent, next_obs, next_action_mask)
            else:
                best_next_q = 0
                reward -= 10  # Death penalty
            
            # Store experience
            replay_buffer.append((chosen_features, reward, best_next_q))
            prev_features = chosen_features
            
            # Batch learning from replay buffer
            if len(replay_buffer) >= BATCH_SIZE:
                mini_batch = random.sample(list(replay_buffer), BATCH_SIZE)
                for (feat, rew, next_q) in mini_batch:
                    agent.learn(feat, rew, next_q)
            
            # Update state
            obs = next_obs
            action_mask = next_action_mask
        
        # End of episode
        agent.decay_epsilon()
        episode_rewards.append(episode_reward)
        episode_lines.append(total_lines)
        
        # Logging
        avg_reward = np.mean(episode_rewards[-50:]) if len(episode_rewards) >= 50 else np.mean(episode_rewards)
        print(f"Ep {episode:3d} | Weights: {np.round(agent.weights, 3)} | "
              f"Score: {episode_reward:6.0f} | Lines: {total_lines:3d} | "
              f"Avg50: {avg_reward:7.1f} | ε: {agent.epsilon:.3f}")
    
    # Final stats
    print("-" * 60)
    print(f"Training complete!")
    print(f"Final Weights: {agent.weights}")
    print(f"Average Score (last 50): {np.mean(episode_rewards[-50:]):.1f}")
    print(f"Best Score: {max(episode_rewards)}")
    
    if SAVE_WEIGHTS:
        np.save("linear_q_weights_grouped.npy", agent.weights)
        print(f"Weights saved to linear_q_weights_grouped.npy")
    
    return agent


if __name__ == "__main__":
    main()
