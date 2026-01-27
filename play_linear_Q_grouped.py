"""
Play Tetris with a trained Linear Q agent using GroupedActionsObservations.
Much faster than the deepcopy-based version.
"""
import numpy as np
import cv2
import argparse

from policy_linear_Q_grouped import LinearQAgentGrouped, create_grouped_env


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to saved weights .npy file")
    parser.add_argument("--render-upscale", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-episodes", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Initialize agent
    agent = LinearQAgentGrouped()
    agent.epsilon = 0.0  # Pure greedy for evaluation
    
    # Load weights
    if args.weights:
        agent.weights = np.load(args.weights)
        print(f"Loaded weights from {args.weights}: {agent.weights}")
    else:
        # Default good weights from training
        agent.weights = np.array([0.5, -0.01, -0.5, -0.5])
        print(f"Using default weights: {agent.weights}")
    
    # Create environment
    env = create_grouped_env(render_mode="human", render_upscale=args.render_upscale)
    
    all_scores = []
    all_lines = []
    
    for ep in range(args.num_episodes):
        obs, info = env.reset(seed=args.seed + ep)
        action_mask = info.get("action_mask", np.ones(obs.shape[0], dtype=np.int8))
        
        terminated = False
        truncated = False
        total_reward = 0
        total_lines = 0
        
        while not (terminated or truncated):
            env.render()
            cv2.waitKey(1)
            
            # Choose best action using feature vectors
            # obs shape: (n_actions, 13) from FeatureVectorObservation
            action, _ = agent.choose_action(obs, action_mask)
            
            if action is None:
                break
            
            # Execute
            obs, reward, terminated, truncated, info = env.step(action)
            action_mask = info.get("action_mask", np.ones(obs.shape[0], dtype=np.int8))
            
            total_reward += reward
            total_lines += info.get("lines_cleared", 0)
            
            print(f"Episode {ep+1} | Score: {total_reward:.0f} | Lines: {total_lines}", end='\r', flush=True)
        
        print(f"\nEpisode {ep+1} Final Score: {total_reward:.0f} | Lines: {total_lines}")
        all_scores.append(total_reward)
        all_lines.append(total_lines)
    
    env.close()
    
    # Summary
    if args.num_episodes > 1:
        print("\n" + "=" * 40)
        print(f"Episodes: {args.num_episodes}")
        print(f"Average Score: {np.mean(all_scores):.1f} +/- {np.std(all_scores):.1f}")
        print(f"Average Lines: {np.mean(all_lines):.1f} +/- {np.std(all_lines):.1f}")
        print(f"Best Score: {max(all_scores):.0f}")


if __name__ == "__main__":
    main()
