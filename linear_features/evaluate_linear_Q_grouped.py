"""
Evaluate Linear Q agent with GroupedActionsObservations.
No rendering for fast benchmarking.
"""
import numpy as np
import argparse
import time

from policy_linear_Q_grouped import LinearQAgentGrouped, create_grouped_env


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to saved weights .npy file")
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Initialize agent
    agent = LinearQAgentGrouped()
    agent.epsilon = 0.0  # Pure greedy
    
    if args.weights:
        agent.weights = np.load(args.weights)
        print(f"Loaded weights: {agent.weights}")
    else:
        agent.weights = np.array([0.5, -0.01, -0.5, -0.5])
        print(f"Using default weights: {agent.weights}")
    
    # Create environment (no rendering for speed)
    env = create_grouped_env(render_mode=None)
    
    rewards = []
    lines = []
    steps_list = []
    
    print(f"\nEvaluating for {args.num_episodes} episodes...")
    print("-" * 50)
    
    start_time = time.time()
    
    for ep in range(args.num_episodes):
        obs, info = env.reset(seed=args.seed + ep)
        action_mask = info.get("action_mask", np.ones(obs.shape[0], dtype=np.int8))
        
        terminated = False
        truncated = False
        total_reward = 0
        total_lines = 0
        steps = 0
        
        while not (terminated or truncated):
            steps += 1
            # obs shape: (n_actions, 13) from FeatureVectorObservation
            action, _ = agent.choose_action(obs, action_mask)
            
            if action is None:
                break
            
            obs, reward, terminated, truncated, info = env.step(action)
            action_mask = info.get("action_mask", np.ones(obs.shape[0], dtype=np.int8))
            total_reward += reward
            total_lines += info.get("lines_cleared", 0)
        
        rewards.append(total_reward)
        lines.append(total_lines)
        steps_list.append(steps)
        
        print(f"Episode {ep+1:3d}/{args.num_episodes} | Score: {total_reward:6.0f} | Lines: {total_lines:3d} | Steps: {steps:4d}")
    
    elapsed = time.time() - start_time
    env.close()
    
    # Results
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Episodes: {args.num_episodes}")
    print(f"Total time: {elapsed:.2f}s ({elapsed/args.num_episodes:.2f}s per episode)")
    print(f"Average Score: {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")
    print(f"Average Lines: {np.mean(lines):.2f} +/- {np.std(lines):.2f}")
    print(f"Average Steps: {np.mean(steps_list):.2f}")
    print(f"Min/Max Score: {np.min(rewards):.0f} / {np.max(rewards):.0f}")
    print(f"Min/Max Lines: {np.min(lines):.0f} / {np.max(lines):.0f}")


if __name__ == "__main__":
    main()
