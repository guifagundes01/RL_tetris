"""Evaluate a trained grouped DQN model on Tetris."""
import argparse

import gymnasium as gym
import numpy as np
import torch

from tetris_gymnasium.envs import Tetris
from tetris_gymnasium.wrappers.grouped import GroupedActionsObservations
from tetris_gymnasium.wrappers.observation import FeatureVectorObservation

from train_lin_grouped_original import QNetwork

import cv2

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="train_lin_grouped.cleanrl_model")
    parser.add_argument("--env-id", type=str, default="tetris_gymnasium/Tetris")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-episodes", type=int, default=30)
    parser.add_argument("--epsilon", type=float, default=0.0, help="Exploration rate")
    parser.add_argument("--render-upscale", type=int, default=40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create environment
    render_mode = "rgb_array"
    env = gym.make(
        args.env_id,
        render_mode=render_mode,
        gravity=True,
    )
    env = GroupedActionsObservations(env, observation_wrappers=[FeatureVectorObservation(env, report_height=True, report_max_height=True, report_holes=True, report_bumpiness=True)])
    env = gym.wrappers.RecordEpisodeStatistics(env)

    # Load model
    model = QNetwork(type("EnvWrap", (), {"single_observation_space": env.observation_space})())
    model.load_state_dict(torch.load(args.model_path, map_location="cpu", weights_only=True))
    model.eval()

    # Store rewards and lines cleared for each episode
    reward_episodes = np.zeros(args.num_episodes)
    lines_episodes = np.zeros(args.num_episodes)

    for episode in range(args.num_episodes):
        obs, info = env.reset(seed=args.seed + episode)
        action_mask = info.get("action_mask", np.ones(obs.shape[0], dtype=np.int8))
        terminated = False
        truncated = False
        total_reward = 0.0
        total_lines = 0

        while not (terminated or truncated):
            env.render()
            cv2.waitKey(1)
            obs_t = torch.as_tensor(obs, dtype=torch.float32)
            if obs_t.ndim == 2:
                obs_t = obs_t.unsqueeze(0)

            with torch.no_grad():
                q_values = model(obs_t).squeeze(-1)
            if q_values.ndim == 1:
                q_values = q_values.unsqueeze(0)

            action_mask_t = torch.as_tensor(action_mask, dtype=torch.bool)
            if action_mask_t.ndim == 1:
                action_mask_t = action_mask_t.unsqueeze(0)

            masked_q = q_values.masked_fill(~action_mask_t, -1e9)

            # Epsilon-greedy action selection
            if np.random.random() < args.epsilon:
                valid = torch.where(action_mask_t[0])[0].cpu().numpy()
                action = int(np.random.choice(valid))
            else:
                action = int(torch.argmax(masked_q, dim=1)[0].item())

            obs, reward, terminated, truncated, info = env.step(action)
            action_mask = info.get("action_mask", np.ones(obs.shape[0], dtype=np.int8))
            total_reward += float(reward)
            total_lines += info.get("lines_cleared", 0)

            print(f"Episode {episode + 1} | Score: {total_reward:.0f} | Lines: {total_lines}", end='\r', flush=True)

        reward_episodes[episode] = total_reward
        lines_episodes[episode] = total_lines
        print(f"Episode {episode + 1}/{args.num_episodes} | Score: {total_reward:.0f} | Lines: {total_lines}")

    env.close()

    # Display results
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Episodes: {args.num_episodes}")
    print(f"Reward per episode: {reward_episodes}")
    print(f"Lines per episode: {lines_episodes}")
    print(f"Average reward: {np.mean(reward_episodes):.2f} +/- {np.std(reward_episodes):.2f}")
    print(f"Average lines: {np.mean(lines_episodes):.2f} +/- {np.std(lines_episodes):.2f}")
    print(f"Min/Max reward: {np.min(reward_episodes):.0f} / {np.max(reward_episodes):.0f}")
    print(f"Min/Max lines: {np.min(lines_episodes):.0f} / {np.max(lines_episodes):.0f}")


if __name__ == "__main__":
    main()
