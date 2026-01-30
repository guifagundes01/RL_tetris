import argparse
import random

import gymnasium as gym
import numpy as np
import torch

from tetris_gymnasium.envs import Tetris
from tetris_gymnasium.wrappers.grouped import GroupedActionsObservations
from tetris_gymnasium.wrappers.observation import FeatureVectorObservation

from train_lin_grouped_dqn import QNetwork

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="runs/train_lin_grouped_github/phi_pi__1__1769719089/train_lin_grouped_github.cleanrl_model")
    parser.add_argument("--env-id", type=str, default="tetris_gymnasium/Tetris")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--render-upscale", type=int, default=40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


    # env = gym.make(args.env_id, render_mode="rgb_array", gravity=False)
    # # env = gym.make(
    # #     args.env_id,
    # #     render_mode="human",
    # #     # gravity=True,
    # #     render_upscale=args.render_upscale,
    # # )
    # env = GroupedActionsObservations(env, observation_wrappers=[FeatureVectorObservation(env, report_height=True, report_max_height=True, report_holes=True, report_bumpiness=True)])



    env = gym.make(args.env_id, render_mode="human", gravity=False)
            # FeatureVectorObservation: [heights(10), max_height(1), holes(1), bumpiness(1)]
    env = GroupedActionsObservations(
        env, observation_wrappers=[FeatureVectorObservation(env)]
    )
    # env = gym.wrappers.RecordVideo(env, f"videos/run_trained_grouped")

    model = QNetwork(type("EnvWrap", (), {"single_observation_space": env.observation_space})())
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    model.eval()

    obs, info = env.reset(seed=args.seed)
    action_mask = info.get("action_mask", np.ones(obs.shape[0], dtype=np.int8))
    terminated = False
    truncated = False
    total_reward = 0.0

    while not (terminated or truncated):
        env.render()
        cv2.waitKey(1)
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        if obs_t.ndim == 2:
            obs_t = obs_t.unsqueeze(0)

        q_values = model(obs_t).squeeze(-1)
        if q_values.ndim == 1:
            q_values = q_values.unsqueeze(0)

        action_mask_t = torch.as_tensor(action_mask, dtype=torch.bool)
        if action_mask_t.ndim == 1:
            action_mask_t = action_mask_t.unsqueeze(0)

        masked_q = q_values.masked_fill(~action_mask_t, -1e9)
        if random.random() < args.epsilon:
            valid = torch.where(action_mask_t[0])[0].cpu().numpy()
            action = int(np.random.choice(valid))
        else:
            action = int(torch.argmax(masked_q, dim=1)[0].item())

        obs, reward, terminated, truncated, info = env.step(action)
        action_mask = info.get("action_mask", np.ones(obs.shape[0], dtype=np.int8))
        total_reward += float(reward)

    print(f"Final score: {total_reward}")
    env.close()


if __name__ == "__main__":
    main()
