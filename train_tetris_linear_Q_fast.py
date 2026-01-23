import argparse
import random
import time
from collections import deque

import gymnasium as gym
import numpy as np

from tetris_gymnasium.envs import Tetris
from tetris_gymnasium.wrappers.grouped import GroupedActionsObservations
from tetris_gymnasium.wrappers.observation import FeatureVectorObservation


def linear_schedule(start_e: float, end_e: float, duration: int, t: int) -> float:
    slope = (end_e - start_e) / max(duration, 1)
    return max(slope * t + start_e, end_e)


def make_env(env_id: str, seed: int, idx: int):
    def thunk():
        env = gym.make(env_id, render_mode=None, gravity=False)
        env = GroupedActionsObservations(
            env, observation_wrappers=[FeatureVectorObservation(env)]
        )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed + idx)
        return env

    return thunk


class LinearQAgent:
    def __init__(
        self,
        feature_dim: int,
        learning_rate: float,
        gamma: float,
        grad_clip: float,
    ):
        self.weights = np.zeros(feature_dim, dtype=np.float32)
        self.lr = learning_rate
        self.gamma = gamma
        self.grad_clip = grad_clip

    def q_values(self, obs: np.ndarray) -> np.ndarray:
        # obs shape: (num_envs, n_actions, feature_dim)
        return np.tensordot(obs, self.weights, axes=([2], [0]))

    def update_batch(self, features: np.ndarray, targets: np.ndarray) -> float:
        # features shape: (batch, feature_dim), targets shape: (batch,)
        features = features.astype(np.float32, copy=False)
        targets = targets.astype(np.float32, copy=False)
        preds = features @ self.weights
        errors = targets - preds
        if self.grad_clip > 0:
            errors = np.clip(errors, -self.grad_clip, self.grad_clip)
        grad = features.T @ errors / features.shape[0]
        if self.grad_clip > 0:
            grad = np.clip(grad, -self.grad_clip, self.grad_clip)
        self.weights += self.lr * grad
        return float(np.mean(errors**2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="tetris_gymnasium/Tetris")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--total-timesteps", type=int, default=250000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--buffer-size", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--learning-starts", type=int, default=3000)
    parser.add_argument("--train-frequency", type=int, default=10)
    parser.add_argument("--start-e", type=float, default=1.0)
    parser.add_argument("--end-e", type=float, default=0.05)
    parser.add_argument("--exploration-fraction", type=float, default=0.25)
    parser.add_argument("--death-penalty", type=float, default=10.0)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--log-interval", type=int, default=200)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed, i) for i in range(args.num_envs)]
    )

    obs, info = envs.reset(seed=args.seed)
    action_mask = info.get("action_mask", np.ones(obs.shape[:2], dtype=np.int8))

    agent = LinearQAgent(
        feature_dim=obs.shape[-1],
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        grad_clip=args.grad_clip,
    )

    replay_buffer = deque(maxlen=args.buffer_size)
    start_time = time.time()

    global_step = 0
    last_log = 0
    last_episode_return = None
    last_episode_len = None
    while global_step < args.total_timesteps:
        epsilon = linear_schedule(
            args.start_e,
            args.end_e,
            int(args.exploration_fraction * args.total_timesteps),
            global_step,
        )

        obs = obs.astype(np.float32, copy=False)
        q_values = agent.q_values(obs)
        actions = np.empty(envs.num_envs, dtype=np.int64)
        for i in range(envs.num_envs):
            if random.random() < epsilon:
                valid = np.where(action_mask[i] == 1)[0]
                actions[i] = int(np.random.choice(valid))
            else:
                masked = np.where(action_mask[i] == 1, q_values[i], -1e9)
                actions[i] = int(np.argmax(masked))

        chosen_features = obs[np.arange(envs.num_envs), actions].astype(
            np.float32, copy=False
        )
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        done = np.logical_or(terminations, truncations)

        if args.death_penalty > 0:
            rewards = rewards - args.death_penalty * terminations.astype(np.float32)

        next_action_mask = infos.get(
            "action_mask", np.ones(next_obs.shape[:2], dtype=np.int8)
        )
        next_q_values = agent.q_values(next_obs)
        masked_next = np.where(next_action_mask == 1, next_q_values, -1e9)
        next_max_q = np.max(masked_next, axis=1)
        next_max_q[done] = 0.0

        targets = rewards + args.gamma * next_max_q
        for i in range(envs.num_envs):
            replay_buffer.append((chosen_features[i], targets[i]))

        if len(replay_buffer) >= args.batch_size and global_step >= args.learning_starts:
            if global_step % args.train_frequency == 0:
                batch = random.sample(replay_buffer, args.batch_size)
                batch_features = np.stack([b[0] for b in batch], axis=0)
                batch_targets = np.array([b[1] for b in batch], dtype=np.float32)
                loss = agent.update_batch(batch_features, batch_targets)

        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    ep = info["episode"]
                    last_episode_return = ep.get("r")
                    last_episode_len = ep.get("l")
                    print(
                        f"timestep={global_step} ({round((global_step / args.total_timesteps) * 100, 2)}%), "
                        f"epsilon={epsilon:.3f}, "
                        f"episodic_return={ep['r']}, "
                        f"episodic_len={ep['l']}, "
                        f"score={ep['r']}"
                    )

        if global_step - last_log >= args.log_interval:
            sps = int(global_step / max(time.time() - start_time, 1e-6))
            score_str = (
                f" | score={last_episode_return}"
                if last_episode_return is not None
                else ""
            )
            print(
                f"SPS={sps} | epsilon={epsilon:.3f} | buffer={len(replay_buffer)}"
                f"{score_str}"
            )
            last_log = global_step

        obs = next_obs
        action_mask = next_action_mask
        global_step += envs.num_envs

    envs.close()
