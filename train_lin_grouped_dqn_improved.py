"""Improved DQN training script for grouped Tetris.

Key improvements over ``train_lin_grouped_original.py``:

- Trains a true action-value function Q(s, a) using per-action feature vectors.
- Uses a separate target network for computing TD targets (classic DQN).
- Uses ``max_a' Q_target(s', a')`` in the Bellman backup.
- Uses Huber loss and gradient clipping for more stable optimization.

This is still closely based on CleanRL's DQN implementation and the original
grouped setup so it should be easy to compare results.
"""

import os
import random
import time
from dataclasses import dataclass
from typing import Callable

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from gymnasium.spaces import Box
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

from tetris_gymnasium.envs import Tetris
from tetris_gymnasium.wrappers.grouped import GroupedActionsObservations
from tetris_gymnasium.wrappers.observation import FeatureVectorObservation


@dataclass
class Args:
    # Experiment / logging
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = True
    wandb_project_name: str = "tetris_gymnasium_grouped"
    wandb_entity: str = None
    capture_video: bool = True
    save_model: bool = False
    upload_model: bool = False
    hf_entity: str = ""
    video_epoch_interval: int = 500

    # Algorithm hyperparameters
    env_id: str = "tetris_gymnasium/Tetris"
    total_timesteps: int = 250000
    learning_rate: float = 5e-4  # slightly smaller LR for stability
    num_envs: int = 1
    buffer_size: int = 30000
    gamma: float = 0.99
    tau: float = 1.0
    target_network_frequency: int = 1000  # update target every N steps
    batch_size: int = 512
    start_e: float = 1.0
    end_e: float = 1e-3
    exploration_fraction: float = 0.25
    learning_starts: int = 3000
    train_frequency: int = 20
    max_grad_norm: float = 10.0


def make_env(env_id, seed, idx, capture_video, run_name, args: Args):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array", gravity=False)
            env = GroupedActionsObservations(
                env, observation_wrappers=[FeatureVectorObservation(env)]
            )
            env = gym.wrappers.RecordVideo(
                env,
                f"videos/{run_name}",
                episode_trigger=lambda x: x % args.video_epoch_interval == 0,
            )
        else:
            env = gym.make(env_id, render_mode="rgb_array", gravity=False)
            env = GroupedActionsObservations(
                env, observation_wrappers=[FeatureVectorObservation(env)]
            )

        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


class QNetwork(nn.Module):
    """Linear / MLP Q-network operating on per-action feature vectors.

    Input shape: (batch, n_actions, feature_dim)
    Output shape: (batch, n_actions, 1)   # Q(s, a) for each action
    """

    def __init__(self, env):
        super().__init__()
        # ``env`` is expected to have ``single_observation_space`` similar to vectorized envs
        if hasattr(env, "single_observation_space"):
            obs_space = env.single_observation_space
        else:
            obs_space = env.observation_space

        # We expect observations to be (n_actions, feature_dim)
        if isinstance(obs_space, Box):
            feature_dim = int(np.prod(obs_space.shape[-1:]))
        else:
            raise ValueError("Unsupported observation space type for QNetwork.")

        self.network = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self._create_weights()

    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        ``x`` can be:
        - (batch, n_actions, feat_dim)
        - (n_actions, feat_dim)
        """
        if x.ndim == 2:
            x = x.unsqueeze(0)  # (1, n_actions, feat_dim)

        b, n_actions, feat_dim = x.shape
        x_flat = x.view(b * n_actions, feat_dim)
        q_flat = self.network(x_flat)  # (b * n_actions, 1)
        q = q_flat.view(b, n_actions, 1)
        return q


def linear_schedule(start_e: float, end_e: float, duration: int, t: int) -> float:
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def main():
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            'Requires stable-baselines3>=2.0.0a1. '
            'Install with: poetry run pip install "stable_baselines3==2.0.0a1"'
        )

    args = tyro.cli(Args)

    # Run name
    greek_letters = [
        "alpha",
        "beta",
        "gamma",
        "delta",
        "epsilon",
        "zeta",
        "eta",
        "theta",
        "iota",
        "kappa",
        "lambda",
        "mu",
        "nu",
        "xi",
        "omicron",
        "pi",
        "rho",
        "sigma",
        "tau",
        "upsilon",
        "phi",
        "chi",
        "psi",
        "omega",
    ]
    run_name = f"{args.exp_name}/{random.choice(greek_letters)}_{random.choice(greek_letters)}__{args.seed}__{int(time.time())}"

    # Tracking with W&B + TensorBoard
    if args.track:
        import wandb

        run = wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
        run.log_code(
            os.path.normpath(
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "../tetris_gymnasium"
                )
            )
        )

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Env setup
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(args.env_id, args.seed + i, i, args.capture_video, run_name, args)
            for i in range(args.num_envs)
        ]
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"

    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    # Replay buffer over full grouped observations (per-action features)
    rb = ReplayBuffer(
        args.buffer_size,
        observation_space=envs.single_observation_space,
        action_space=envs.single_action_space,
        device=device,
        handle_timeout_termination=False,
    )

    start_time = time.time()

    obs, info = envs.reset(seed=args.seed)
    action_mask = info["action_mask"][0]

    epoch = 0
    global_step = 0
    epoch_lines_cleared = 0

    while global_step < args.total_timesteps:
        # Epsilon-greedy policy over Q(s, a)
        epsilon = linear_schedule(
            args.start_e,
            args.end_e,
            int(args.exploration_fraction * args.total_timesteps),
            global_step,
        )

        if random.random() < epsilon:
            actions = np.array(
                [
                    np.random.choice(np.where(action_mask == 1)[0])
                    for _ in range(envs.num_envs)
                ]
            )
        else:
            obs_t = torch.as_tensor(obs, device=device, dtype=torch.float32)
            q_values = q_network(obs_t).squeeze(-1)  # (num_envs, n_actions)

            action_mask_t = torch.as_tensor(
                info["action_mask"], device=device, dtype=torch.bool
            )
            if action_mask_t.ndim == 1:
                action_mask_t = action_mask_t.unsqueeze(0)

            q_values = q_values.masked_fill(~action_mask_t, -1e9)
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # Step environment
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        global_step += 1

        action_mask = infos["action_mask"][0]
        epoch_lines_cleared += infos["lines_cleared"][0]

        # Logging episodic stats
        if "final_info" in infos:
            for info_ep in infos["final_info"]:
                if info_ep and "episode" in info_ep:
                    print(
                        f"epoch={epoch}, "
                        f"timestep={global_step} ({round((global_step / args.total_timesteps)* 100, 2)}%), "
                        f"epsilon={epsilon:.3f}, "
                        f"episodic_return={info_ep['episode']['r']}, "
                        f"episodic_len={info_ep['episode']['l']}, "
                        f"episodic_lines={epoch_lines_cleared}"
                    )
                    writer.add_scalar(
                        "charts/episodic_return", info_ep["episode"]["r"], global_step
                    )
                    writer.add_scalar(
                        "charts/episodic_lines", epoch_lines_cleared, global_step
                    )
                    writer.add_scalar(
                        "charts/episodic_length", info_ep["episode"]["l"], global_step
                    )
                    epoch_lines_cleared = 0
                    epoch += 1

        # Handle truncations: swap in final_observation where needed
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]

        # Store transition in replay buffer
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        obs = next_obs

        # Training
        if global_step > args.learning_starts and global_step % args.train_frequency == 0:
            data = rb.sample(args.batch_size)

            with torch.no_grad():
                # Target network for next-state values
                next_q_values = target_network(data.next_observations).squeeze(-1)
                target_max, _ = next_q_values.max(dim=1)
                td_target = data.rewards.flatten() + args.gamma * target_max * (
                    1.0 - data.dones.flatten()
                )

            # Q(s, a) for actions actually taken
            q_values = q_network(data.observations).squeeze(-1)
            actions_t = data.actions.long().view(-1)
            if actions_t.ndim == 1:
                actions_t = actions_t.unsqueeze(-1)
            current_q = q_values.gather(1, actions_t).squeeze(-1)

            assert current_q.shape == td_target.shape

            loss = F.smooth_l1_loss(current_q, td_target)

            if global_step % 100 == 0:
                writer.add_scalar("losses/td_loss", loss.item(), global_step)
                writer.add_scalar(
                    "losses/target_q", td_target.mean().item(), global_step
                )
                writer.add_scalar(
                    "losses/old_q", current_q.mean().item(), global_step
                )
                sps = int(global_step / (time.time() - start_time))
                print("SPS:", sps)
                print("Loss", loss.item())
                writer.add_scalar("charts/SPS", sps, global_step)
                writer.add_scalar("schedule/epsilon", epsilon, global_step)

            optimizer.zero_grad()
            loss.backward()
            if args.max_grad_norm is not None and args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(q_network.parameters(), args.max_grad_norm)
            optimizer.step()

        # Target network update (hard/soft)
        if global_step > args.learning_starts and (
            global_step % args.target_network_frequency == 0
        ):
            for target_param, param in zip(
                target_network.parameters(), q_network.parameters()
            ):
                target_param.data.copy_(
                    args.tau * param.data + (1.0 - args.tau) * target_param.data
                )

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(q_network.state_dict(), model_path)
        print(f"model saved to {model_path}")

    envs.close()
    writer.close()


if __name__ == "__main__":
    main()

