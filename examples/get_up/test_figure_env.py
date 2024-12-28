import argparse
import os
import pickle
import shutil

from figure_env import FigureEnv, get_cfgs
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from torch.utils.tensorboard import SummaryWriter

import torch
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="getup")
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("--n_envs", type=int, default=1)
    args = parser.parse_args()

    backend = gs.constants.backend.cpu if device == "cpu" else gs.constants.backend.gpu
    gs.init(logging_level="info", backend=backend)

    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()

    env = FigureEnv(
        num_envs=args.n_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg, device=device, show_viewer=True,
    )

    log_dir = f"logs/{args.exp_name}"
    writer = SummaryWriter(log_dir=log_dir, flush_secs=10)

    gs.tools.run_in_another_thread(fn=run_sim_random_actions, args=(env, writer))

    if args.vis:
        env.scene.viewer.start()

def run_sim_random_actions(env, writer):
    obs, _ = env.reset()
    iter = 0
    with torch.no_grad():
        while True:
            actions_rand = torch.rand((env.num_envs, env.num_actions), device=env.device, dtype=gs.tc_float) * 2 - 1
            obs, _, rews, dones, infos = env.step(actions_rand)

            for key, value in infos.items():
                if not isinstance(value, dict):

                    if key in ["base_lin_vel", "base_ang_vel", "base_euler", "projected_gravity"]:
                        value_one_env = value[0] # take the first environment
                        writer.add_scalars('Observations/' + key, {"x": value_one_env[0],"y": value_one_env[1],"z": value_one_env[2]}, iter)

                elif key == "episode_sums":
                    for k, v in value.items():
                        writer.add_scalar('Rewards/' + k, v, iter)

            iter += 1

if __name__ == "__main__":
    main()