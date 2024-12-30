import argparse
import os
import pickle
import shutil
from datetime import datetime

from figure_env import FigureEnv, get_cfgs, add2tensorboard
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

    env_cfg, obs_cfg, reward_cfg = get_cfgs()

    env = FigureEnv(
        num_envs=args.n_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, device=device, show_viewer=True,
    )

    # Create log_dir name using time of day as YYYYMMDD_HHmmSS
    log_dir = f"logs/{args.exp_name}/{datetime.now().strftime('%Y%m%d_%H%M%S')}/"
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
            actions_rand = torch.zeros_like(actions_rand)
            obs, _, rews, dones, infos = env.step(actions_rand)

            iter += 1

if __name__ == "__main__":
    main()