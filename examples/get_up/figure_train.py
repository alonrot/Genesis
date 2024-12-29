import argparse
import os
import pickle
import shutil
from datetime import datetime

from figure_env import FigureEnv, get_train_cfg, get_cfgs
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from torch.utils.tensorboard import SummaryWriter

import torch
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="getup")
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=500)
    args = parser.parse_args()

    backend = gs.constants.backend.cpu if device == "cpu" else gs.constants.backend.gpu
    gs.init(logging_level="info", backend=backend)

    log_dir = f"logs/{args.exp_name}/{datetime.now().strftime('%Y%m%d_%H%M%S')}/"
    env_cfg, obs_cfg, reward_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=log_dir, flush_secs=10)

    env = FigureEnv(
        num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, device=device, writer=writer
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=device)

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()

"""
# training
python examples/locomotion/go2_train.py
"""
