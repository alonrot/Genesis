import argparse
import os
import pickle
import shutil
from datetime import datetime

from figure_env import FigureEnv
from genesis.figure_cfg import get_train_cfg, get_cfgs
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from torch.utils.tensorboard import SummaryWriter

import torch
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, required=True)
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=500)
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("--ckpt", type=int, default=-1)

    # Override some parameters from existing training configuration
    parser.add_argument("--save_interval", type=int, default=-1)
    args = parser.parse_args()

    backend = gs.constants.backend.cpu if device == "cpu" else gs.constants.backend.gpu
    gs.init(logging_level="info", backend=backend)

    assert args.exp_name != "", "Please provide an experiment name"

    if args.ckpt != -1: # resume training
        log_dir = f"logs/{args.exp_name}/"
        env_cfg, obs_cfg, reward_cfg, train_cfg = pickle.load(open(f"logs/{args.exp_name}/cfgs.pkl", "rb"))
    
        # Override some parameters from existing training configuration
        if args.save_interval != -1:
            assert args.save_interval > 1, "Save interval must be greater than 1"
            train_cfg["runner"]["save_interval"] = args.save_interval
    else:
        log_dir = f"logs/{args.exp_name}/{datetime.now().strftime('%Y%m%d_%H%M%S')}/"
        env_cfg, obs_cfg, reward_cfg = get_cfgs()
        train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
    
    os.makedirs(log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=log_dir, flush_secs=10)

    env = FigureEnv(
        num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, device=device, writer=writer, show_viewer=args.vis,
    )

    print("Creating runner")
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=device)

    # If training is resumed, load the runner
    if args.ckpt != -1:
        print("Loading model")
        resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
        runner.load(resume_path)

    # If training is not resumed, save the configurations
    if args.ckpt == -1:
        print("Saving configurations")
        pickle.dump(
            [env_cfg, obs_cfg, reward_cfg, train_cfg],
            open(f"{log_dir}/cfgs.pkl", "wb"),
        )

    init_at_random_ep_len = False

    if args.vis:
        gs.tools.run_in_another_thread(fn=run, args=(runner, args.max_iterations, init_at_random_ep_len))
        env.scene.viewer.start()
    else:
        runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)

def run(runner, max_iterations, init_at_random_ep_len):
    runner.learn(num_learning_iterations=max_iterations, init_at_random_ep_len=init_at_random_ep_len)

if __name__ == "__main__":
    main()

"""
# training
python examples/locomotion/go2_train.py
"""
