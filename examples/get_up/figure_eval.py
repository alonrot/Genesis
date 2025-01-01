import argparse
import os
import pickle
import shutil

from figure_env import FigureEnv, add2tensorboard
from rsl_rl.runners import OnPolicyRunner
from datetime import datetime

import genesis as gs

import torch
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

from torch.utils.tensorboard import SummaryWriter

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="getup")
    parser.add_argument("--ckpt", type=int, default=100)
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("--rand_actions", action="store_true", default=False)
    parser.add_argument("-B", "--num_envs", type=int, default=1)
    args = parser.parse_args()

    backend = gs.constants.backend.cpu if device == "cpu" else gs.constants.backend.gpu
    gs.init(logging_level="info", backend=backend)

    log_dir = f"logs/{args.exp_name}/"
    env_cfg, obs_cfg, reward_cfg, train_cfg = pickle.load(open(f"logs/{args.exp_name}/cfgs.pkl", "rb"))

    writer = SummaryWriter(log_dir=log_dir, flush_secs=10)

    env = FigureEnv(
        num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, device=device, show_viewer=args.vis, writer=writer
    )

    print("Creating runner")
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=device)

    print("Loading model")
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)

    print("Getting inference policy")
    policy = runner.get_inference_policy(device=device)

    print("Running sim")

    if args.rand_actions:
        gs.tools.run_in_another_thread(fn=run_sim_random_actions, args=(env, policy, writer))
    else:
        gs.tools.run_in_another_thread(fn=run_sim, args=(env, policy, writer))
        
    if args.vis:
        env.scene.viewer.start()

def run_sim(env, policy, writer):
    obs, _ = env.reset()
    iter = 0
    with torch.no_grad():
        while True:
            actions = policy(obs)
            obs, _, rews, dones, infos = env.step(actions)

            # print("done", dones)
            # print("infos[episode_length_buf]: ", infos["episode_length_buf"])
            # print("infos[reset]: ", infos["reset"])

            add2tensorboard(writer, infos, iter)

            iter += 1


def run_sim_random_actions(env, policy, writer):
    obs, _ = env.reset()
    iter = 0
    with torch.no_grad():
        while True:
            actions_rand = torch.rand((env.num_envs, env.num_actions), device=env.device, dtype=gs.tc_float) * 2 - 1
            obs, _, rews, dones, infos = env.step(actions_rand)

            add2tensorboard(writer, infos, iter)

            iter += 1

if __name__ == "__main__":
    main()

"""
# evaluation
python examples/get_up/figure_eval.py -e getup --ckpt 100
"""