import argparse
import os
import pickle
import shutil

from figure_env import FigureEnv
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

import torch
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

def get_train_cfg(exp_name, max_iterations):

    train_cfg_dict = {
        "algorithm": {
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 1.0,
        },
        "runner": {
            "algorithm_class_name": "PPO",
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "num_steps_per_env": 24,
            "policy_class_name": "ActorCritic",
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
            "runner_class_name": "runner_class_name",
            "save_interval": 100,
        },
        "runner_class_name": "OnPolicyRunner",
        "seed": 1,
    }

    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        "num_actions": 30, # NOTE: For now, set as many as dofs. Later, exclude neck and others
        # termination
        "termination_if_roll_greater_than": 10,  # degree
        "termination_if_pitch_greater_than": 10,
        # base pose
        "base_init_pos": [0.0, 0.0, 0.3],
        "base_init_quat": [0.7071, 0.0, 0.7071, 0.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
    }
    obs_cfg = {
        "num_obs": 45,
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }
    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.3,
        "feet_height_target": 0.075,
        "reward_scales": {
            "zero_lateral_base_vel": 1.0,
            "zero_base_yaw_twist": 0.2,
            "action_rate": -0.005,
            "base_pitch_yaw_tilt": 1.0,
            "com_position_rt_base": 1.0,
            "com_position_rt_base_terminal": 1.0,
            "final_body_pose_terminal": 1.0,
            # "similar_to_default": -0.1,
        },
    }
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [0.5, 0.5],
        "lin_vel_y_range": [0, 0],
        "ang_vel_range": [0, 0],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():

    gs.init(logging_level="info")

    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()

    env = FigureEnv(
        num_envs=2, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg, device=device
    )

    actions = torch.zeros((env.num_envs, env.num_actions), device=env.device, dtype=gs.tc_float)
    obs, _, rew, reset, extras = env.step(actions)

    print("obs", obs.shape)
    print("rew", rew.shape)
    print("reset", reset.shape)
    print("extras", extras.shape)


if __name__ == "__main__":
    main()