import torch
import math
import numpy as np
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat

import os
if 'figva' in os.uname()[1]:
    MJCF_PATH = "/shared/home/alonrot/project-x/robot_config/sim/b_sample/b_sample.xml"
else:
    MJCF_PATH = "/Users/alonrot/work_figure_ai/ws/project-x/robot_config/sim/b_sample/b_sample.xml"

from genesis.skeleton_properties import KP,  KD, torque_lb, torque_ub

from genesis.pose_library import crawl_pose_elbows_semi_flexed, t_pose_ground_random, t_pose, t_pose_ground, t_pose_arms_up, ready_to_push, push_up_halfway, push_up, to_crawl, downward_facing_dog, joint_names_fingers

def get_train_cfg(exp_name, max_iterations):

    train_cfg_dict = {
        "algorithm": {
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.1,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.0003,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 16,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [512, 512, 256],
            "critic_hidden_dims": [512, 512, 256],
            "init_noise_std": 1.0,
        },
        "runner": {
            "algorithm_class_name": "PPO",
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "num_steps_per_env": 256,
            "policy_class_name": "ActorCritic",
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
            "runner_class_name": "runner_class_name",
            "save_interval": 50,
        },
        "runner_class_name": "OnPolicyRunner",
        "seed": 1,
    }

    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        "num_actions": 14, # NOTE: For now, set as many as dofs. Later, exclude neck and others
        # termination
        "termination_if_roll_greater_than": 10,  # degree
        "termination_if_yaw_greater_than": 10,
        # base pose
        "base_init_pos": [0.0, 0.0, 0.4],
        "base_init_quat": [0.7071, 0.0, 0.7071, 0.0],
        "episode_length_s": 20.0,
        "action_scale": 0.2,
        "clip_actions": 100.0,
    }
    obs_cfg = {
        "num_obs": 59,
        "obs_scales": {
            "lin_vel": 1.0,
            "ang_vel": 1.0,
            "base_euler": 0.1,
            "base_pos": 0.1,
            "projected_gravity": 1.0,
            "dof_pos": 1./3.1415,
            "dof_vel": 1.0,
            "actions": 0.25,
        },
    }
    reward_cfg = {
        "tracking_sigma": 0.2,
        "reward_scales": {
            "zero_lateral_base_vel": 0.01,
            "zero_base_yaw_twist": 0.01,
            # "action_rate": 0.5,
            "base_sideways_tilt": 2.0, # gravity-based
            # "com_position_rt_base": 1.0,
            # "com_position_rt_base_terminal": 1.0,
            # "final_body_pose_terminal": 10.0,
            "final_body_pose": 50.0,
            "early_termination_base_yaw_tilt": 50.0,
            "early_termination_base_roll_tilt": 50.0,
        },
    }

    return env_cfg, obs_cfg, reward_cfg


learnable_joints = [
    "left.hip_y",
    "left.knee",
    "left.ankle_y",
    "left.shoulder_j1",
    "left.elbow",
    "left.wrist_roll",
    "left.wrist_yaw",
    "right.hip_y",
    "right.knee",
    "right.ankle_y",
    "right.shoulder_j1",
    "right.elbow",
    "right.wrist_roll",
    "right.wrist_yaw",
]

def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


def add2tensorboard(writer, infos, iter):

    for key, value in infos.items():

        if not isinstance(value, dict):

            if key in ["base_lin_vel", "base_ang_vel", "base_euler", "projected_gravity"]:
                value_one_env = value[0] # take the first environment
                writer.add_scalars('Observations/' + key, {"x": value_one_env[0],"y": value_one_env[1],"z": value_one_env[2]}, iter)
            elif key in ["reset", "episode_length_buf"]:
                value_one_env = value[0] # take the first environment
                writer.add_scalar('Observations/' + key, value_one_env, iter)

        elif key == "episode_sums":
            for k, v in value.items():
                value_one_env = v[0] # take the first environment
                writer.add_scalar('Rewards/' + k, value_one_env, iter)

class FigureEnv:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, show_viewer=False, device="cpu", writer=None):
        self.device = torch.device(device)

        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        # assert self.num_actions == len(KP) == len(KD), "Number of actions must match number of joints (for now)"

        self.simulate_action_latency = False  # there is a 1 step latency on real robot
        self.dt = 0.02  # control frequence on real robot is 50hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        self.show_viewer = show_viewer

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(n_rendered_envs=1 if not show_viewer else num_envs),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
                enable_self_collision=True,
                gravity=(0.0, 0.0, -9.81),
            ),
            show_viewer=self.show_viewer,
        )

        # add plain
        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

        # add robot
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=self.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=self.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(file=MJCF_PATH,
            pos   = (0, 0, 0.3),
            euler = (0, 90, 0), # we follow scipy's extrinsic x-y-z rotation convention, in degrees,
            # quat  = (1.0, 0.0, 0.0, 0.0), # we use w-x-y-z convention for quaternions,
            scale = 1.0),
        )

        assert len(KP) == len(KD)
        joint_names = KP.keys()
        assert all(name in KD.keys() for name in joint_names)

        # build
        if self.show_viewer:
            self.scene.build(n_envs=self.num_envs, env_spacing=(2.0, 2.0))
        else:
            self.scene.build(n_envs=self.num_envs)

        # names to indices
        self.motor_dofs = [self.robot.get_joint(name).dof_idx_local for name in joint_names]

        # Indices of the controlled joints
        self.idx_controlled_joints = [self.robot.get_joint(name).dof_idx_local for name in learnable_joints]

        # PD control parameters
        # self.robot.set_dofs_kp([self.env_cfg["kp"]] * self.num_actions, self.motor_dofs)
        # self.robot.set_dofs_kv([self.env_cfg["kd"]] * self.num_actions, self.motor_dofs)

        # set positional gains
        self.robot.set_dofs_kp(
            kp             = np.array([KP[joint_name] for joint_name in joint_names]),
            dofs_idx_local = self.motor_dofs,
        )
        # set velocity gains
        self.robot.set_dofs_kv(
            kv             = np.array([KD[joint_name] for joint_name in joint_names]),
            dofs_idx_local = self.motor_dofs,
        )

        # set force range for safety
        self.robot.set_dofs_force_range(
            lower          = np.array([torque_lb[joint_name] for joint_name in joint_names]),
            upper          = np.array([torque_ub[joint_name] for joint_name in joint_names]),
            dofs_idx_local = self.motor_dofs,
        )

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt # same as scaling the average reward by the total time (i.e., longer episodes, higher reward), while the average reward is the sum of instantaneous rewards divided by the number of time steps
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

        print("self.reward_scales: ", self.reward_scales)

        # initialize buffers
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=gs.tc_float).repeat(
            self.num_envs, 1
        )
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.rew_buf_terminal = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)
        # self.dof_pos = torch.zeros_like(self.actions)
        # self.dof_vel = torch.zeros_like(self.actions)
        # self.last_dof_vel = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros((self.num_envs, len(self.motor_dofs)), device=self.device, dtype=gs.tc_float)
        self.dof_vel = torch.zeros((self.num_envs, len(self.motor_dofs)), device=self.device, dtype=gs.tc_float)
        self.last_dof_vel = torch.zeros((self.num_envs, len(self.motor_dofs)), device=self.device, dtype=gs.tc_float)
        self.base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.default_dof_pos = torch.tensor(
            [crawl_pose_elbows_semi_flexed[name] for name in joint_names],
            device=self.device,
            dtype=gs.tc_float,
        )
        self.terminal_dof_pos = torch.tensor(
            [crawl_pose_elbows_semi_flexed[name] for name in joint_names],
            device=self.device,
            dtype=gs.tc_float,
        )
        self.extras = dict()  # extra information for logging
        
        # Tensorboard writer
        self.writer = writer
        self.global_counter = 0

    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions

        # Print max, min, mean and std of actions
        # print("actions: ", actions)
        # print("max: ", torch.max(actions))
        # print("min: ", torch.min(actions))
        # print("mean: ", torch.mean(actions))
        # print("std: ", torch.std(actions))

        
        # Learn deviations from the initial body pose
        # target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos

        # # Learn delta actions
        # target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.dof_pos

        # Learn delta actions for controlled joints only
        target_controlled_joints_pos_current_plus_actions = exec_actions * self.env_cfg["action_scale"] + self.robot.get_dofs_position(self.idx_controlled_joints)

        target_joints_pos_to_send = self.robot.get_dofs_position(self.motor_dofs)
        # print("target_controlled_joints_pos_current_plus_actions.shape: ", target_controlled_joints_pos_current_plus_actions.shape)
        # print("target_joints_pos_to_send.shape: ", target_joints_pos_to_send.shape)
        
        # Add target_controlled_joints_pos_current_plus_actions to target_joints_pos_to_send in the right indices
        # print("self.idx_controlled_joints: ", self.idx_controlled_joints)
        # print("self.motor_dofs: ", self.motor_dofs)
        for idx in self.idx_controlled_joints:
            idx_local = self.motor_dofs.index(idx)
            target_joints_pos_to_send[:,idx_local] = target_controlled_joints_pos_current_plus_actions[:,self.idx_controlled_joints.index(idx)]


        # print("target_joints_pos_to_send: ", target_joints_pos_to_send)
        # print("target_joints_pos_to_send.shape: ", target_joints_pos_to_send.shape)

        self.robot.control_dofs_position(target_joints_pos_to_send, self.motor_dofs)
        




        # To close the fingers, we need to expand self.motor_dofs to incldue them - otherwise, we can't control them
        # joint_names_with_fingers = list(joint_names) + joint_names_fingers
        # self.motor_dofs_with_fingers = [self.robot.get_joint(name).dof_idx_local for name in joint_names_with_fingers]
        # target_joints_pos_to_send_plus_closed_fingers = torch.cat((target_joints_pos_to_send, torch.tensor([[1.74533, 2.234025, 2.234025, 2.234025, 2.234025, 2.234025, 1.74533, 2.234025, 2.234025, 2.234025, 2.234025, 2.234025]])), axis=1)
        # self.robot.control_dofs_position(target_joints_pos_to_send_plus_closed_fingers, self.motor_dofs_with_fingers)

        # self.robot.control_dofs_position(target_dof_pos, self.motor_dofs)
        self.scene.step()

        # update buffers
        self.episode_length_buf += 1
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        self.base_euler = quat_to_xyz( # [deg]
            transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat)
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)

        # check termination and reset
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 2]) > self.env_cfg["termination_if_yaw_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=self.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        # compute reward
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew


        # compute observations
        self.obs_buf = torch.cat(
            [
                self.base_lin_vel * self.obs_scales["lin_vel"],  # 3
                self.base_ang_vel * self.obs_scales["ang_vel"],  # 3
                self.base_euler * self.obs_scales["base_euler"],  # 3
                self.base_pos * self.obs_scales["base_pos"],  # 3
                self.projected_gravity * self.obs_scales["projected_gravity"],  # 3
                self.dof_pos * self.obs_scales["dof_pos"],  # 30
                self.actions * self.obs_scales["actions"],  # 14
            ],
            axis=-1,
        )

        # print("self.obs_buf: ", self.obs_buf)
        # print("self.actions: ", self.actions)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

        # Collect observation as info
        self.extras["base_ang_vel"] = self.base_ang_vel
        self.extras["base_lin_vel"] = self.base_lin_vel
        self.extras["base_euler"] = self.base_euler
        self.extras["base_pos"] = self.base_pos
        self.extras["projected_gravity"] = self.projected_gravity
        self.extras["dof_pos"] = self.dof_pos
        self.extras["dof_vel"] = self.dof_vel
        self.extras["actions"] = self.actions
        self.extras["last_actions"] = self.last_actions
        self.extras["episode_length_buf"] = self.episode_length_buf
        self.extras["reset"] = self.reset_buf
        self.extras["episode_sums"] = self.episode_sums
        
        # if self.writer is not None:
        #     add2tensorboard(self.writer, self.extras, self.global_counter)
        #     self.global_counter += 1

        # if torch.any(self.reset_buf):
        #     print("self.episode_length_buf: ", self.episode_length_buf)

        return self.obs_buf, None, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return None

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # reset dofs
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motor_dofs,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        # reset base
        self.base_pos[envs_idx] = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.robot.zero_all_dofs_velocity(envs_idx)

        # reset buffers
        self.last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.obs_buf, None

    # ------------ reward functions----------------
    def _reward_zero_lateral_base_vel(self):
        # Penalize lateral base velocity
        # import pdb; pdb.set_trace()
        # lin_vel_error = torch.sum(torch.square(self.base_lin_vel[:, 1]), dim=1)
        lin_vel_error = torch.sum(torch.square(self.base_lin_vel[:, 1:2]), dim=1)
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_zero_base_yaw_twist(self):
        # Penalize yaw twist component
        # import pdb; pdb.set_trace()
        ang_vel_error = torch.sum(torch.square(self.base_ang_vel[:, 2:3]), dim=1)
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])

    # def _reward_action_rate(self):
    #     # Penalize changes in actions
    #     # import pdb; pdb.set_trace()
    #     return -torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    # def _reward_base_pitch_yaw_tilt(self):
    #     # Penalize base tilting: Assuming that the torso will be mostly rotated,
    #     # we penalize pitch and yaw tilting in Base frame coordinates.
    #     base_pitch_tilt_error = torch.sum(torch.square(self.projected_gravity[:,1::]), dim=1)
    #     return torch.exp(-base_pitch_tilt_error / self.reward_cfg["tracking_sigma"])

    def _reward_base_sideways_tilt(self):
        # Penalize base sideways tilt by projecting the gravity vector on the base frame and 
        # extacting the y component (sideways tilt)
        base_sideways_tilt = torch.sum(torch.square(self.projected_gravity[:,1:2]), dim=1)
        return torch.exp(-base_sideways_tilt / self.reward_cfg["tracking_sigma"])
    
    # def _reward_com_position_rt_base(self):
    #     # Penalize COM position such that it gets closer and closer to be between the feet
    #     return torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

    # def _reward_com_position_rt_base_terminal(self):
    #     # Terminal cost
    #     return torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

    # def _reward_final_body_pose_terminal(self):
    #     final_pos_error = torch.sum(torch.square(self.dof_pos[self.reset_buf] - self.terminal_dof_pos), dim=1)
        
    #     self.rew_buf_terminal[:] = 0.0
    #     self.rew_buf_terminal[self.reset_buf] = -final_pos_error

    #     #TODO(alonrot): Only apply this reward if the episode is terminated without timeout?
    #     return self.rew_buf_terminal
    
    def _reward_final_body_pose(self):
        final_pos_error = torch.sum(torch.square(self.dof_pos - self.terminal_dof_pos), dim=1)
        
        #TODO(alonrot): Only apply this reward if the episode is terminated without timeout?
        return -final_pos_error
    
    def _reward_early_termination_base_yaw_tilt(self):
        base_yaw_tilted = torch.abs(self.base_euler[:, 2]) > self.env_cfg["termination_if_yaw_greater_than"]
        return -base_yaw_tilted.float()
    
    def _reward_early_termination_base_roll_tilt(self):
        base_roll_tilted = torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]
        return -base_roll_tilted.float()
