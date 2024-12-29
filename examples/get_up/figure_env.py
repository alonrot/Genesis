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

# from genesis.skeleton_properties import KP, KD, torque_lb, torque_ub

# from genesis.pose_library import crawl_pose_elbows_semi_flexed, t_pose_ground_random, t_pose, t_pose_ground, t_pose_arms_up, ready_to_push, push_up_halfway, push_up, to_crawl, downward_facing_dog

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
        "termination_if_roll_greater_than": 20,  # degree
        "termination_if_pitch_greater_than": 20,
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
        "num_obs": 72,
        "obs_scales": {
            "lin_vel": 1.0,
            "ang_vel": 1.0,
            "base_euler": 0.1,
            "projected_gravity": 1.0,
            "dof_pos": 1.0,
            "dof_vel": 1.0,
            "actions": 1.0,
        },
    }
    reward_cfg = {
        "tracking_sigma": 0.2,
        "reward_scales": {
            "zero_lateral_base_vel": 1.0,
            "zero_base_yaw_twist": 1.0,
            "action_rate": 0.005,
            "base_pitch_yaw_tilt": 1.0,
            "com_position_rt_base": 1.0,
            "com_position_rt_base_terminal": 1.0,
            # "final_body_pose_terminal": 10.0,
            "final_body_pose": 50.0,
        },
    }

    return env_cfg, obs_cfg, reward_cfg

KP = {
    "left.hip_z": 300,
    "left.hip_x": 600,
    "left.hip_y": 600,
    "left.knee": 800,
    "left.ankle_y": 150,
    "left.ankle_x": 80,
    "left.shoulder_j1": 400,
    "left.shoulder_j2": 200,
    "left.upper_arm_twist": 50,
    "left.elbow": 350,
    "left.wrist_roll": 25,
    "left.wrist_pitch": 25,
    "left.wrist_yaw": 50,
    "right.hip_z": 300,
    "right.hip_x": 600,
    "right.hip_y": 600,
    "right.knee": 800,
    "right.ankle_y": 150,
    "right.ankle_x": 80,
    "right.shoulder_j1": 400,
    "right.shoulder_j2": 200,
    "right.upper_arm_twist": 50,
    "right.elbow": 350,
    "right.wrist_roll": 25,
    "right.wrist_pitch": 25,
    "right.wrist_yaw": 50,
    "spine_x": 500,
    "spine_z": 500,
    "neck_no": 12.5,
    "neck_yes": 35
}

joint_names = [joint_name for joint_name in KP.keys()]


KD = {
    "left.hip_z": 15,
    "left.hip_x": 15,
    "left.hip_y": 30,
    "left.knee": 30,
    "left.ankle_y": 5,
    "left.ankle_x": 2,
    "left.shoulder_j1": 8,
    "left.shoulder_j2": 15,
    "left.upper_arm_twist": 5.0,
    "left.elbow": 4,
    "left.wrist_roll": 0.75,
    "left.wrist_pitch": 0.75,
    "left.wrist_yaw": 0.75,
    "right.hip_z": 15,
    "right.hip_x": 15,
    "right.hip_y": 30,
    "right.knee": 30,
    "right.ankle_y": 5,
    "right.ankle_x": 2,
    "right.shoulder_j1": 8,
    "right.shoulder_j2": 15,
    "right.upper_arm_twist": 5.0,
    "right.elbow": 4,
    "right.wrist_roll": 0.75,
    "right.wrist_pitch": 0.75,
    "right.wrist_yaw": 0.75,
    "spine_x": 35,
    "spine_z": 35,
    "neck_no": 1,
    "neck_yes": 1
}

torque_lb = {
    "neck_yes": -19.3,
    "neck_no": -19.3,
    "left.shoulder_j1": -73.6848,
    "right.shoulder_j1": -73.6848,
    "left.shoulder_j2": -73.6848,
    "right.shoulder_j2": -73.6848,
    "left.upper_arm_twist": -73.6848,
    "right.upper_arm_twist": -73.6848,
    "left.elbow": -73.6848,
    "right.elbow": -73.6848,
    "left.wrist_roll": -19.3,
    "right.wrist_roll": -19.3,
    "left.wrist_yaw": -19.3,
    "right.wrist_yaw": -19.3,
    "left.wrist_pitch": -19.3,
    "right.wrist_pitch": -19.3,
    "spine_z": -272.827,
    "spine_x": -272.827,
    "left.hip_y": -272.827,
    "right.hip_y": -272.827,
    "left.hip_x": -272.827,
    "right.hip_x": -272.827,
    "left.hip_z": -272.827,
    "right.hip_z": -272.827,
    "left.knee": -272.827,
    "right.knee": -272.827,
    "left.ankle_y": -124.764,
    "right.ankle_y": -124.764,
    "left.ankle_x": -22.356,
    "right.ankle_x": -22.356,
}


torque_ub = {
    "neck_yes": 19.3,
    "neck_no": 19.3,
    "left.shoulder_j1": 73.6848,
    "right.shoulder_j1": 73.6848,
    "left.shoulder_j2": 73.6848,
    "right.shoulder_j2": 73.6848,
    "left.upper_arm_twist": 73.6848,
    "right.upper_arm_twist": 73.6848,
    "left.elbow": 73.6848,
    "right.elbow": 73.6848,
    "left.wrist_roll": 19.3,
    "right.wrist_roll": 19.3,
    "left.wrist_yaw": 19.3,
    "right.wrist_yaw": 19.3,
    "left.wrist_pitch": 19.3,
    "right.wrist_pitch": 19.3,
    "spine_z": 272.827,
    "spine_x": 272.827,
    "left.hip_y": 272.827,
    "right.hip_y": 272.827,
    "left.hip_x": 272.827,
    "right.hip_x": 272.827,
    "left.hip_z": 272.827,
    "right.hip_z": 272.827,
    "left.knee": 272.827,
    "right.knee": 272.827,
    "left.ankle_y": 124.764,
    "right.ankle_y": 124.764,
    "left.ankle_x": 22.356,
    "right.ankle_x": 22.356,
}


downward_facing_dog = {
  "left.hip_y": -2.70121,
  "left.hip_x": 0.446085,
  "left.hip_z": -0.142601,
  "left.knee": 1.11906,
  "left.ankle_y": -0.0428307,
  "left.ankle_x": -0.208207,
  "right.hip_y": -2.70135,
  "right.hip_x": -0.445718,
  "right.hip_z": 0.141505,
  "right.knee": 1.12053,
  "right.ankle_y": -0.0409522,
  "right.ankle_x": 0.207152,
  "spine_x": 0.012726,
  "spine_z": -0.0124875,
  "left.shoulder_j1": -1.61139,
  "left.shoulder_j2": -0.395551,
  "left.upper_arm_twist": -0.27558,
  "left.elbow": 0.00866535,
  "left.wrist_roll": 0.00441669,
  "left.wrist_pitch": 0.0826308,
  "left.wrist_yaw": -0.253052,
  "right.shoulder_j1": 1.64751,
  "right.shoulder_j2": 0.372409,
  "right.upper_arm_twist": -0.264508,
  "right.elbow": -0.0629978,
  "right.wrist_roll": 0.00408184,
  "right.wrist_pitch": -0.0856984,
  "right.wrist_yaw": 0.286192,
  "neck_no": -7.58171e-05,
  "neck_yes": 0.000681877,
}

crawl_pose_elbows_semi_flexed = {
    # Left side
    "left.hip_z": -0.15,
    "left.hip_x": 0.4,
    "left.hip_y": -2.0,
    "left.knee": 2.356,
    "left.ankle_y": -1.053,
    "left.ankle_x": -0.2,
    "left.shoulder_j1": -1.63,
    "left.shoulder_j2": -0.12,
    "left.upper_arm_twist": -0.185,
    "left.elbow": -1.1,
    "left.wrist_roll": 0.0,
    "left.wrist_pitch": 0.0,
    "left.wrist_yaw": -0.5,

    # Right side
    "right.hip_z": 0.15,
    "right.hip_x": -0.4,
    "right.hip_y": -2.0,
    "right.knee": 2.356,
    "right.ankle_y": -1.053,
    "right.ankle_x": 0.2,
    "right.shoulder_j1": 1.63,
    "right.shoulder_j2": 0.12,
    "right.upper_arm_twist": -0.185,
    "right.elbow": 1.1,
    "right.wrist_roll": 0.0,
    "right.wrist_pitch": 0.0,
    "right.wrist_yaw": 0.5,

    # Spine and neck
    "spine_z": 0.0,
    "spine_x": 0.0,
    "neck_no": 0.0,
    "neck_yes": 0.0,
}



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
        assert self.num_actions == len(KP) == len(KD), "Number of actions must match number of joints (for now)"

        self.simulate_action_latency = True  # there is a 1 step latency on real robot
        self.dt = 0.02  # control frequence on real robot is 50hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(n_rendered_envs=1),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
                gravity=(0.0, 0.0, -9.81),
            ),
            show_viewer=show_viewer,
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

        self.dofs_idx = [self.robot.get_joint(name).dof_idx_local for name in joint_names]

        # build
        self.scene.build(n_envs=self.num_envs)

        # names to indices
        self.motor_dofs = [self.robot.get_joint(name).dof_idx_local for name in joint_names]

        # PD control parameters
        # self.robot.set_dofs_kp([self.env_cfg["kp"]] * self.num_actions, self.motor_dofs)
        # self.robot.set_dofs_kv([self.env_cfg["kd"]] * self.num_actions, self.motor_dofs)

        # set positional gains
        self.robot.set_dofs_kp(
            kp             = np.array([KP[joint_name] for joint_name in joint_names]),
            dofs_idx_local = self.dofs_idx,
        )
        # set velocity gains
        self.robot.set_dofs_kv(
            kv             = np.array([KD[joint_name] for joint_name in joint_names]),
            dofs_idx_local = self.dofs_idx,
        )

        # set force range for safety
        self.robot.set_dofs_force_range(
            lower          = np.array([torque_lb[joint_name] for joint_name in joint_names]),
            upper          = np.array([torque_ub[joint_name] for joint_name in joint_names]),
            dofs_idx_local = self.dofs_idx,
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
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.default_dof_pos = torch.tensor(
            [crawl_pose_elbows_semi_flexed[name] for name in joint_names],
            device=self.device,
            dtype=gs.tc_float,
        )
        self.terminal_dof_pos = torch.tensor(
            [downward_facing_dog[name] for name in joint_names],
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
        
        # Learn deviations from the initial body pose
        # target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos

        # Learn delta actions
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.dof_pos

        self.robot.control_dofs_position(target_dof_pos, self.motor_dofs)
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

        # resample commands
        envs_idx = (
            (self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)
            .nonzero(as_tuple=False)
            .flatten()
        )

        # check termination and reset
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        # self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        # self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]

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
                self.base_euler * self.obs_scales["base_euler"],  # 4
                self.projected_gravity * self.obs_scales["projected_gravity"],  # 3
                self.dof_pos * self.obs_scales["dof_pos"],  # 30
                self.actions * self.obs_scales["actions"],  # 30
            ],
            axis=-1,
        )

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

        # Collect observation as info
        self.extras["base_ang_vel"] = self.base_ang_vel
        self.extras["base_lin_vel"] = self.base_lin_vel
        self.extras["base_euler"] = self.base_euler
        self.extras["projected_gravity"] = self.projected_gravity
        self.extras["dof_pos"] = self.dof_pos
        self.extras["dof_vel"] = self.dof_vel
        self.extras["actions"] = self.actions
        self.extras["last_actions"] = self.last_actions
        self.extras["episode_length_buf"] = self.episode_length_buf
        self.extras["reset"] = self.reset_buf
        self.extras["episode_sums"] = self.episode_sums
        
        if self.writer is not None:
            add2tensorboard(self.writer, self.extras, self.global_counter)
            self.global_counter += 1

        if torch.any(self.reset_buf):
            print("self.episode_length_buf: ", self.episode_length_buf)

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

    def _reward_action_rate(self):
        # Penalize changes in actions
        # import pdb; pdb.set_trace()
        return -torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_base_pitch_yaw_tilt(self):
        # Penalize base tilting: Assuming that the torso will be mostly rotated,
        # we penalize pitch and yaw tilting in Base frame coordinates.
        base_pitch_tilt_error = torch.sum(torch.square(self.projected_gravity[:,1::]), dim=1)
        return torch.exp(-base_pitch_tilt_error / self.reward_cfg["tracking_sigma"])

    def _reward_com_position_rt_base(self):
        # Penalize COM position such that it gets closer and closer to be between the feet
        return torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

    def _reward_com_position_rt_base_terminal(self):
        # Terminal cost
        return torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

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