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


# joint_names = [
#     "left.hip_z",
#     "left.hip_x",
#     "left.hip_y",
#     "left.knee",
#     "left.ankle_y",
#     "left.ankle_x",
#     "left.shoulder_j1",
#     "left.shoulder_j2",
#     "left.upper_arm_twist",
#     "left.elbow",
#     "left.wrist_roll",
#     "left.wrist_pitch",
#     "left.wrist_yaw",
#     "right.hip_z",
#     "right.hip_x",
#     "right.hip_y",
#     "right.knee",
#     "right.ankle_y",
#     "right.ankle_x",
#     "right.shoulder_j1",
#     "right.shoulder_j2",
#     "right.upper_arm_twist",
#     "right.elbow",
#     "right.wrist_roll",
#     "right.wrist_pitch",
#     "right.wrist_yaw",
#     "spine_x",
#     "spine_z",
#     "neck_no",
#     "nexk_yes",
# ]

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


class FigureEnv:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False, device="cpu"):
        self.device = torch.device(device)

        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        assert self.num_actions == len(KP) == len(KD), "Number of actions must match number of joints (for now)"
        self.num_commands = command_cfg["num_commands"]

        self.simulate_action_latency = True  # there is a 1 step latency on real robot
        self.dt = 0.02  # control frequence on real robot is 50hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

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
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

        # initialize buffers
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=gs.tc_float).repeat(
            self.num_envs, 1
        )
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=self.device, dtype=gs.tc_float)
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]],
            device=self.device,
            dtype=gs.tc_float,
        )
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
        self.extras = dict()  # extra information for logging

    def _resample_commands(self, envs_idx):
        self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["lin_vel_x_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["lin_vel_y_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["ang_vel_range"], (len(envs_idx),), self.device)

    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        self.robot.control_dofs_position(target_dof_pos, self.motor_dofs)
        self.scene.step()

        # update buffers
        self.episode_length_buf += 1
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        self.base_euler = quat_to_xyz(
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
        self._resample_commands(envs_idx)

        # check termination and reset
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
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
                self.base_ang_vel * self.obs_scales["ang_vel"],  # 3
                self.projected_gravity,  # 3
                self.commands * self.commands_scale,  # 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12
                self.dof_vel * self.obs_scales["dof_vel"],  # 12
                self.actions,  # 12
            ],
            axis=-1,
        )

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

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

        self._resample_commands(envs_idx)

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.obs_buf, None

    # ------------ reward functions----------------
    def _reward_zero_lateral_base_vel(self):
        # Penalize lateral base velocity
        lin_vel_error = torch.sum(torch.square(self.base_lin_vel[:, 1]), dim=1)
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_zero_base_yaw_twist(self):
        # Penalize yaw twist component
        ang_vel_error = torch.sum(self.base_ang_vel[:, 2]**2)
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_base_pitch_yaw_tilt(self):
        # Penalize base tilting: Assuming that the torso will be mostly rotated,
        # we penalize pitch and yaw tilting in Base frame coordinates.
        return torch.sum(torch.square(self.projected_gravity[:,1::]), dim=1)

    def _reward_com_position_rt_base(self):
        # Penalize COM position such that it gets closer and closer to be between the feet
        return 0.0

    def _reward_com_position_rt_base_terminal(self):
        # Terminal cost
        return 0.0

    def _reward_final_body_pose_terminal(self):
        # Terminal body pose
        # assert False, "Untested"
        return torch.sum(torch.square(self.dof_pos - downward_facing_dog[self.dofs_idx]), dim=1)


    # def _reward_similar_to_default(self):
    #     # Penalize joint poses far away from default pose
    #     return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

