import torch
import math
import numpy as np
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat

from genesis.engine.solvers.rigid.rigid_solver_decomp import RigidSolver

import os
if 'figva' in os.uname()[1]:
    MJCF_PATH = "/shared/home/alonrot/project-x/robot_config/sim/b_sample/b_sample.xml"
else:
    MJCF_PATH = "/Users/alonrot/work_figure_ai/ws/project-x/robot_config/sim/b_sample/b_sample.xml"

from genesis.skeleton_properties import KP,  KD, torque_lb, torque_ub

from genesis.pose_library import crawl_pose_elbows_semi_flexed, t_pose_ground_random, t_pose, t_pose_ground, t_pose_arms_up, ready_to_push, push_up_halfway, push_up, to_crawl, downward_facing_dog, closed_fingers_pos
joint_names = KP.keys()

from genesis.figure_cfg import learnable_joints

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

        self.close_fingers = True
        self.randomize_friction = False
        if "randomize_friction" in env_cfg.keys():
            self.randomize_friction = env_cfg["randomize_friction"]
            
        self.ground_constant_friction = False
        if "ground_constant_friction" in env_cfg.keys():
            self.ground_constant_friction = env_cfg["ground_constant_friction"]

        assert (self.ground_constant_friction and self.randomize_friction) == False, "ground_constant_friction and randomize_friction are mutually exclusive"

        self.push_pose_stay = True
        if self.randomize_friction:
            self.push_pose_stay = True

        # If False, we add delta actions to the default pose
        self.control_based_on_current_pos = False

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
            show_FPS=False,
        )

        # add plain
        self.ground = self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
        
        # self.ground = self.scene.add_entity(morph=gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True), visualize_contact=True)
        # self.ground.set_friction_ratio(friction_ratio=0.5 + torch.rand(self.num_envs, 1), link_indices=[0])

        if self.ground_constant_friction:
            for link in self.ground.links:
                print(f"self.ground Link {link.name}")
                for geom in link.geoms:
                    geom._friction = 0.1
                    print("self.ground geom.friction", geom.friction)
                    print("self.ground geom.coup_friction", geom.coup_friction)


        # add robot

        if self.randomize_friction:
            self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos_randomize_friction"], device=self.device)
        else:
            self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=self.device)

        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=self.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(file=MJCF_PATH,
            # pos   = (0, 0, 0.3),
            pos   = tuple(self.env_cfg["base_init_pos"]),
            euler = (0, 90, 0), # we follow scipy's extrinsic x-y-z rotation convention, in degrees,
            # quat  = (1.0, 0.0, 0.0, 0.0), # we use w-x-y-z convention for quaternions,
            scale = 1.0),
        )

        assert len(KP) == len(KD)
        assert all(name in KD.keys() for name in joint_names)

        # build
        if self.show_viewer:
            self.scene.build(n_envs=self.num_envs, env_spacing=(2.0, 2.0))
        else:
            self.scene.build(n_envs=self.num_envs)

        # print("Get rigid solver")
        # self.rigid_solver = None
        # for solver in self.scene.sim.solvers:
        #     if not isinstance(solver, RigidSolver):
        #         continue
        #     self.rigid_solver = solver

        # print("len(self.scene.sim.solvers): ", len(self.scene.sim.solvers))
        # assert self.rigid_solver is not None, "Collecting rigid_solver to access link positions rt world"

        # left_hand_pos_rt_world = self.rigid_solver.get_links_pos(idx_left_hand)
        # left_hand_pos_rt_world = self.rigid_solver.get_links_pos(idx_left_hand)

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

        if self.randomize_friction:
            self.ground.set_friction_ratio(
                friction_ratio=0.5 + torch.rand(self.scene.n_envs, self.ground.n_links),
                link_indices=np.arange(0, self.ground.n_links),
            )

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt # same as scaling the average reward by the total time (i.e., longer episodes, higher reward), while the average reward is the sum of instantaneous rewards divided by the number of time steps
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

        print("self.reward_scales: ", self.reward_scales)

        # contact_info = self.robot.get_contacts(with_entity=self.ground)

        # print("CONTACTS")
        # print("CONTACTS")
        # print("CONTACTS")
        # print("CONTACTS")
        # print("CONTACTS")
        # contact_info = self.robot.get_contacts(with_entity=self.ground)

        # for key, val in contact_info.items():
        #     print("key: ", key)
        #     print("val:", val)


        # for links_env_idx_a, links_env_idx_b, valid_mask_in_env in zip(contact_info["link_a"], contact_info["link_b"], contact_info["valid_mask"]):
        #     print("links_env_idx_a: ", links_env_idx_a)
        #     print("links_env_idx_b: ", links_env_idx_b)

        #     links_all = self.scene.rigid_solver.links
        #     for link_a_idx, link_b_idx, valid in zip(links_env_idx_a, links_env_idx_b, valid_mask_in_env):
        #         link_a = links_all[link_a_idx]
        #         link_b = links_all[link_b_idx]

        #         print(f"Contact <{link_a.name}, {link_b.name}> | Valid: {valid}")

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
        self.dof_pos_controlled = torch.zeros((self.num_envs, len(self.idx_controlled_joints)), device=self.device, dtype=gs.tc_float)
        self.target_joints_pos_send = torch.zeros((self.num_envs, len(self.motor_dofs)), device=self.device, dtype=gs.tc_float)
        self.dof_vel = torch.zeros((self.num_envs, len(self.motor_dofs)), device=self.device, dtype=gs.tc_float)
        self.last_dof_vel = torch.zeros((self.num_envs, len(self.motor_dofs)), device=self.device, dtype=gs.tc_float)
        self.base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        
        if self.push_pose_stay:
            self.default_dof_pos = torch.tensor(
                [push_up_halfway[name] for name in joint_names],
                device=self.device,
                dtype=gs.tc_float,
            )
            self.terminal_dof_pos = torch.tensor(
                [push_up_halfway[name] for name in joint_names],
                device=self.device,
                dtype=gs.tc_float,
            )
        else:
            self.default_dof_pos = torch.tensor(
                [t_pose_ground[name] for name in joint_names],
                device=self.device,
                dtype=gs.tc_float,
            )
            self.terminal_dof_pos = torch.tensor(
                [push_up_halfway[name] for name in joint_names],
                device=self.device,
                dtype=gs.tc_float,
            )

        self.left_hand_position = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.right_hand_position = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.left_hand_orientation = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.right_hand_orientation = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        
        self.left_hand_linear_velocity = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.right_hand_linear_velocity = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.left_hand_angular_velocity = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.right_hand_angular_velocity = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        
        self.right_hand_in_contact = torch.zeros((self.num_envs, 1), device=self.device, dtype=gs.tc_int)
        self.left_hand_in_contact = torch.zeros((self.num_envs, 1), device=self.device, dtype=gs.tc_int)

        self.extras = dict()  # extra information for logging
        
        # Tensorboard writer
        self.writer = writer
        self.global_counter = 0

        if self.close_fingers:
            self.joint_names_with_fingers = list(joint_names) + list(closed_fingers_pos.keys())
            self.motor_dofs_with_fingers = [self.robot.get_joint(name).dof_idx_local for name in self.joint_names_with_fingers]
            self.fingers_values = torch.zeros((self.num_envs, len(closed_fingers_pos)), device=self.device, dtype=gs.tc_float)
            self.fingers_values[:] = torch.tensor([list(closed_fingers_pos.values())])

    def hand_in_contact(self, which_hand: str, verbo: bool = False) -> int:
        assert which_hand in ["left", "right"]
        
        arm_links_names = ["lower_forearm", "wrist", "palm"]
        arm_links_names += ["distal_2", "distal_3", "distal_4", "distal_5"]
        arm_links_names += ["medial_2", "medial_3", "medial_4", "medial_5"]
        arm_links_names_hand = []
        for arm_link in arm_links_names:
            arm_links_names_hand += [f"{which_hand}.{arm_link}"]

        if verbo:
            print("arm_links_names_hand: ", arm_links_names_hand)

        contact_info = self.robot.get_contacts(with_entity=self.ground)

        is_hand_in_contact = torch.zeros((self.num_envs, 1), device=self.device, dtype=gs.tc_int)
        cc = 0
        links_all = self.scene.rigid_solver.links
        for links_env_idx_a, links_env_idx_b, valid_mask_in_env in zip(contact_info["link_a"], contact_info["link_b"], contact_info["valid_mask"]):
            # if verbo:
            #     print("links_env_idx_a: ", links_env_idx_a)
            #     print("links_env_idx_b: ", links_env_idx_b)

            is_hand_in_contact[cc,0] = 0

            for link_a_idx, link_b_idx, valid in zip(links_env_idx_a, links_env_idx_b, valid_mask_in_env):

                if verbo and valid:
                    link_a = links_all[link_a_idx]
                    link_b = links_all[link_b_idx]
                    print(f"Contact <{link_a.name}, {link_b.name}> | Valid: {valid}")

                # Detect contact with the ground (ground corresponds to link idx 0)
                if valid and link_a_idx == 0 and link_b_idx == 0:
                    continue
                
                if valid and link_a_idx == 0 and links_all[link_b_idx].name in arm_links_names_hand:
                    is_hand_in_contact[cc,0] = 1

                if valid and link_b_idx == 0 and links_all[link_a_idx].name in arm_links_names_hand:
                    is_hand_in_contact[cc,0] = 1

            cc += 1

        return is_hand_in_contact


    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions

        # Print max, min, mean and std of actions
        # print("actions: ", actions)
        # print("max: ", torch.max(actions))
        # print("min: ", torch.min(actions))
        # print("mean: ", torch.mean(actions))
        # print("std: ", torch.std(actions))

        # Hand pose
        self.left_hand_position[:,:] = self.robot.get_link_pos("left.hand")
        self.right_hand_position[:,:] = self.robot.get_link_pos("right.hand")
        self.left_hand_orientation[:,:] = self.robot.get_link_quat("left.hand")
        self.right_hand_orientation[:,:] = self.robot.get_link_quat("right.hand")

        # Hand twist
        self.left_hand_linear_velocity[:,:] = self.robot.get_link_vel("left.hand")
        self.right_hand_linear_velocity[:,:] = self.robot.get_link_vel("right.hand")
        self.left_hand_angular_velocity[:,:] = self.robot.get_link_ang("left.hand")
        self.right_hand_angular_velocity[:,:] = self.robot.get_link_ang("right.hand")
        
        # idx_link_left_hand = self.robot.get_link("left.hand").idx
        # print("idx_link_left_hand:", idx_link_left_hand)
        # self.collision_pairs = self.robot.detect_collision(env_idx=0)
        # print("self.collision_pairs:", self.collision_pairs)
        # for pair in self.collision_pairs:
        #     if idx_link_left_hand == pair[1]:
        #         print("Left hand in contact")

        self.right_hand_in_contact[:,:] = self.hand_in_contact(which_hand="right", verbo=False)
        # print("self.right_hand_in_contact: ", self.right_hand_in_contact)
        self.left_hand_in_contact[:,:] = self.hand_in_contact(which_hand="left", verbo=False)
        # print("self.left_hand_in_contact: ", self.left_hand_in_contact)

        # print("CONTACTS")
        # print("CONTACTS")
        # print("CONTACTS")
        # print("CONTACTS")
        # print("CONTACTS")
        # contact_info = self.robot.get_contacts(with_entity=self.ground)

        # for key, val in contact_info.items():
        #     print("key: ", key)
        #     print("val:", val)


        # for links_env_idx_a, links_env_idx_b, valid_mask_in_env in zip(contact_info["link_a"], contact_info["link_b"], contact_info["valid_mask"]):
        #     print("links_env_idx_a: ", links_env_idx_a)
        #     print("links_env_idx_b: ", links_env_idx_b)
        #     links_env_a = self.scene.rigid_solver.links[list(links_env_idx_a)]
        #     links_env_b = self.scene.rigid_solver.links[list(links_env_idx_b)]

        #     for link_a, link_b in zip(links_env_a, links_env_b):
        #         print(f"Contact <{link_a.name}, {link_b.name}> | Valid: {valid_mask_in_env}")


        # print("CONTACTS")
        # print("CONTACTS")
        # print("CONTACTS")
        # print("CONTACTS")
        # print("CONTACTS")
        # contact_info = self.robot.get_contacts(with_entity=self.ground)

        # for links_env_idx_a, links_env_idx_b, valid_mask_in_env in zip(contact_info["link_a"], contact_info["link_b"], contact_info["valid_mask"]):
        #     print("links_env_idx_a: ", links_env_idx_a)
        #     print("links_env_idx_b: ", links_env_idx_b)

        #     links_all = self.scene.rigid_solver.links
        #     for link_a_idx, link_b_idx, valid in zip(links_env_idx_a, links_env_idx_b, valid_mask_in_env):

        #         # Detect contact with the ground (ground corresponds to link idx 0)
        #         if valid and (link_a_idx == 0 or link_b_idx == 0):

        #             link_a = links_all[link_a_idx]
        #             link_b = links_all[link_b_idx]
        #             print(f"Contact <{link_a.name}, {link_b.name}> | Valid: {valid}")

        # Learn deviations from the initial body pose
        # target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos

        # # Learn delta actions
        # target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.dof_pos

        # Learn delta actions for controlled joints only

        # exec_actions[...] = 0.0
        
        if self.control_based_on_current_pos:

            # target_controlled_joints_pos_current_plus_actions = exec_actions * self.env_cfg["action_scale"] + self.robot.get_dofs_position(self.idx_controlled_joints)
            target_controlled_joints_pos_current_plus_actions = exec_actions * self.env_cfg["action_scale"] + self.robot.get_dofs_position(self.idx_controlled_joints)

            self.target_joints_pos_send = self.robot.get_dofs_position(self.motor_dofs)
            # self.target_joints_pos_send[:,:] = self.default_dof_pos
            # print("target_controlled_joints_pos_current_plus_actions.shape: ", target_controlled_joints_pos_current_plus_actions.shape)
            # print("self.target_joints_pos_send.shape: ", self.target_joints_pos_send.shape)
            
            # Add target_controlled_joints_pos_current_plus_actions to self.target_joints_pos_send in the right indices
            # print("self.idx_controlled_joints: ", self.idx_controlled_joints)
            # print("self.motor_dofs: ", self.motor_dofs)
            for idx in self.idx_controlled_joints:
                idx_local = self.motor_dofs.index(idx)
                self.target_joints_pos_send[:,idx_local] = target_controlled_joints_pos_current_plus_actions[:,self.idx_controlled_joints.index(idx)]

            # Ensure self.target_joints_pos_send is not pointing to self.default_dof_pos
            assert not torch.all(self.target_joints_pos_send == self.default_dof_pos), "After modfying self.target_joints_pos_send, self.default_dof_pos should stay the same"

        else:
            self.target_joints_pos_send[:,:] = self.default_dof_pos
            delta_actions_on_controlled_joints = exec_actions * self.env_cfg["action_scale"]
            for idx in self.idx_controlled_joints:
                idx_local = self.motor_dofs.index(idx)
                self.target_joints_pos_send[:,idx_local] += delta_actions_on_controlled_joints[:,self.idx_controlled_joints.index(idx)]


        # print("self.target_joints_pos_send: ", self.target_joints_pos_send)
        # print("self.target_joints_pos_send.shape: ", self.target_joints_pos_send.shape)

        assert not torch.any(torch.isnan(self.target_joints_pos_send)) and not torch.any(torch.isinf(self.target_joints_pos_send)), "self.target_joints_pos_send contains NaNs or Infs"

        # # To close the fingers, we need to expand self.motor_dofs to incldue them - otherwise, we can't control them
        # self.joint_names_with_fingers = list(joint_names) + joint_names_fingers
        # self.motor_dofs_with_fingers = [self.robot.get_joint(name).dof_idx_local for name in self.joint_names_with_fingers]
        # target_joints_pos_to_send_plus_closed_fingers = torch.cat((self.target_joints_pos_send, torch.tensor([[1.74533, 2.234025, 2.234025, 2.234025, 2.234025, 2.234025, 1.74533, 2.234025, 2.234025, 2.234025, 2.234025, 2.234025, 1.57, 1.57, 1.57, 1.57, 1.57, 1.57, 1.57, 1.57, 1.57, 1.57]])), axis=1)
        # self.robot.control_dofs_position(target_joints_pos_to_send_plus_closed_fingers, self.motor_dofs_with_fingers)

        if self.close_fingers:
            target_joints_pos_to_send_plus_closed_fingers = torch.cat((self.target_joints_pos_send,self.fingers_values), axis=1)
            self.robot.control_dofs_position(target_joints_pos_to_send_plus_closed_fingers, self.motor_dofs_with_fingers)
        else:
            self.robot.control_dofs_position(self.target_joints_pos_send, self.motor_dofs)

        # # print joint names
        # for joint in self.robot.joints:
        #     print("joint.name: ", joint.name)

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

        # compute reward
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # assert not torch.any(torch.isnan(self.rew_buf)) and not torch.any(torch.isinf(self.rew_buf)), "self.rew_buf contains NaNs or Infs"

        # # compute observations
        # self.obs_buf = torch.cat(
        #     [
        #         self.base_lin_vel * self.obs_scales["lin_vel"],  # 3
        #         self.base_ang_vel * self.obs_scales["ang_vel"],  # 3
        #         self.base_euler * self.obs_scales["base_euler"],  # 3
        #         self.base_pos * self.obs_scales["base_pos"],  # 3
        #         self.projected_gravity * self.obs_scales["projected_gravity"],  # 3
        #         self.dof_pos * self.obs_scales["dof_pos"],  # 30
        #         self.actions * self.obs_scales["actions"],  # 14
        #     ],
        #     axis=-1,
        # )

        self.dof_pos_controlled[:] = self.robot.get_dofs_position(self.idx_controlled_joints)
      
        # compute observations
        self.obs_buf = torch.cat(
            [   self.base_euler * self.obs_scales["base_euler"],  # 3
                self.dof_pos_controlled * self.obs_scales["dof_pos"],  # 14
                self.projected_gravity[:,1:2] * self.obs_scales["projected_gravity"],  # 1
                # self.actions * self.obs_scales["actions"],  # 14
            ],
            axis=-1,
        )

        # check termination and reset
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 2]) > self.env_cfg["termination_if_yaw_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]
        are_envs_with_nans = torch.any(torch.isnan(self.obs_buf), dim=-1) | torch.any(torch.isnan(self.rew_buf), dim=-1)
        self.reset_buf |= are_envs_with_nans

        # print("episode_length_buf: ", self.episode_length_buf)
        # print("self.max_episode_length: ", self.max_episode_length)

        # Count the number of environments in which NaNs were detected
        n_envs_with_nans = torch.sum(are_envs_with_nans).item()

        # Allow NaNs in one environment at a time and reset it
        # assert n_envs_with_nans < 1, f"NaNs detected in {n_envs_with_nans} > 1 environments"
        if n_envs_with_nans > 0:
            print("[WARNING]: NaNs detected in ", n_envs_with_nans, " environments. Resetting them.")
            print("self.reset_buf: ", self.reset_buf)
            
            # Identify which item contains the NaN and Inf values
            if torch.any(torch.isnan(self.base_lin_vel)) or torch.any(torch.isinf(self.base_lin_vel)):
                print("NaN in self.base_lin_vel: ", torch.isnan(self.base_lin_vel).nonzero(as_tuple=False))
                print("Inf in self.base_lin_vel: ", torch.isinf(self.base_lin_vel).nonzero(as_tuple=False))
            if torch.any(torch.isnan(self.base_ang_vel)) or torch.any(torch.isinf(self.base_ang_vel)):
                print("NaN in self.base_ang_vel: ", torch.isnan(self.base_ang_vel).nonzero(as_tuple=False))
                print("Inf in self.base_ang_vel: ", torch.isinf(self.base_ang_vel).nonzero(as_tuple=False))
            if torch.any(torch.isnan(self.base_euler)) or torch.any(torch.isinf(self.base_euler)):
                print("NaN in self.base_euler: ", torch.isnan(self.base_euler).nonzero(as_tuple=False))
                print("Inf in self.base_euler: ", torch.isinf(self.base_euler).nonzero(as_tuple=False))
            if torch.any(torch.isnan(self.base_pos)) or torch.any(torch.isinf(self.base_pos)):
                print("NaN in self.base_pos: ", torch.isnan(self.base_pos).nonzero(as_tuple=False))
                print("Inf in self.base_pos: ", torch.isinf(self.base_pos).nonzero(as_tuple=False))
            if torch.any(torch.isnan(self.projected_gravity)) or torch.any(torch.isinf(self.projected_gravity)):
                print("NaN in self.projected_gravity: ", torch.isnan(self.projected_gravity).nonzero(as_tuple=False))
                print("Inf in self.projected_gravity: ", torch.isinf(self.projected_gravity).nonzero(as_tuple=False))
            if torch.any(torch.isnan(self.dof_pos)) or torch.any(torch.isinf(self.dof_pos)):
                print("NaN in self.dof_pos: ", torch.isnan(self.dof_pos).nonzero(as_tuple=False))
                print("Inf in self.dof_pos: ", torch.isinf(self.dof_pos).nonzero(as_tuple=False))
            if torch.any(torch.isnan(self.actions)) or torch.any(torch.isinf(self.actions)):
                print("NaN in self.actions: ", torch.isnan(self.actions).nonzero(as_tuple=False))
                print("Inf in self.actions: ", torch.isinf(self.actions).nonzero(as_tuple=False))
            if torch.any(torch.isnan(self.obs_buf)) or torch.any(torch.isinf(self.obs_buf)):
                print("NaN in self.obs_buf: ", torch.isnan(self.obs_buf).nonzero(as_tuple=False))
                print("Inf in self.obs_buf: ", torch.isinf(self.obs_buf).nonzero(as_tuple=False))

            # Print min max of actions
            print("actions: ", self.actions)
            print("max: ", torch.max(self.actions))
            print("min: ", torch.min(self.actions))


            for name, reward_func in self.reward_functions.items():
                rew = reward_func() * self.reward_scales[name]
                print("Reward for ", name, ": ", rew)
                print("NaN in ", name, ": ", torch.isnan(rew).nonzero(as_tuple=False))

            assert n_envs_with_nans < int(self.num_envs*0.01), f"NaNs detected in {n_envs_with_nans} of environments ( > 1%)"


        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=self.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        if torch.any(torch.isnan(self.obs_buf)) or torch.any(torch.isinf(self.obs_buf)):
            print("[WARNING]: NaNs or Infs detected in self.obs_buf, even though we reset the environments with NaNs.")

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

    def _reward_action_rate(self):
        # Penalize changes in actions
        # import pdb; pdb.set_trace()
        return -torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
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
    #     self.rew_buf_terminal[:] = 0.0
    #     if torch.any(self.reset_buf):
    #         # final_pos_error = torch.sqrt(torch.mean(torch.square(self.dof_pos[self.reset_buf] - self.terminal_dof_pos), dim=1))
    #         final_pos_error = torch.sqrt(torch.mean(torch.square(self.dof_pos - self.terminal_dof_pos), dim=1))

    #         # print("final_pos_error: ", final_pos_error)
    #         # print("final_pos_error.shape: ", final_pos_error.shape)
    #         # import pdb; pdb.set_trace()

    #         is_near = final_pos_error < self.reward_cfg["terminal_reward_dof_near_threshold"]
    #         # print("reset_and_near:self.reset_buf ", reset_and_near)
    #         # print("reset_and_near.shape: ", reset_and_near.shape)

    #         reset_and_near = torch.logical_and(self.reset_buf, is_near)

    #         self.rew_buf_terminal[reset_and_near] = 1.0

    #     if torch.any(torch.isnan(self.rew_buf_terminal)):
    #         self.rew_buf_terminal[torch.isnan(self.rew_buf_terminal)] = 0.0

    #     #TODO(alonrot): Only apply this reward if the episode is terminated without timeout?
    #     return self.rew_buf_terminal


    
    def _reward_final_body_pose(self):
        final_pos_error = torch.sum(torch.square(self.dof_pos - self.terminal_dof_pos), dim=1)

        # print("final_pos_error: ", final_pos_error)
        # print("final_pos_error.shape: ", final_pos_error.shape)
        # print("final_pos_error max: ", torch.max(final_pos_error))
        # print("final_pos_error min: ", torch.min(final_pos_error))
        # print("final_pos_error mean: ", torch.mean(final_pos_error))
        # print("final_pos_error std: ", torch.std(final_pos_error))

        return -final_pos_error

    def _reward_final_body_pose_exp(self):
        final_pos_error = torch.sum(torch.square(self.dof_pos - self.terminal_dof_pos), dim=1)
        return torch.exp(-final_pos_error / self.reward_cfg["tracking_sigma_final_body_pose"])
    
    def _reward_early_termination_base_yaw_tilt(self):
        base_yaw_tilted = torch.abs(self.base_euler[:, 2]) > self.env_cfg["termination_if_yaw_greater_than"]
        return -base_yaw_tilted.float()
    
    def _reward_early_termination_base_roll_tilt(self):
        base_roll_tilted = torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]
        return -base_roll_tilted.float()

    def _reward_left_hand_slip(self):        
        return -self.left_hand_in_contact[:,0] * torch.sum(torch.square(self.left_hand_linear_velocity[:,0:2]), dim=1)

    def _reward_right_hand_slip(self):
        return -self.right_hand_in_contact[:,0] * torch.sum(torch.square(self.right_hand_linear_velocity[:,0:2]), dim=1)