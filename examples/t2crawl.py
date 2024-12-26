import argparse
import torch
import numpy as np
import genesis as gs
from time import time

import pdb
from skeleton_properties import KP, KD, torque_lb, torque_ub

from pose_library import crawl_pose_elbows_semi_flexed, t_pose_ground_random, t_pose, t_pose_ground, t_pose_arms_up, ready_to_push, push_up_halfway, push_up, to_crawl, downward_facing_dog

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(backend=gs.cpu)

    ########################## create a scene ##########################

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.5, 0.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
        ),
        show_viewer=args.vis,
        rigid_options=gs.options.RigidOptions(
            dt=0.01,
            gravity=(0.0, 0.0, -9.81),
        ),
    )

    mjcf = "/Users/alonrot/work_figure_ai/ws/project-x/robot_config/sim/b_sample/b_sample.xml"

    ########################## entities ##########################
    plane = scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(
        gs.morphs.MJCF(file=mjcf,
        pos   = (0, 0, 0.3),
        euler = (0, 90, 0), # we follow scipy's extrinsic x-y-z rotation convention, in degrees,
        # quat  = (1.0, 0.0, 0.0, 0.0), # we use w-x-y-z convention for quaternions,
        scale = 1.0),
    )

    assert len(KP) == len(KD)
    joint_names = KP.keys()
    assert all(name in KD.keys() for name in joint_names)

    dofs_idx = [robot.get_joint(name).dof_idx_local for name in joint_names]

    ########################## build ##########################
    scene.build()

    ############ Optional: set control gains ############
    # set positional gains
    robot.set_dofs_kp(
        kp             = np.array([KP[joint_name] for joint_name in joint_names]),
        dofs_idx_local = dofs_idx,
    )
    # set velocity gains
    robot.set_dofs_kv(
        kv             = np.array([KD[joint_name] for joint_name in joint_names]),
        dofs_idx_local = dofs_idx,
    )
    # set force range for safety
    robot.set_dofs_force_range(
        lower          = np.array([torque_lb[joint_name] for joint_name in joint_names]),
        upper          = np.array([torque_ub[joint_name] for joint_name in joint_names]),
        dofs_idx_local = dofs_idx,
    )

    gs.tools.run_in_another_thread(fn=run_sim, args=(scene, robot, dofs_idx, joint_names, args.vis))
    if args.vis:
        scene.viewer.start()

def simple_ramp_from_current_to_desired_pose(current_pose, desired_pose, alpha):
    return (1 - alpha) * current_pose + alpha * desired_pose

def run_sim(scene, robot, dofs_idx, joint_names, enable_vis):

    # # From go2_env.py
    # self.motor_dofs = [self.robot.get_joint(name).dof_idx_local for name in self.env_cfg["dof_names"]]
    # self.base_pos[:] = self.robot.get_pos()
    # self.base_quat[:] = self.robot.get_quat()
    # self.base_euler = quat_to_xyz(
    #     transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat)
    # )
    # inv_base_quat = inv_quat(self.base_quat)
    # self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat)
    # self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
    # self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
    # self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)
    # self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)

    t_total = 60.0
    t_start = time()
    t_elapsed = 0.0
    t_loop_start = 0.0
    t_increment = 3.0
    t_hold_start = 10.0
    
    # Set desired to measured
    joint_positions_desired = robot.get_dofs_position(dofs_idx).clone().detach().numpy()

    alpha = 0.01

    while t_elapsed < t_total:

        joint_positions_target = np.array([t_pose_arms_up[joint_name] for joint_name in joint_names])

        # Hard reset
        if t_elapsed < t_hold_start:
            joint_positions_target = np.array([t_pose_ground[joint_name] for joint_name in joint_names])
        elif t_elapsed < t_hold_start + t_increment:
            joint_positions_target = np.array([t_pose_arms_up[joint_name] for joint_name in joint_names])
        elif t_elapsed < t_hold_start + 2.*t_increment:
            joint_positions_target = np.array([ready_to_push[joint_name] for joint_name in joint_names])
        elif t_elapsed < t_hold_start + 3.*t_increment:
            joint_positions_target = np.array([push_up_halfway[joint_name] for joint_name in joint_names])
        elif t_elapsed < t_hold_start + 4.*t_increment:
            joint_positions_target = np.array([push_up[joint_name] for joint_name in joint_names])
        elif t_elapsed < t_hold_start + 5.*t_increment:
            joint_positions_target = np.array([to_crawl[joint_name] for joint_name in joint_names])
        else:
            joint_positions_target = np.array([crawl_pose_elbows_semi_flexed[joint_name] for joint_name in joint_names])

        joint_positions_desired = simple_ramp_from_current_to_desired_pose(joint_positions_desired, joint_positions_target, alpha)

        robot.control_dofs_position(joint_positions_desired, dofs_idx)

        scene.step()

        # Compute loop frequency
        t_loop_end = time()
        t_elapsed = t_loop_end - t_start
        print(1 / (t_loop_end - t_loop_start), "FPS")

        print("t_elapsed: ", t_elapsed)

        # Reset
        t_loop_start = t_loop_end

    if enable_vis:
        scene.viewer.stop()


if __name__ == "__main__":
    main()

    """
    run as python examples/t2crawl.py --vis
    """
