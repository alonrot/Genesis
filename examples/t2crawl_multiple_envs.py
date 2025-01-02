import argparse
import torch
import numpy as np
import genesis as gs
from time import time

import pdb
from genesis.skeleton_properties import KP, KD, torque_lb, torque_ub

from genesis.pose_library import crawl_pose_elbows_semi_flexed, t_pose_ground_random, t_pose, t_pose_ground, t_pose_arms_up, ready_to_push, push_up_halfway, push_up, to_crawl, downward_facing_dog, joint_names_fingers, closed_fingers_pos

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(backend=gs.cpu, precision="32")

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


    # create 4 parallel environments
    B = 4
    scene.build(n_envs=B, env_spacing=(2.5, 2.5))

    # for link in plane.links:
    #     print("link: ", link)

    # print("Nlinks: ", len(plane.links))

    # plane.set_friction_ratio(
    #     friction_ratio=0.5 + torch.rand(scene.n_envs, plane.n_links),
    #     link_indices=np.arange(0, plane.n_links),
    # )


    # plane.set_friction_ratio(
    #     friction_ratio=torch.tensor([[5.0],[5.0],[5.0],[5.0]]),
    #     link_indices=np.arange(0, plane.n_links),
    # )

    # robot.set_friction_ratio(
    #     friction_ratio=1.0 + 0.2*torch.rand(scene.n_envs, robot.n_links),
    #     link_indices=np.arange(0, robot.n_links),
    # )

    for link in plane.links:
        print(f"plane Link {link.name}")
        for geom in link.geoms:
            geom._friction = 0.1
            print("plane geom.friction", geom.friction)
            print("plane geom.coup_friction", geom.coup_friction)


    # for link in robot.links:
    #     print(f"ROBOT Link {link.name}")
    #     for geom in link.geoms:
    #         geom._friction = 0.1
    #         print("ROBOT geom.friction", geom.friction)
    #         print("ROBOT geom.coup_friction", geom.coup_friction)


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


    # robot.set_mass_shift(
    #     mass_shift=-0.5 + torch.rand(scene.n_envs, robot.n_links),
    #     link_indices=np.arange(0, robot.n_links),
    # )
    # robot.set_COM_shift(
    #     com_shift=-0.05 + 0.1 * torch.rand(scene.n_envs, robot.n_links, 3),
    #     link_indices=np.arange(0, robot.n_links),
    # )

    gs.tools.run_in_another_thread(fn=run_sim, args=(scene, robot, dofs_idx, joint_names, B, args.vis))
    if args.vis:
        scene.viewer.start()

def simple_ramp_from_current_to_desired_pose(current_pose, desired_pose, alpha):
    return (1 - alpha) * current_pose + alpha * desired_pose

def run_sim(scene, robot, dofs_idx, joint_names, B, enable_vis):

    t_total = 60.0
    t_start = time()
    t_elapsed = 0.0
    t_loop_start = 0.0
    t_increment = 3.0
    t_hold_start = 10.0
    
    # Set desired to measured
    joint_positions_measured = robot.get_dofs_position(dofs_idx)
    joint_positions_desired = joint_positions_measured.clone().detach().numpy()
    # NOTE: Because we've set B = 4 at scene build, the measured is not a 1-D vector, but a [4 x nDofs] matrix
    # In this example, we set the desired to be equal to one of the measured to simplify the ramp() and other
    # computations below. Doing this properly involves exiding the functions below to handle bacthes of desired
    # poses.
    joint_positions_desired = joint_positions_desired[0]

    alpha = 0.01

    fingers_values = torch.zeros((B, len(closed_fingers_pos)))
    fingers_values[:] = torch.tensor([list(closed_fingers_pos.values())])

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
            
            # robot.set_friction_ratio(
            #     friction_ratio=4.5 + 0.2*torch.rand(scene.n_envs, robot.n_links),
            #     link_indices=np.arange(0, robot.n_links),
            # )

        elif t_elapsed < t_hold_start + 4.*t_increment:
            joint_positions_target = np.array([push_up[joint_name] for joint_name in joint_names])
        elif t_elapsed < t_hold_start + 5.*t_increment:
            joint_positions_target = np.array([to_crawl[joint_name] for joint_name in joint_names])
        else:
            joint_positions_target = np.array([crawl_pose_elbows_semi_flexed[joint_name] for joint_name in joint_names])

        joint_positions_desired = simple_ramp_from_current_to_desired_pose(joint_positions_desired, joint_positions_target, alpha)

        joint_positions_desired_batched = torch.tile(
            torch.tensor(joint_positions_desired, device=gs.device), (B, 1)
        )

        # print("joint_positions_desired_batched.shape: ", joint_positions_desired_batched.shape)

        # To close the fingers, we need to expand self.motor_dofs to incldue them - otherwise, we can't control them
        joint_names_with_fingers = list(joint_names) + list(closed_fingers_pos.keys())
        motor_dofs_with_fingers = [robot.get_joint(name).dof_idx_local for name in joint_names_with_fingers]

        target_joints_pos_to_send_plus_closed_fingers = torch.cat((joint_positions_desired_batched, fingers_values), axis=1)
        robot.control_dofs_position(target_joints_pos_to_send_plus_closed_fingers, motor_dofs_with_fingers)


        # robot.control_dofs_position(joint_positions_desired_batched, dofs_idx_local=dofs_idx)

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
    python examples/t2crawl_multiple_envs.py --vis
    """
