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

        
        """
        Where we're at:

        - qpos is NOT the joint positions, but something else. It's 77-dimensional. 
        - If we can extract the joint positions from qpos, we can call robot.control_dofs_position(). Otherwise, we need to call robot.set_qpos() with the 77-dimensional qpos which is NOT
        the same as controlling the robot, but simply setting its state (teleporting it without obeying physics).

        IN the documentation I couldn't find anywhere what qpos is. Looking at the definition of set_qpos() as in robot.set_qpos(qpos) didn't help either because
        it's not clear what qpos is.

        We can't use OMPL hwere because it's not MAC compatible. pip install dind't work. brew install dind't work. We may need to copmpile from source BUT even then,
        it looks like the  robot.plan_path() function below doesn't work with free joints, i.e., only works with fixed base entities, like a franka arm

        Woudl be interesting to see if the training of go1 uses any kniematics, or if it's all joint positions.


        The idea for training the crawling behavior is to:
        - Use RL to generate hand poses that we can track using postion control, that will give us the arms
        - LEt RL figure out the "rolling" of the leg joints. All combined should give us the behvaiuor.
        
        Another idea, we can't train using IK, is to just train on joint positions, control them using the PD functions below, 
        and the using forward kinematics to maybe restrict the hand movements to a set of trajectories generated from a 
        prior, like a slow gait for just hands using a central pattern generator. It could be a constraint tot he RL problem, or just a cost penalization
        But be careful because we wanna be coarse with it. RL takes the parmaeters of the entral pattern generator and then figures our arm movements to
        match those gaits, which is like implicitly doing IK.

        I mean, ideally we can use IK, honestly. But the IK of genesis may not work as our IK, so here the assumption is that both IK can generate a motion 
        that tracks the hand positions. 

        Maybe the best approach ehre is just to train on joint positions and try to coarsely restruct the hand movements to roughly follow a prior of generated splines
        We more or less know how the hand movements shoudl look like. 
        """
        
        # left_hand = robot.get_link('left.hand')
        # right_hand = robot.get_link('right.hand')

        # center_left_hand = np.array([0.4, 0.2, 0.25])
        # center_right_hand = np.array([0.4, -0.2, 0.25])

        left_hand = robot.get_link('left.hand')
        right_hand = robot.get_link('right.hand')

        center_left_hand = np.array([0.4, 0.2, 0.5])
        center_right_hand = np.array([0.4, -0.2, 0.5])

        # NOTE: Unless single arm, qpos is usually not the joint positions, but
        # rather
        # qpos = robot.inverse_kinematics(
        #     link = left_hand,
        #     pos  = center_left_hand,
        #     # quat = np.array([0, 1, 0, 0]),
        #     # pos_mask = [False, False, True], # only restrict direction of z-axis
        #     # rot_mask = [False, False, True], # only restrict direction of z-axis
        # )
        # print("qpos: ", qpos)
        # print("len(qpos):", len(qpos)) # 77

        qpos, error_pose = robot.inverse_kinematics_multilink(
            links = [left_hand, right_hand],
            poss  = [center_left_hand, center_right_hand],
            return_error=True
            # quat = np.array([0, 1, 0, 0]),
            # pos_mask = [False, False, True], # only restrict direction of z-axis
            # rot_mask = [False, False, True], # only restrict direction of z-axis
        )

        print("robot.n_qs: ", robot.n_qs)
        print("robot.n_dofs: ", robot.n_dofs)
        print("robot.n_links: ", robot.n_links)
        print("robot.n_joints: ", robot.n_joints)
        print("robot.q_start: ", robot.q_start)
        print("robot.joints: ", robot.joints)

        # print("qpos: ", qpos)
        # print("len(qpos):", len(qpos)) # 77

        # NOTE(alonrot): "Motion planning not suported for rigid entities with free joints" -> found in robot.plan_path()
        # installed OMPL via brew because the wheels aren incompatible (x64 shoudl be but isn't). After installing via brew,
        # it still cannot find OMPL. 
        # pip install ompl -> ERROR: Could not find a version that satisfies the requirement ompl (from versions: none)
        # path = robot.plan_path(
        #     qpos_goal = qpos,
        #     num_waypoints = 200,
        # )

        # # execute the planned path
        # for waypoint in path:
        #     robot.control_dofs_position(waypoint)
        #     scene.step()


        # joint_positions_desired = simple_ramp_from_current_to_desired_pose(joint_positions_desired, joint_positions_target, alpha)

        # DOESNT work because qpos is not the same as joint_positions, qpos has 77 dimensions.
        # robot.control_dofs_position(qpos)
        # scene.step()

        # Moves up the foot instead of the arm, which tells me that qpos contains the information we need, but the indexing is waht we need to figre out. 
        # robot.control_dofs_position(qpos[0:len(dofs_idx)], dofs_idx_local=dofs_idx)
        # scene.step()

        # This is funny: The hand ends up being where it's supposed to be, 
        # but the arms aren't moving. INstead, the entire robot body is moving.
        # Would need to change the regularization here
        # robot.set_qpos(qpos)
        # scene.visualizer.update()

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
    run as python examples/t2crawl_ik.py --vis
    """
