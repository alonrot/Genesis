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




t_pose_ground_random = {
    # Left side
    "left.hip_y": -0.40,
    "left.hip_x": 0.91,
    "left.hip_z": 0.42,
    "left.knee": 0.31,
    "left.ankle_y": -0.19,
    "left.ankle_x": 0.30,
    "left.shoulder_j1": -0.42,
    "left.shoulder_j2": 0.45,
    "left.upper_arm_twist": 0.21,
    "left.elbow": -0.39,
    "left.wrist_roll": 0.01,
    "left.wrist_pitch": 0.29,
    "left.wrist_yaw": 0.37,

    # Right side
    "right.shoulder_j1": 0.27,
    "right.shoulder_j2": -0.12,
    "right.upper_arm_twist": 0.56,
    "right.elbow": -0.09,
    "right.wrist_roll": -0.18,
    "right.wrist_pitch": 0.25,
    "right.wrist_yaw": -0.28,
    "right.hip_y": -0.03,
    "right.hip_x": 0.24,
    "right.hip_z": 0.24,
    "right.knee": 0.11,
    "right.ankle_y": 0.22,
    "right.ankle_x": 0.13,

    # Spine and neck
    "spine_z": -0.19,
    "spine_x": -0.12,
    "neck_no": -0.05,
    "neck_yes": -0.22,
}

t_pose = {
    # Left side (modifications from zero_pose)
    "left.shoulder_j1": -0.80,
    "left.shoulder_j2": 1.20,
    "left.upper_arm_twist": 0.77,
    "left.wrist_roll": -1.57,  # -0.5 * pi

    # Right side (modifications from zero_pose)
    "right.shoulder_j1": 0.80,  # Opposite sign due to $SIGN = -1
    "right.shoulder_j2": -1.20,  # Opposite sign due to $SIGN = -1
    "right.upper_arm_twist": 0.77,
    "right.wrist_roll": -1.57,  # -0.5 * pi

    # All other joints (inherit from zero_pose, values are 0)
    "left.hip_z": 0.0,
    "left.hip_x": 0.0,
    "left.hip_y": 0.0,
    "left.knee": 0.0,
    "left.ankle_y": 0.0,
    "left.ankle_x": 0.0,
    "left.elbow": 0.0,
    "left.wrist_pitch": 0.0,
    "left.wrist_yaw": 0.0,

    "right.hip_z": 0.0,
    "right.hip_x": 0.0,
    "right.hip_y": 0.0,
    "right.knee": 0.0,
    "right.ankle_y": 0.0,
    "right.ankle_x": 0.0,
    "right.elbow": 0.0,
    "right.wrist_pitch": 0.0,
    "right.wrist_yaw": 0.0,

    "spine_z": 0.0,
    "spine_x": 0.0,
    "neck_no": 0.0,
    "neck_yes": 0.0,
}

t_pose_ground = {
    "left.shoulder_j1": -0.80,
    "left.shoulder_j2": 1.20,
    "left.upper_arm_twist": 0.77,
    "left.wrist_roll": 0.0,  # Overwritten from t_pose
    "right.shoulder_j1": 0.80,
    "right.shoulder_j2": -1.20,
    "right.upper_arm_twist": 0.77,
    "right.wrist_roll": 0.0,  # Overwritten from t_pose
    "spine_z": 0.0,
    "spine_x": 0.0,
    "neck_no": 0.0,
    "neck_yes": -0.44,  # Overwritten
    "left.hip_z": 0.0,
    "left.hip_x": 0.0,
    "left.hip_y": 0.0,
    "left.knee": 0.0,
    "left.ankle_y": 0.0,
    "left.ankle_x": 0.0,
    "left.elbow": 0.0,
    "left.wrist_pitch": 0.0,
    "left.wrist_yaw": 0.0,
    "right.hip_z": 0.0,
    "right.hip_x": 0.0,
    "right.hip_y": 0.0,
    "right.knee": 0.0,
    "right.ankle_y": 0.0,
    "right.ankle_x": 0.0,
    "right.elbow": 0.0,
    "right.wrist_pitch": 0.0,
    "right.wrist_yaw": 0.0,
}

t_pose_arms_up = {
    # **t_pose_ground,  # Inherit all values
    # "left.shoulder_j1": -0.80,
    # "left.shoulder_j2": 1.20,
    # "left.upper_arm_twist": 0.77,
    "left.wrist_roll": 0.0,  # Overwritten from t_pose
    # "right.shoulder_j1": 0.80,
    # "right.shoulder_j2": -1.20,
    # "right.upper_arm_twist": 0.77,
    "right.wrist_roll": 0.0,  # Overwritten from t_pose
    "spine_z": 0.0,
    "spine_x": 0.0,
    "neck_no": 0.0,
    "neck_yes": -0.44,  # Overwritten
    "left.hip_z": 0.0,
    # "left.hip_x": 0.0,
    "left.hip_y": 0.0,
    # "left.knee": 0.0,
    # "left.ankle_y": 0.0,
    "left.ankle_x": 0.0,
    "left.elbow": 0.0,
    # "left.wrist_pitch": 0.0,
    # "left.wrist_yaw": 0.0,
    "right.hip_z": 0.0,
    # "right.hip_x": 0.0,
    "right.hip_y": 0.0,
    # "right.knee": 0.0,
    # "right.ankle_y": 0.0,
    "right.ankle_x": 0.0,
    "right.elbow": 0.0,
    # "right.wrist_pitch": 0.0,
    # "right.wrist_yaw": 0.0,

    "left.shoulder_j1": 0.77,  # Overwritten
    "left.shoulder_j2": 0.2,  # Overwritten
    "left.upper_arm_twist": 0.16,  # Overwritten
    "left.wrist_pitch": -0.2,  # Overwritten
    "left.wrist_yaw": -1.57,  # Overwritten
    "left.knee": 0.5,  # Overwritten
    "left.ankle_y": -1.0,  # Overwritten
    "left.hip_x": 0.4,  # Overwritten
    "right.shoulder_j1": -0.77,  # Overwritten
    "right.shoulder_j2": -0.2,  # Overwritten
    "right.upper_arm_twist": 0.16,  # Overwritten
    "right.wrist_pitch": 0.2,  # Overwritten
    "right.wrist_yaw": 1.57,  # Overwritten
    "right.knee": 0.5,  # Overwritten
    "right.ankle_y": -1.0,  # Overwritten
    "right.hip_x": -0.4,  # Overwritten
}


ready_to_push = {
    **t_pose_arms_up,  # Inherit all values
    "left.shoulder_j1": 0.7,  # Overwritten
    "left.shoulder_j2": 0.2,  # Retained
    "left.upper_arm_twist": 0.16,  # Retained
    "left.elbow": -2.36,  # Overwritten
    "left.wrist_roll": -0.7,  # Overwritten
    "left.wrist_pitch": -0.2,  # Retained
    "left.wrist_yaw": -1.57,  # Retained
    "right.shoulder_j1": -0.7,  # Overwritten
    "right.shoulder_j2": -0.2,  # Retained
    "right.upper_arm_twist": 0.16,  # Retained
    "right.elbow": 2.36,  # Overwritten
    "right.wrist_roll": -0.7,  # Overwritten
    "right.wrist_pitch": 0.2,  # Retained
    "right.wrist_yaw": 1.57,  # Retained
}


push_up_halfway = {
    **t_pose_arms_up,  # Inherit all values
    "left.shoulder_j1": -0.29,  # Overwritten
    "left.shoulder_j2": -0.18,  # Overwritten
    "left.upper_arm_twist": 0.54,  # Overwritten
    "left.elbow": -1.46,  # Overwritten
    "left.wrist_roll": -1.1,  # Overwritten
    "left.wrist_pitch": 0.16,  # Overwritten
    "left.wrist_yaw": -1.57,  # Retained
    "right.shoulder_j1": 0.29,  # Overwritten
    "right.shoulder_j2": 0.18,  # Overwritten
    "right.upper_arm_twist": 0.54,  # Overwritten
    "right.elbow": 1.46,  # Overwritten
    "right.wrist_roll": -1.1,  # Overwritten
    "right.wrist_pitch": -0.16,  # Overwritten
    "right.wrist_yaw": 1.57,  # Retained
}



push_up = {
    **push_up_halfway,  # Inherit all values
    "left.shoulder_j1": -1.5,  # Overwritten
    "left.shoulder_j2": -0.35,  # Overwritten
    "left.elbow": 0.0,  # Overwritten
    "left.hip_y": -1.0,  # Overwritten
    "right.shoulder_j1": 1.5,  # Overwritten
    "right.shoulder_j2": 0.35,  # Overwritten
    "right.elbow": 0.0,  # Overwritten
    "right.hip_y": -1.0,  # Overwritten
}



to_crawl = {
    **push_up,  # Inherit all values
    "left.shoulder_j1": -1.5,  # Retained
    "left.shoulder_j2": -0.35,  # Retained
    "left.upper_arm_twist": -0.2,  # Overwritten
    "left.elbow": -1.5,  # Overwritten
    "left.wrist_roll": 0.0,  # Overwritten
    "left.wrist_yaw": 0.0,  # Overwritten
    "left.hip_y": -2.0,  # Overwritten
    "right.shoulder_j1": 1.5,  # Retained
    "right.shoulder_j2": 0.35,  # Retained
    "right.upper_arm_twist": -0.2,  # Overwritten
    "right.elbow": 1.5,  # Overwritten
    "right.wrist_roll": 0.0,  # Overwritten
    "right.wrist_yaw": 0.0,  # Overwritten
    "right.hip_y": -2.0,  # Overwritten
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

joint_names_fingers = [
    "left.thumb_rotate",
    "left.THJ1",
    "left.FFJ1",
    "left.MFJ1",
    "left.RFJ1",
    "left.LFJ1",
    "right.thumb_rotate",
    "right.THJ1",
    "right.FFJ1",
    "right.MFJ1",
    "right.RFJ1",
    "right.LFJ1",
    "left.THJ2",
    "left.FFJ2",
    "left.MFJ2",
    "left.RFJ2",
    "left.LFJ2",
    "right.THJ2",
    "right.FFJ2",
    "right.MFJ2",
    "right.RFJ2",
    "right.LFJ2",
]