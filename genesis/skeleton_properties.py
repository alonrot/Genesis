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