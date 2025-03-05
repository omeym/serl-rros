import numpy as np
from franka_env.envs.franka_env import DefaultEnvConfig


class PegEnvConfig(DefaultEnvConfig):
    """Set the configuration for FrankaEnv."""

    ROBOT_IP: str = "192.168.10.122"
    REALSENSE_CAMERAS = {
        "wrist_1": "932122060300",
        "wrist_2": "023322060732",
    }
    TARGET_POSE = np.array(
        [0.76991, -0.03989, 0.08642, -1.5308283, 0.01884956, -3.11663445]
    )
    # nisara - P2
    RESET_POSE = np.array(
        [
            0.76791,
            -0.03337,
            0.12004,
            np.deg2rad(-92.54),
            np.deg2rad(0.54),
            np.deg2rad(179.84),
        ]
    )
    REWARD_THRESHOLD: np.ndarray = np.array(
        [0.002, 0.002, 0.01, 0.0349066, 0.0349066, 0.0349066]
    )
    APPLY_GRIPPER_PENALTY = False
    ACTION_SCALE = np.array([0.02, 0.1, 1])
    RANDOM_RESET = False
    RANDOM_XY_RANGE = 0.01
    RANDOM_RZ_RANGE = np.pi / 6
    USE_GRIPPER: bool = False
    ABS_POSE_LIMIT_LOW = np.array(
        [
            TARGET_POSE[0] - RANDOM_XY_RANGE,
            TARGET_POSE[1] - RANDOM_XY_RANGE,
            TARGET_POSE[2],
            TARGET_POSE[3] - RANDOM_RZ_RANGE,
            TARGET_POSE[4] - 0.01,
            TARGET_POSE[5] - 0.01,
        ]
    )
    ABS_POSE_LIMIT_HIGH = np.array(
        [
            TARGET_POSE[0] + RANDOM_XY_RANGE,
            TARGET_POSE[1] + RANDOM_XY_RANGE,
            TARGET_POSE[2] + 0.05,
            TARGET_POSE[3] + RANDOM_RZ_RANGE,
            TARGET_POSE[4] + 0.01,
            TARGET_POSE[5] + 0.01,
        ]
    )
    COMPLIANCE_PARAM = {
        "translational_stiffness": 2000,
        "translational_damping": 0.89,
        "rotational_stiffness": 150,
        "rotational_damping": 0.7,
        "translational_Ki": 0,
        "translational_clip_x": 0.003,
        "translational_clip_y": 0.003,
        "translational_clip_z": 0.01,
        "translational_clip_neg_x": 0.003,
        "translational_clip_neg_y": 0.003,
        "translational_clip_neg_z": 0.01,
        "rotational_clip_x": 0.02,
        "rotational_clip_y": 0.02,
        "rotational_clip_z": 0.02,
        "rotational_clip_neg_x": 0.02,
        "rotational_clip_neg_y": 0.02,
        "rotational_clip_neg_z": 0.02,
        "rotational_Ki": 0,
    }
    PRECISION_PARAM = {
        "translational_stiffness": 3000,
        "translational_damping": 0.89,
        "rotational_stiffness": 300,
        "rotational_damping": 0.9,
        "translational_Ki": 0.1,
        "translational_clip_x": 0.01,
        "translational_clip_y": 0.01,
        "translational_clip_z": 0.01,
        "translational_clip_neg_x": 0.01,
        "translational_clip_neg_y": 0.01,
        "translational_clip_neg_z": 0.01,
        "rotational_clip_x": 0.05,
        "rotational_clip_y": 0.05,
        "rotational_clip_z": 0.05,
        "rotational_clip_neg_x": 0.05,
        "rotational_clip_neg_y": 0.05,
        "rotational_clip_neg_z": 0.05,
        "rotational_Ki": 0.1,
    }
