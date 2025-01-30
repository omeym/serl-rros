import time
import sys

env = "serl"
print("Ensure to change anaconda3 or miniconda3 and change your environment name")
print("Conda Environment name is: ", env)
sys.path.append(
    "/home/" + "omey" + "/anaconda3/envs/" + env + "/lib/python3.10/site-packages"
)
import rclpy
import rclpy.duration
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rcl_interfaces.msg import ParameterValue
from rcl_interfaces.srv import GetParameters

import numpy as np
import pandas as pd
import json
import os
from scipy.spatial.transform import Rotation as R

from geometry_msgs.msg import Pose, Point, Quaternion, PoseStamped
import rclpy.task
from sensor_msgs.msg import JointState

from moveit_msgs.msg import (
    RobotState,
    MoveItErrorCodes,
)
from moveit_msgs.srv import GetPositionFK
import optas

import os

MAIN_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


class JointCartesianConverter(Node):

    move_group_name_ = "arm"
    namespace_ = "lbr"

    joint_state_topic_ = "joint_states"
    lbr_state_topic_ = "state"
    pose_topic_ = "pose"
    plan_srv_name_ = "plan_kinematic_path"
    ik_srv_name_ = "compute_ik"
    fk_srv_name_ = "compute_fk"
    execute_action_name_ = "execute_trajectory"
    fri_execute_action_name_ = "joint_trajectory_controller/follow_joint_trajectory"

    base_ = "link_0"
    end_effector_ = "link_ee"

    def __init__(self, joint_data_file_path):
        super().__init__("joint_cartesian_converter")
        self._init_jacobian()

        self.joint_data_file_path = joint_data_file_path
        self.fk_client_callback = MutuallyExclusiveCallbackGroup()
        self.fk_client_ = self.create_client(
            GetPositionFK,
            f"{self.namespace_}/{self.fk_srv_name_}",
            callback_group=self.fk_client_callback,
        )
        if not self.fk_client_.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("FK service not available.")
            exit(1)

        self.joint_data = pd.read_csv(self.joint_data_file_path)

    def _retrieve_parameter(self, service: str, parameter_name: str) -> ParameterValue:
        parameter_client = self.create_client(GetParameters, service)
        while not parameter_client.wait_for_service(timeout_sec=1.0):
            if not rclpy.ok():
                self.get_logger().error(
                    "Interrupted while waiting for the service. Exiting."
                )
                return None
            self.get_logger().info(f"Waiting for '{service}' service...")
        request = GetParameters.Request(names=[parameter_name])
        future = parameter_client.call_async(request=request)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is None:
            self.get_logger().error(f"Failed to retrieve '{parameter_name}'.")
            return None
        return future.result().values[0]

    def _init_jacobian(self):
        print("Initializing Jacobian")

        self._robot_description = self._retrieve_parameter(
            "/lbr/robot_state_publisher/get_parameters", "robot_description"
        ).string_value
        self._robot = optas.RobotModel(
            urdf_string=self._robot_description, time_derivs=[0, 1]
        )
        self.jacobian_func = self._robot.get_link_geometric_jacobian_function(
            link=self.end_effector_, base_link=self.base_, numpy_output=True
        )
        return

    def quat_2_euler(self, quat):
        """calculates and returns: yaw, pitch, roll from given quaternion"""
        return R.from_quat(quat).as_euler("xyz")

    def convert(self):

        # Convert each row in the dataframe to cartesian coordinates and save in the dataframe
        for index, row in self.joint_data.iterrows():

            joint_state = JointState()
            joint_state.name = ["A1", "A2", "A3", "A4", "A5", "A6", "A7"]
            joint_state.position = [
                row["P1"],
                row["P2"],
                row["P3"],
                row["P4"],
                row["P5"],
                row["P6"],
                row["P7"],
            ]
            joint_state.velocity = [
                row["V1"],
                row["V2"],
                row["V3"],
                row["V4"],
                row["V5"],
                row["V6"],
                row["V7"],
            ]
            joint_state.effort = [
                row["E1"],
                row["E2"],
                row["E3"],
                row["E4"],
                row["E5"],
                row["E6"],
                row["E7"],
            ]
            joint_state.header.stamp = self.get_clock().now().to_msg()
            joint_state.header.frame_id = f"{self.namespace_}/{self.base_}"
            current_robot_state = RobotState()
            current_robot_state.joint_state = joint_state

            request = GetPositionFK.Request()

            request.header.frame_id = f"{self.namespace_}/{self.base_}"
            request.header.stamp = self.get_clock().now().to_msg()

            request.fk_link_names.append(self.end_effector_)
            request.robot_state = current_robot_state

            future = self.fk_client_.call_async(request)

            rclpy.spin_until_future_complete(self, future)
            if future.result() is None:
                self.get_logger().error("Failed to get FK solution")
                return None

            response = future.result()
            if response.error_code.val != MoveItErrorCodes.SUCCESS:
                self.get_logger().error(
                    f"Failed to get FK solution: {response.error_code.val}"
                )
                return None

            pose = response.pose_stamped[0].pose
            euler_pose = self.quat_2_euler(
                np.array(
                    [
                        pose.orientation.x,
                        pose.orientation.y,
                        pose.orientation.z,
                        pose.orientation.w,
                    ]
                )
            )
            self.joint_data.at[index, "X"] = pose.position.x
            self.joint_data.at[index, "Y"] = pose.position.y
            self.joint_data.at[index, "Z"] = pose.position.z
            self.joint_data.at[index, "QX"] = pose.orientation.x
            self.joint_data.at[index, "QY"] = pose.orientation.y
            self.joint_data.at[index, "QZ"] = pose.orientation.z
            self.joint_data.at[index, "QW"] = pose.orientation.w
            self.joint_data.at[index, "Roll"] = euler_pose[0]
            self.joint_data.at[index, "Pitch"] = euler_pose[1]
            self.joint_data.at[index, "Yaw"] = euler_pose[2]

            # Compute the end_effector velocity
            current_joint_angles = [
                joint_state.position[0],
                joint_state.position[1],
                joint_state.position[2],
                joint_state.position[3],
                joint_state.position[4],
                joint_state.position[5],
                joint_state.position[6],
            ]
            current_jacobian = self.jacobian_func(current_joint_angles)

            cart_vel = np.dot(current_jacobian, joint_state.velocity)
            cart_wrench = np.matmul(
                np.linalg.pinv(np.transpose(current_jacobian)), joint_state.effort
            )

            self.joint_data.at[index, "Vx"] = cart_vel[0]
            self.joint_data.at[index, "Vy"] = cart_vel[1]
            self.joint_data.at[index, "Vz"] = cart_vel[2]
            self.joint_data.at[index, "Va"] = cart_vel[3]
            self.joint_data.at[index, "Vb"] = cart_vel[4]
            self.joint_data.at[index, "Vc"] = cart_vel[5]

            # # Overriding Wrench Vals
            # self.joint_data.at[index, "Fx"] = cart_wrench[0]
            # self.joint_data.at[index, "Fy"] = cart_wrench[1]
            # self.joint_data.at[index, "Fz"] = cart_wrench[2]
            # self.joint_data.at[index, "Tx"] = cart_wrench[3]
            # self.joint_data.at[index, "Ty"] = cart_wrench[4]
            # self.joint_data.at[index, "Tz"] = cart_wrench[5]

        self.joint_data.to_csv(self.joint_data_file_path, index=False)


def main(args=None):
    rclpy.init(args=args)

    # with open('config.json', 'r') as config_file:
    #     config = json.load(config_file)

    # processed_folder_path = config['processed_csv_folder_path']
    # csv_name = config['replay_csv_name']
    csv_name = "2013-01-17_00-42-46.csv"
    processed_folder_path = os.path.join(
        MAIN_DIR, "data_collector", "data_collector", "processed_data"
    )
    csv_folder = os.path.join(processed_folder_path, os.path.splitext(csv_name)[0])
    joint_data_file_path = os.path.join(csv_folder, "recorded_data.csv")

    joint_cartesian_converter = JointCartesianConverter(joint_data_file_path)
    joint_cartesian_converter.convert()

    rclpy.shutdown()
