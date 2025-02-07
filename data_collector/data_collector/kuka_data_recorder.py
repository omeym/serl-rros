import rclpy
import os
import pandas as pd
import csv
import math
import json
from cv_bridge import CvBridge
import cv2
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.action import ActionClient
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import WrenchStamped
from message_filters import Subscriber, ApproximateTimeSynchronizer

class KukaDataRecorder(Node):
    def __init__(self, data_folder, joint_states_file):
        super().__init__("kuka_data_recorder")

        self.data_folder = data_folder
        self.joint_states_file = joint_states_file
        self.save_data_folder = os.path.join(self.data_folder, os.path.splitext(self.joint_states_file)[0])
        os.makedirs(self.save_data_folder, exist_ok=True)
        # Create two folders for the images
        self.imageA_folder = os.path.join(self.save_data_folder, "imageA")
        self.imageB_folder = os.path.join(self.save_data_folder, "imageB")
        os.makedirs(self.imageA_folder, exist_ok=True)
        os.makedirs(self.imageB_folder, exist_ok=True)

        # topic_callback_group = MutuallyExclusiveCallbackGroup()
        # client_callback_group = MutuallyExclusiveCallbackGroup()

        # Initialize everything for trajectory replaying
        self.feedback = None
        self._action_client = ActionClient(self, FollowJointTrajectory, '/lbr/joint_trajectory_controller/follow_joint_trajectory')
        self.joint_names = ["A1", "A2", "A3", "A4", "A5", "A6", "A7"]
        self.joint_trajectories = self.readJoinStatesFromCsv(os.path.join(self.data_folder, "processed", self.joint_states_file))
        # This can be done through goal sending
        self.initializeRobotPose()

        # Create a synchronized callback for all the information
        self.imageA_topic = "/camera1/camera1/color/image_raw"
        self.imageB_topic = "/camera2/camera2/color/image_raw"
        self.joint_state_topic = "/lbr/joint_states"
        self.wrench_topic = "/lbr/force_torque_broadcaster/wrench"

        self.imageA_sub = Subscriber(self, Image, self.imageA_topic)
        self.imageB_sub = Subscriber(self, Image, self.imageB_topic)
        self.joint_state_sub = Subscriber(self, JointState, self.joint_state_topic)
        self.wrench_sub = Subscriber(self, WrenchStamped, self.wrench_topic)

        self.synchronizer =  ApproximateTimeSynchronizer(
            [self.imageA_sub, self.imageB_sub, self.joint_state_sub, self.wrench_sub],
            queue_size=100,
            slop=0.01,
        )

        self.cvbridge = CvBridge()

        self.synchronizer.registerCallback(self.recordSyncData)
        self.get_logger().info("Data Synchronizer Node Initialized")

        # self.dataframe = pd.DataFrame(columns=[
        #     'timestamp', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7',
        #     'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7',
        #     'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz'
        # ])
        self.dataframe = []

        # This needs to be done through moveit planner since the points need to be interpolated to a trajectory
        self.executeTrajectory()


    def recordSyncData(self, imageA_msg, imageB_msg, joint_state_msg, wrench_msg):
        # The timestamp of imageA_msg will be used to name the columns
        timestamp = f"{imageA_msg.header.stamp.sec}_{imageA_msg.header.stamp.nanosec}"

        self.get_logger().info(f"Recording data at timestamp {timestamp}")

        # Save the joint state
        data_row = {'timestamp': timestamp}

        for i in range(7):
            # imp imp: Remember that kuka publishes A1, A2, A4, A3, A5, A6, A7 (NOTE THE ORDER)
            j = i
            if i == 3:
                j = 4
            if i == 4: 
                j = 3
            data_row[f'P{j+1}'] = joint_state_msg.position[i]
            data_row[f'V{j+1}'] = joint_state_msg.velocity[i]
            data_row[f'E{j+1}'] = joint_state_msg.effort[i]

        # Save the wrench
        data_row['Fx'] = wrench_msg.wrench.force.x
        data_row['Fy'] = wrench_msg.wrench.force.y
        data_row['Fz'] = wrench_msg.wrench.force.z
        data_row['Tx'] = wrench_msg.wrench.torque.x
        data_row['Ty'] = wrench_msg.wrench.torque.y
        data_row['Tz'] = wrench_msg.wrench.torque.z

        self.dataframe.append(data_row)

        # Save the images
        imageA_filename = os.path.join(self.imageA_folder, f"{timestamp}_A.jpg")
        imageB_filename = os.path.join(self.imageB_folder, f"{timestamp}_B.jpg")
        image_A = self.cvbridge.imgmsg_to_cv2(imageA_msg, desired_encoding='bgr8')
        cv2.imwrite(imageA_filename, image_A)
        image_B = self.cvbridge.imgmsg_to_cv2(imageB_msg, desired_encoding='bgr8')
        cv2.imwrite(imageB_filename, image_B)


    def readJoinStatesFromCsv(self, file_path):
        joint_states = []
        with open(file_path, mode='r') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                joint_states.append([math.radians(float(row['A1'])), math.radians(float(row['A2'])), math.radians(float(row['A3'])),
                                    math.radians(float(row['A4'])), math.radians(float(row['A5'])), math.radians(float(row['A6'])), math.radians(float(row['A7']))])
        print(joint_states[0])
        return joint_states

    def initializeRobotPose(self):
        # The robot needs to be initialized to the first position of the trajectory 
        # to ensure that there is no sudden movement when the trajectory is replayed
        goal_msg = FollowJointTrajectory.Goal()
        trajectory_msg = JointTrajectory()
        trajectory_msg.joint_names = self.joint_names
        
        self.get_logger().info(f"Processing trajectory point init")

        point = JointTrajectoryPoint()
        point.positions = self.joint_trajectories[0]
        point.time_from_start.sec = 4  # Set the seconds part to 0

        trajectory_msg.points.append(point)
        goal_msg.trajectory = trajectory_msg
        self._action_client.wait_for_server()

        send_goal_future = self._action_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)

        rclpy.spin_until_future_complete(self, send_goal_future)

        goal_handle = send_goal_future.result()

        if not goal_handle.accepted:
            self.get_logger().info(f"Goal for init point was rejected")
            return
        
        self.get_logger().info(f"Goal for init point was accepted")
        get_result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, get_result_future)
        result = get_result_future.result().result
        self.get_logger().info(f"Result : {result}, Initial position reached")
        print("Press Enter to continue ...")
        input()

    def executeTrajectory(self):
        goal_msg = FollowJointTrajectory.Goal()
        trajectory_msg = JointTrajectory()
        trajectory_msg.joint_names = self.joint_names
        for i, joint_values in enumerate(self.joint_trajectories):
            point = JointTrajectoryPoint()
            point.positions = joint_values
            point.time_from_start = rclpy.duration.Duration(seconds=(i+1) * 0.1).to_msg()
            trajectory_msg.points.append(point)
        goal_msg.trajectory = trajectory_msg
        self._action_client.wait_for_server()

        send_goal_future = self._action_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)

        # rclpy.spin_until_future_complete(self, send_goal_future)

        # goal_handle = send_goal_future.result()

        # if not goal_handle.accepted:
        #     self.get_logger().info(f"Goal for trajectory was rejected")
        #     return
        
        # self.get_logger().info(f"Goal for trajectory was accepted")
        # get_result_future = goal_handle.get_result_async()
        # rclpy.spin_until_future_complete(self, get_result_future)
        # result = get_result_future.result().result
        # self.get_logger().info(f"Result : {result}, Trajectory executed?")

        send_goal_future.add_done_callback(self.goal_response_callback)
    
    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info(f"Goal for trajectory was rejected")
            return
        self.get_logger().info(f"Goal for trajectory was accepted")
        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self.result_callback)

    def result_callback(self, future):
        result = future.result()
        if result.result.error_code == FollowJointTrajectory.Result.SUCCESSFUL:
            self.get_logger().info("WOHOOO")
            self.shutdown_node()
        else:
            self.get_logger().info("OHH NO")

    def feedback_callback(self, feedback_msg):
        self.feedback = feedback_msg.feedback

    def shutdown_node(self):
        self.get_logger().info("Shutting down node now")
        self.save_data()
        self.destroy_node()
        rclpy.shutdown()
    
    def save_data(self):
        df = pd.DataFrame(self.dataframe)
        save_path = os.path.join(self.save_data_folder, 'recorded_data.csv')
        df.to_csv(save_path, index=False)
        self.get_logger().info("Saved data!")

def main(args=None):

    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
    
    data_folder = config['log_folder_path']
    joint_states_file = config['replay_csv_name']
    rclpy.init(args=args)
    executor = MultiThreadedExecutor()
    node = KukaDataRecorder(data_folder=data_folder, joint_states_file=joint_states_file)
    executor.add_node(node)
    try:
        
        executor.spin()
    except Exception as e:
        node.get_logger().error(f"An error occurred: {e}")
    finally:
        # if hasattr(node, 'dataframe'):
        #     df = pd.DataFrame(node.dataframe)
        #     save_path = os.path.join(node.data_folder, 'recorded_data.csv')
        #     df.to_csv(save_path, index=False)
        #     node.get_logger().info(f"Recorded data saved!")
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()

if __name__ == "__main__":
    main()