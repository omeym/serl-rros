'''
FORMAT OF THE CREATED PKL FILE:
    The pkl file is a dictionary with the following keys:
    dict(
            observations=obs,
            actions=actions,
            next_observations=next_obs,
            rewards=rew,
            masks=1.0 - done,
            dones=done,
        )
    where
    observations : {
        "state": gym.spaces.Box(-np.inf, np.inf, shape=(7 + 6 + 3 + 3,)),
        "wrist_1": gym.spaces.Box(0, 255, shape=(128, 128, 3), dtype=np.uint8),
        "wrist_2": gym.spaces.Box(0, 255, shape=(128, 128, 3), dtype=np.uint8),
    }
    actions = gym.spaces.Box(
                np.ones((7,), dtype=np.float32) * -1,
                np.ones((7,), dtype=np.float32),
            )
    The action space's last element is actually for the gripper, but we are not using it.
    It should be x,y,z,roll,pitch,yaw
    state : concatenation of tcp_poses (7), tcp_velocities (6), tcp_forces (3), tcp_torques (3)
    image names are in the format {timestamp}_A.png and {timestamp}_B.png
'''


import numpy as np 
import pandas as pd 
import pickle 
import os 
import json
from PIL import Image
from scipy.spatial.transform import Rotation as R
from transformations import construct_adjoint_matrix, construct_homogeneous_matrix




class SingleTrajectoryConverter:
    def __init__(self, recorded_data_file_path, imageA_folder_path, imageB_folder_path, action_scale, target_pose, reward_threshold):
        
        self.recorded_data_file_path = recorded_data_file_path
        self.imageA_folder_path = imageA_folder_path
        self.imageB_folder_path = imageB_folder_path
        self.action_scale = action_scale
        self.target_pose = target_pose
        self.reward_threshold = reward_threshold

        self.recorded_data = pd.read_csv(recorded_data_file_path)

        self.adjoint_matrix = None
        self.T_r_o_inv = None

        self.transitions = []

    def transform_observation(self, obs):
        """
        Transform observations from spatial(base) frame into body(end-effector) frame
        using the adjoint matrix
        """
        adjoint_inv = np.linalg.inv(self.adjoint_matrix)
        obs[['V1', 'V2', 'V3', 'V4', 'V5', 'V6']] = adjoint_inv @ obs[['V1', 'V2', 'V3', 'V4', 'V5', 'V6']].values

        T_b_o = construct_homogeneous_matrix(obs[['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7']].values)
        T_b_r = self.T_r_o_inv @ T_b_o

        # Reconstruct transformed tcp_pose vector
        p_b_r = T_b_r[:3, 3]
        theta_b_r = R.from_matrix(T_b_r[:3, :3]).as_quat()
        obs[['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7']] = np.concatenate((p_b_r, theta_b_r))

        return obs

    def calculate_next_action(self, curr_pos, next_pos):
        action = np.zeros(7)
        action[:3] = (next_pos[:3] - curr_pos[:3]) / self.action_scale[0]
        action[3:6] = (R.from_quat(next_pos[3:7]) * R.from_quat(curr_pos[3:7]).inv()).as_euler("xyz") / self.action_scale[1]
        action[6] = 0.0
        return action

    def convertToPickle(self):

        observations = []
        actions = []
        next_observations = []
        rewards = []
        masks = []
        dones = []

        done = 0.0

        first_observation = self.recorded_data.iloc[0]
        # Construct the adjoint matrix and T_r_o_inv using the first observation
        # T_r_o_inv remains constant but the adjoint matrix updates every iteration
        curr_pos = first_observation[['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7']].values
        self.adjoint_matrix = construct_adjoint_matrix(curr_pos)
        self.T_r_o_inv = np.linalg.inv(construct_homogeneous_matrix(curr_pos))
        next_transformed_obs = self.transform_observation(first_observation)


        for i in range(1, len(self.recorded_data) - 1):

            current_row = self.recorded_data.iloc[i-1]
            next_row = self.recorded_data.iloc[i]

            curr_time = current_row["timestamp"]
            next_time = next_row["timestamp"]

            img_size = (128, 128)

            curr_wrist_1_image = Image.open(os.path.join(self.imageA_folder_path, f"{curr_time}_A.jpg")).resize(img_size)
            curr_wrist_2_image = Image.open(os.path.join(self.imageB_folder_path, f"{curr_time}_B.jpg")).resize(img_size)

            next_wrist_1_image = Image.open(os.path.join(self.imageA_folder_path, f"{next_time}_A.jpg")).resize(img_size)
            next_wrist_2_image = Image.open(os.path.join(self.imageB_folder_path, f"{next_time}_B.jpg")).resize(img_size)

            curr_obs_no_transform_pose = current_row[['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7']].values
            next_obs_no_transform_pose = next_row[['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7']].values
            action = self.calculate_next_action(curr_obs_no_transform_pose, next_obs_no_transform_pose)
        
            self.adjoint_matrix = construct_adjoint_matrix(curr_obs_no_transform_pose)

            curr_transformed_obs = next_transformed_obs
            next_transformed_obs = self.transform_observation(next_row)
            
            curr_euler_angles = R.from_quat(curr_transformed_obs[['P4', 'P5', 'P6', 'P7']].values).as_euler('xyz')
            next_euler_angles = R.from_quat(next_transformed_obs[['P4', 'P5', 'P6', 'P7']].values).as_euler('xyz')

            observation = {
                "state": np.concatenate([
                    curr_transformed_obs[['P1', 'P2', 'P3']].values,
                    curr_euler_angles,
                    curr_transformed_obs[['V1', 'V2', 'V3', 'V4', 'V5', 'V6']].values,
                    curr_transformed_obs[['Fx', 'Fy', 'Fz']].values,
                    curr_transformed_obs[['Tx', 'Ty', 'Tz']].values
                ]),
                "wrist_1": np.array(curr_wrist_1_image),
                "wrist_2": np.array(curr_wrist_2_image)
            }

            next_observation = {
                "state": np.concatenate([
                    next_transformed_obs[['P1', 'P2', 'P3']].values,
                    next_euler_angles,
                    next_transformed_obs[['V1', 'V2', 'V3', 'V4', 'V5', 'V6']].values,
                    next_transformed_obs[['Fx', 'Fy', 'Fz']].values,
                    next_transformed_obs[['Tx', 'Ty', 'Tz']].values
                ]),
                "wrist_1": np.array(next_wrist_1_image),
                "wrist_2": np.array(next_wrist_2_image)
            }

            if done != 1.0:
                tcp_pose = current_row[['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw']].values
                tcp_pose[3:] = np.abs(tcp_pose[3:])
                if np.all(tcp_pose - self.target_pose <= self.reward_threshold):
                    reward = 1.0
                    done = 1.0
                else:
                    reward = 0.0
            else:
                reward = 0.0

            
            data_dict = {
                "observations": observation,
                "actions": action,
                "next_observations": next_observation,
                "rewards": reward,
                "masks": 1.0 - done,
                "dones": done
            }

            self.transitions.append(data_dict)

            # observations.append(observation)
            # actions.append(action)
            # next_observations.append(next_observation)
            # rewards.append(reward)
            # masks.append(1.0 - done)
            # dones.append(done)

            curr_wrist_1_image.close()
            curr_wrist_2_image.close()
            next_wrist_1_image.close()
            next_wrist_2_image.close()

        return self.transitions

def main():

    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
    processed_folder_path = config['processed_csv_folder_path']
    output_pkl_file_path = os.path.join(processed_folder_path, 'recorded_data.pkl')

    action_scale = np.array(config['ACTION_SCALE'])
    target_pose = np.array(config['TARGET_POSE'])
    reward_threshold = np.array(config['REWARD_THRESHOLD'])


    all_transitions = []
    
    for folder_name in os.listdir(processed_folder_path):
        folder_path = os.path.join(processed_folder_path, folder_name)
        if os.path.isdir(folder_path):
            recorded_data_file_path = os.path.join(folder_path, 'recorded_data.csv')
            imageA_folder_path = os.path.join(folder_path, 'imageA')
            imageB_folder_path = os.path.join(folder_path, 'imageB')
            if os.path.exists(recorded_data_file_path) and os.path.exists(imageA_folder_path) and os.path.exists(imageB_folder_path):
                singleTrajectoryConverter = SingleTrajectoryConverter(
                    recorded_data_file_path, 
                    imageA_folder_path, 
                    imageB_folder_path,
                    action_scale,
                    target_pose,
                    reward_threshold)
                transitions = singleTrajectoryConverter.convertToPickle()
                all_transitions.extend(transitions)

    with open(output_pkl_file_path, 'wb') as f:
        pickle.dump(all_transitions, f)

    print(f"Data saved to {output_pkl_file_path}")


if __name__ == "__main__":
    main()

