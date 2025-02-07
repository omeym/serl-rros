import pandas as pd
import json
import re
import os
import glob

def parse_robot_data(file_path):
    """
    Parses the robot data file, renames joint columns to J1-J7,
    extracts and renames time, force, and torque values.
    
    Parameters:
        file_path (str): Path to the data file.
    
    Returns:
        pd.DataFrame: A DataFrame containing time, renamed joint values,
                      and renamed force/torque data.
    """
    # Step 1: Read the header line starting with '%'
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('%'):
                header = line.strip().lstrip('%').strip().split()
                break
        else:
            raise ValueError("No header line starting with '%' found in the file.")
    
    # Step 2: Load the data into a DataFrame
    df = pd.read_csv(
        file_path,
        delim_whitespace=True,    # Assuming the data is space-separated
        comment='%',              # Skip any lines starting with '%'
        names=header,             # Use the extracted header
        skiprows=1                # Skip the header line
    )
    
    # Step 3: Identify and rename joint columns
    joint_pattern = re.compile(r'axisQMsr_LBR_iiwa_14_R820_1\[(\d+)\]')
    joint_columns = [col for col in header if joint_pattern.match(col)]
    
    if len(joint_columns) != 7:
        raise ValueError(f"Expected 7 joint columns, found {len(joint_columns)}.")
    
    # Sort joint columns based on their index and rename them to J1-J7
    joint_columns_sorted = sorted(
        joint_columns,
        key=lambda x: int(joint_pattern.match(x).group(1))
    )
    joint_rename_map = {col: f'A{idx+1}' for idx, col in enumerate(joint_columns_sorted)}
    df.rename(columns=joint_rename_map, inplace=True)
    
    # Step 4: Extract and combine time columns
    if 'ZeitInSec' not in df.columns or 'ZeitInNanoSec' not in df.columns:
        raise ValueError("Required time columns 'ZeitInSec' and/or 'ZeitInNanoSec' are missing.")
    
    # Combine 'ZeitInSec' and 'ZeitInNanoSec' into a single 'time' column in seconds
    df['time'] = df['ZeitInSec'] + df['ZeitInNanoSec'] * 1e-9
    
    # Drop the original time columns as they are now combined
    df.drop(columns=['ZeitInSec', 'ZeitInNanoSec'], inplace=True)
    
    # Step 5: Extract and rename force and torque columns
    force_columns = {
        'cartForce1_X': 'Fx',
        'cartForce1_Y': 'Fy',
        'cartForce1_Z': 'Fz'
    }
    torque_columns = {
        'cartTorque1_TauX': 'Tx',
        'cartTorque1_TauY': 'Ty',
        'cartTorque1_TauZ': 'Tz'
    }
    
    # Verify that the required force and torque columns exist
    missing_force_columns = [col for col in force_columns.keys() if col not in df.columns]
    missing_torque_columns = [col for col in torque_columns.keys() if col not in df.columns]
    missing_columns = missing_force_columns + missing_torque_columns
    if missing_columns:
        raise ValueError(f"The following required columns are missing in the data: {missing_columns}")
    
    # Rename force and torque columns
    df.rename(columns=force_columns, inplace=True)
    df.rename(columns=torque_columns, inplace=True)
    
    # Step 6: Select the relevant columns
    selected_columns = ['time'] + list(joint_rename_map.values()) + list(force_columns.values()) + list(torque_columns.values())
    df_selected = df[selected_columns]
    
    return df_selected


def main():

    with open('config.json', 'r') as config_file:
        config = json.load(config_file)

    log_folder_path = config['log_folder_path']
    save_df_path = config['processed_csv_folder_path']

    os.makedirs(save_df_path, exist_ok=True)

    log_files = glob.glob(os.path.join(log_folder_path, "*.log"))

    for file_path in log_files:
        returned_df = parse_robot_data(file_path=file_path)
        save_file_path = os.path.join(save_df_path, os.path.splitext(os.path.basename(file_path))[0] + '.csv')
        returned_df.to_csv(save_file_path, index=False)

if __name__ == "__main__":
    main()