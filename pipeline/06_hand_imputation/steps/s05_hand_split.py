# Import required libraries
from src.utils import *
from src.performers import PERFORMER_MAP, performer_id
import pandas as pd
import numpy as np
import os
from tqdm import tqdm


fps = 120  # Frames per second (FPS)
sec = 3
step = 1
motion_ext = '.csv'

for (c, n), trials in sorted(PERFORMER_MAP.items()):
    pid = performer_id(c, n)

    # Check if folder exists
    folder_path = f'Data/Pipeline/06_hand_imputation/Imp/replace/Body/{pid}'
    if not os.path.exists(folder_path):
        print(f"Folder does not exist, skipping: {folder_path}")
        continue

    motionFilePathLst = makeMotionFilePathLst(folder_path, motion_ext)

    for filePath in tqdm(motionFilePathLst, desc=pid):

        data = pd.read_csv(filePath)

        # Identify columns belonging to right ("R"), left ("L"), and others
        right_columns = [col for col in data.columns if col.startswith('R')]
        left_columns = [col for col in data.columns if col.startswith('L')]
        
        
        # 1. Extract joint names
        # Joint names are the first part separated by '_' from column names
        r_joints = sorted(set(col.split('_')[0] for col in right_columns))

        # 2. Sort in x, y, z order for each joint
        r_sorted_columns = []
        for joint in r_joints:
            for axis in ['x', 'y', 'z']:
                r_sorted_columns.extend([col for col in right_columns if col == f"{joint}_{axis}"])

        
        # 1. Extract joint names
        # Joint names are the first part separated by '_' from column names
        l_joints = sorted(set(col.split('_')[0] for col in left_columns))

        # 2. Sort in x, y, z order for each joint
        l_sorted_columns = []
        for joint in l_joints:
            for axis in ['x', 'y', 'z']:
                l_sorted_columns.extend([col for col in left_columns if col == f"{joint}_{axis}"])

        # Separate left and right hand data
        rightHand = data[r_sorted_columns]
        leftHand = data[l_sorted_columns]
        
        other_columns = [col for col in data.columns if col not in r_sorted_columns + l_sorted_columns]

        # Step 2: Wrist Translation to Origin
        rWrist_columns = ['RWrist_x', 'RWrist_y', 'RWrist_z']
        lWrist_columns = ['LWrist_x', 'LWrist_y', 'LWrist_z']

        # Initialize adjusted hand data
        rightHand_adjusted = rightHand.copy()
        leftHand_adjusted = leftHand.copy()

        if all(col in rightHand.columns for col in rWrist_columns):
            rWrist_origin = rightHand[rWrist_columns].values  # convert to numpy array
            rightHand_adjusted = rightHand.copy().values      # convert to numpy array

            # Expand rWrist_origin to shape (1000, 75)
            expanded_rWrist_origin = np.tile(rWrist_origin, (1, rightHand.shape[1] // 3))
            rightHand_adjusted -= expanded_rWrist_origin      # subtract from the whole array

            # Convert back to a DataFrame
            rightHand_adjusted = pd.DataFrame(rightHand_adjusted, columns=rightHand.columns)

        if all(col in leftHand.columns for col in lWrist_columns):
            lWrist_origin = leftHand[lWrist_columns].values
            leftHand_adjusted = leftHand.copy().values

            # Expand lWrist_origin to shape (1000, 75)
            expanded_lWrist_origin = np.tile(lWrist_origin, (1, leftHand.shape[1] // 3))
            leftHand_adjusted -= expanded_lWrist_origin       # subtract from the whole array

            # Convert back to a DataFrame
            leftHand_adjusted = pd.DataFrame(leftHand_adjusted, columns=leftHand.columns)



        # Function to create rotation matrix
        def rotation_matrix_to_align_with_z(normal):
            z_axis = np.array([0, 0, 1])
            normal = normal / np.linalg.norm(normal)  # Normalize
            v = np.cross(normal, z_axis)
            s = np.linalg.norm(v)
            c = np.dot(normal, z_axis)

            if s == 0:  # Already aligned with Z-axis
                return np.eye(3)

            vx = np.array([[0, -v[2], v[1]],
                        [v[2], 0, -v[0]],
                        [-v[1], v[0], 0]])
            R = np.eye(3) + vx + np.dot(vx, vx) * ((1 - c) / (s ** 2))
            return R

        # Align hand data to satisfy specific configuration
        def align_specific_configuration(hand_data, Wrist_columns, lim_columns, llm_columns):
            """Align hand data to satisfy the conditions:
            LWrist = (0, 0, 0), LIM = (n, n, 0), LLM = (n, 0, 0).
            """
            aligned_hand = []
            rotation_to_z_matrices = []
            rotation_to_x_matrices = []

            for i in range(len(hand_data)):
                # Extract Wrist, LIM, and LLM positions
                Wrist = hand_data[Wrist_columns].iloc[i].values
                lim = hand_data[lim_columns].iloc[i].values - Wrist  # Relative to Wrist
                llm = hand_data[llm_columns].iloc[i].values - Wrist  # Relative to Wrist

                # Step 1: Rotate to align the plane defined by LIM and LLM to Z=0
                normal_current = np.cross(lim, llm)
                normal_current = normal_current / np.linalg.norm(normal_current)  # Normalize
                rotation_to_z = rotation_matrix_to_align_with_z(normal_current)
                rotation_to_z_matrices.append(rotation_to_z)

                lim_rotated = np.dot(lim, rotation_to_z.T)
                llm_rotated = np.dot(llm, rotation_to_z.T)

                # Step 2: Rotate further to align LLM with the X-axis
                angle_to_x = np.arctan2(llm_rotated[1], llm_rotated[0])
                rotation_to_x = np.array([
                    [np.cos(-angle_to_x), -np.sin(-angle_to_x), 0],
                    [np.sin(-angle_to_x), np.cos(-angle_to_x), 0],
                    [0, 0, 1]
                ])
                rotation_to_x_matrices.append(rotation_to_x)

                # Update points after rotation
                row_points = hand_data.iloc[i].values.reshape(-1, 3) - Wrist  # Relative to Wrist
                row_points_rotated = np.dot(row_points, rotation_to_z.T)
                row_points_final = np.dot(row_points_rotated, rotation_to_x.T)

                # Store final aligned points
                aligned_hand.append(row_points_final.flatten())

            # Convert aligned hand to DataFrame
            aligned_hand = pd.DataFrame(aligned_hand, columns=hand_data.columns)

            return aligned_hand, rotation_to_z_matrices, rotation_to_x_matrices

        
        os.makedirs(f'Data/Pipeline/06_hand_imputation/AD/Split/Right/{pid}',exist_ok=True)
        os.makedirs(f'Data/Pipeline/06_hand_imputation/AD/Split/Left/{pid}',exist_ok=True)


        filename = os.path.basename(filePath)
        rightHandSplitPath=f'Data/Pipeline/06_hand_imputation/AD/Split/Right/{pid}/{filename}'
        leftHandSplitPath=f'Data/Pipeline/06_hand_imputation/AD/Split/Left/{pid}/{filename}'


        # Save rotated data to CSV
        aligned_left_hand_specific, left_rotation_to_z, left_rotation_to_x = align_specific_configuration(
            leftHand_adjusted,
            Wrist_columns=['LWrist_x', 'LWrist_y', 'LWrist_z'],
            lim_columns=['LIM_x', 'LIM_y', 'LIM_z'],
            llm_columns=['LLM_x', 'LLM_y', 'LLM_z']
        )
        aligned_left_hand_specific.to_csv(leftHandSplitPath, index=False)

        aligned_right_hand_specific, right_rotation_to_z, right_rotation_to_x = align_specific_configuration(
            rightHand_adjusted,
            Wrist_columns=['RWrist_x', 'RWrist_y', 'RWrist_z'],
            lim_columns=['RIM_x', 'RIM_y', 'RIM_z'],
            llm_columns=['RLM_x', 'RLM_y', 'RLM_z']
        )
        aligned_right_hand_specific.to_csv(rightHandSplitPath, index=False)


        os.makedirs(f'Data/Pipeline/06_hand_imputation/Metadata/AD/Split/Right/{pid}',exist_ok=True)
        os.makedirs(f'Data/Pipeline/06_hand_imputation/Metadata/AD/Split/Left/{pid}',exist_ok=True)
        

        rightHandMetaPath=f'Data/Pipeline/06_hand_imputation/Metadata/AD/Split/Right/{pid}/{filename}'
        leftHandMetaPath=f'Data/Pipeline/06_hand_imputation/Metadata/AD/Split/Left/{pid}/{filename}'
        
        
        # Save transformation metadata for left hand
        metadata_list_left = []
        for i in range(len(lWrist_origin)):
            metadata = {
                'frame': i,
                'Wrist_x': lWrist_origin[i][0],
                'Wrist_y': lWrist_origin[i][1],
                'Wrist_z': lWrist_origin[i][2],
                'rotation_to_z': ','.join(map(str, left_rotation_to_z[i].flatten().tolist())) if i < len(left_rotation_to_z) else '',
                'rotation_to_x': ','.join(map(str, left_rotation_to_x[i].flatten().tolist())) if i < len(left_rotation_to_x) else ''
            }
            metadata_list_left.append(metadata)
        metadata_df_left = pd.DataFrame(metadata_list_left)
        metadata_df_left.to_csv(leftHandMetaPath, index=False)

        # Save transformation metadata for right hand
        metadata_list_right = []
        for i in range(len(rWrist_origin)):
            metadata = {
                'frame': i,
                'Wrist_x': rWrist_origin[i][0],
                'Wrist_y': rWrist_origin[i][1],
                'Wrist_z': rWrist_origin[i][2],
                'rotation_to_z': ','.join(map(str, right_rotation_to_z[i].flatten().tolist())) if i < len(right_rotation_to_z) else '',
                'rotation_to_x': ','.join(map(str, right_rotation_to_x[i].flatten().tolist())) if i < len(right_rotation_to_x) else ''
            }
            metadata_list_right.append(metadata)
        metadata_df_right = pd.DataFrame(metadata_list_right)
        metadata_df_right.to_csv(rightHandMetaPath, index=False)
