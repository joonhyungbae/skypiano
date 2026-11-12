from src.performers import PERFORMER_MAP, performer_id
# Import required libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegWriter
from tqdm import tqdm
from src.utils import *
from src.visualization import *
from scipy.stats import zscore

# Restore original data
def restore_original_data(aligned_data, metadata_df):

    restored_data = []

    for i, row in metadata_df.iterrows():
        Wrist_origin = np.array([row['Wrist_x'], row['Wrist_y'], row['Wrist_z']])
        rotation_to_z = np.array(list(map(float, row['rotation_to_z'].split(',')))).reshape(3, 3)
        rotation_to_x = np.array(list(map(float, row['rotation_to_x'].split(',')))).reshape(3, 3)

        # Step 1: Apply inverse of 2nd rotation
        inverse_rotation_to_x = np.linalg.inv(rotation_to_x)
        points_rotated_back_x = np.dot(aligned_data.iloc[i].values.reshape(-1, 3), inverse_rotation_to_x.T)

        # Step 2: Apply inverse of 1st rotation
        inverse_rotation_to_z = np.linalg.inv(rotation_to_z)
        points_rotated_back_z = np.dot(points_rotated_back_x, inverse_rotation_to_z.T)

        # Step 3: Restore Wrist positions
        points_restored = points_rotated_back_z + Wrist_origin

        restored_data.append(points_restored.flatten())

    return pd.DataFrame(restored_data, columns=aligned_data.columns)


sec = 3
step=1
motionExt = '.csv'

for way in ['Left','Right']:

    motionFilePathLst=[]
    for (c_val, n_val), trials in sorted(PERFORMER_MAP.items()):
        pid = performer_id(c_val, n_val)
        try:
            baseFolderPath = f'Data/Pipeline/06_hand_imputation/Imp/concat/{way}/{pid}'
            
            # Check if folder exists
            if not os.path.exists(baseFolderPath):
                print(f"Folder does not exist: {baseFolderPath}")
                continue
                
            filePathLst = makeMotionFilePathLst(baseFolderPath, motionExt)  # Path list creation
            
            if filePathLst:  # Only if file list is not empty
                motionFilePathLst=motionFilePathLst+filePathLst
            else:
                print(f"No files found: {baseFolderPath}")
                
        except Exception as e:
            print(f"Error processing performer {pid} ({way}): {e}")
            pass
        
    for path in tqdm(motionFilePathLst):
        try:
            norm_path = path.replace('\\', '/')
            savePath=norm_path.replace('concat','restore')

            if os.path.exists(savePath):
                continue

            # Extract pid and filename from path: .../Imp/concat/{way}/{pid}/{filename}.csv
            parts = norm_path.split('/')
            pid = parts[-2]
            filename = parts[-1].replace('.csv', '')

            # Create save directory
            save_dir = f'Data/Pipeline/06_hand_imputation/Imp/restore/{way}/{pid}'
            os.makedirs(save_dir, exist_ok=True)

            metaPath=f'Data/Pipeline/06_hand_imputation/Metadata/AD/Split/{way}/{pid}/{filename}.csv'
            
            # Check if metadata file exists
            if not os.path.exists(metaPath):
                print(f"Metadata file does not exist: {metaPath}")
                continue
            
            # Check if motion data file exists
            if not os.path.exists(path):
                print(f"Motion data file does not exist: {path}")
                continue
            
            data=pd.read_csv(path)
            metaData=pd.read_csv(metaPath)
            metaData=metaData[:len(data)]
            
            # Example to restore right hand
            restored_hand = restore_original_data(
                data,
                metaData
            )
            
            restored_hand.to_csv(savePath, index=False)
            
        except Exception as e:
            print(f"Restore failed for {way}/{path}: {type(e).__name__}: {e}")
            continue

