import os
import cv2
import glob
import time
from datetime import datetime
from tqdm import tqdm
import multiprocessing.pool as mpp
import numpy as np

# Date/time string formats
FORMAT_DATE = "%Y%m%d"
FORMAT_DATETIME = "%Y%m%d%H%M%S"

# Joint column definitions
joint_cols=['RIM_x', 'RIM_y', 'RIM_z', 'RIP_x', 'RIP_y', 'RIP_z', 
            'RLM_x', 'RLM_y', 'RLM_z', 'RLP_x', 'RLP_y', 'RLP_z', 
            'RMM_x', 'RMM_y', 'RMM_z', 'RMP_x', 'RMP_y', 'RMP_z', 
            'RRM_x', 'RRM_y', 'RRM_z', 'RRP_x', 'RRP_y', 'RRP_z', 
            'RTC_x', 'RTC_y', 'RTC_z', 'RTM_x', 'RTM_y', 'RTM_z', 
            'RElbow_x', 'RElbow_y', 'RElbow_z', 'RSh_x', 'RSh_y', 'RSh_z', 
            'RWrist_x', 'RWrist_y', 'RWrist_z',
                 
            'LIM_x', 'LIM_y', 'LIM_z', 'LIP_x', 'LIP_y', 'LIP_z', 
            'LLM_x', 'LLM_y', 'LLM_z', 'LLP_x', 'LLP_y', 'LLP_z', 
            'LMM_x', 'LMM_y', 'LMM_z', 'LMP_x', 'LMP_y', 'LMP_z', 
            'LRM_x', 'LRM_y', 'LRM_z', 'LRP_x', 'LRP_y', 'LRP_z', 
            'LTC_x', 'LTC_y', 'LTC_z', 'LTM_x', 'LTM_y', 'LTM_z', 
            'LElbow_x', 'LElbow_y', 'LElbow_z', 'LSh_x', 'LSh_y', 'LSh_z', 
            'LWrist_x', 'LWrist_y', 'LWrist_z']

# Body joint columns for body imputation
# body_joint_cols: includes Elbow/Sh — used to compute model context features (s01-s03)
# body_fill_cols: excludes Elbow/Sh — the actual NaN-replacement targets in s04
# (Stage 06 handles wrist through fingers only; Elbow/Sh are handled separately downstream)
body_joint_cols = ['RWrist_x', 'RWrist_y', 'RWrist_z',
                   'RIM_x', 'RIM_y', 'RIM_z',
                   'RLM_x', 'RLM_y', 'RLM_z',
                   'RElbow_x', 'RElbow_y', 'RElbow_z',
                   'RSh_x', 'RSh_y', 'RSh_z',
                   'LWrist_x', 'LWrist_y', 'LWrist_z',
                   'LIM_x', 'LIM_y', 'LIM_z',
                   'LLM_x', 'LLM_y', 'LLM_z',
                   'LElbow_x', 'LElbow_y', 'LElbow_z',
                   'LSh_x', 'LSh_y', 'LSh_z']

body_fill_cols = ['RWrist_x', 'RWrist_y', 'RWrist_z',
                  'RIM_x', 'RIM_y', 'RIM_z',
                  'RLM_x', 'RLM_y', 'RLM_z',
                  'LWrist_x', 'LWrist_y', 'LWrist_z',
                  'LIM_x', 'LIM_y', 'LIM_z',
                  'LLM_x', 'LLM_y', 'LLM_z']

# Joint connections for motion calculation
connections = {
    # Right hand thumb
    'RTM': 'RTC', 'RTC': 'RWrist',
    # Right hand index finger
    'RIP': 'RIM', 'RIM': 'RWrist',
    # Right hand middle finger
    'RMP': 'RMM', 'RMM': 'RWrist',
    # Right hand ring finger
    'RRP': 'RRM', 'RRM': 'RWrist',
    # Right hand little finger
    'RLP': 'RLM', 'RLM': 'RWrist',
    # Right arm
    'RWrist': 'RElbow', 'RElbow': 'RSh',
    # Left hand thumb
    'LTM': 'LTC', 'LTC': 'LWrist',
    # Left hand index finger
    'LIP': 'LIM', 'LIM': 'LWrist',
    # Left hand middle finger
    'LMP': 'LMM', 'LMM': 'LWrist',
    # Left hand ring finger
    'LRP': 'LRM', 'LRM': 'LWrist',
    # Left hand little finger
    'LLP': 'LLM', 'LLM': 'LWrist',
    # Left arm
    'LWrist': 'LElbow', 'LElbow': 'LSh'
}

# Body connections for body joint calculations
body_connections = {
    'RIM': 'RWrist',
    'RLM': 'RWrist',
    'RWrist': 'RElbow', 'RElbow': 'RSh',
    'LIM': 'LWrist',
    'LLM': 'LWrist',
    'LWrist': 'LElbow', 'LElbow': 'LSh',
}

# Parallel-processing progress visualization built for jupyter
def namePathList(path, Ext):
    pathListdir = path  # path for the listdir package, e.g. 'C:/data'
    pathGlob = path+'/*'  # path for the glob package, e.g. 'C:/data/*'
    file_list = os.listdir(pathListdir)
    fileName_list = [file for file in file_list if file.endswith(Ext)]
    file_list = glob.glob(pathGlob)
    filePath_list = [file for file in file_list if file.endswith(Ext)]
    return (fileName_list, filePath_list)

def namePathLst(path, Ext):
    """Function imported from utils/utils.py - used by reproduction_02"""
    pathListdir = path  # path for the listdir package, e.g. 'C:./data'
    pathGlob = path+'/*'  # path for the glob package, e.g. 'C:./data/*'
    file_list = os.listdir(pathListdir)
    fileName_list = [file for file in file_list if file.endswith(Ext)]
    file_list = glob.glob(pathGlob)
    filePath_list = [file for file in file_list if file.endswith(Ext)]    
    return (fileName_list, filePath_list)

def makeMotionFilePathLst(motionFolderPath, motionExt, date=None):
    """Unified makeMotionFilePathLst - the date parameter is optional"""
    if date is not None:
        # utils/utils.py version - includes date filtering
        motionFilePathLst = namePathLst(motionFolderPath, motionExt)[1]
        motionFilePathLst = [i for i in motionFilePathLst if date in i]
        return motionFilePathLst
    else:
        # Original src/utils.py version - no date filtering
        motionFilePathLst = namePathList(motionFolderPath, motionExt)[1]
        return motionFilePathLst

# Parallel-processing progress visualization built for jupyter
def istarmap(self, func, iterable, chunksize=1):
    self._check_running()
    if chunksize < 1:
        raise ValueError(
            "Chunksize must be 1+, not {0:n}".format(
                chunksize))

    task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
    result = mpp.IMapIterator(self)
    self._taskqueue.put(
        (
            self._guarded_task_generation(result._job,
                                          mpp.starmapstar,
                                          task_batches),
            result._set_length
        ))
    return (item for chunk in result for item in chunk)


mpp.Pool.istarmap = istarmap





'''
Quaternion computations
'''

# Function to normalize a vector
def normalize(vector):
    norm = np.linalg.norm(vector)
    return vector / norm if norm != 0 else vector

# Function to compute quaternion from two vectors
def compute_quaternion(vec_u, vec_v):
    vec_u = normalize(vec_u)
    vec_v = normalize(vec_v)

    axis = np.cross(vec_u, vec_v)
    axis = normalize(axis)

    dot_product = np.dot(vec_u, vec_v)
    angle = np.arccos(np.clip(dot_product, -1.0, 1.0))

    w = np.cos(angle / 2)
    xyz = axis * np.sin(angle / 2)
    
    del vec_u, vec_v, axis, dot_product
    
    return np.array([w, xyz[0], xyz[1], xyz[2]])

# Function to compute three-joint sequences and quaternions for a single frame
def compute_three_joint_quaternions(row, connections, way="Right"):
    quaternions = {}
    for joint_a, joint_b in connections.items():
        if joint_a.startswith(way[0]) and joint_b.startswith(way[0]):
            joint_c = connections.get(joint_b, None)
            if joint_c and joint_c.startswith(way[0]):
                coords_a = np.array([row[f'{joint_a}_x'], row[f'{joint_a}_y'], row[f'{joint_a}_z']])
                coords_b = np.array([row[f'{joint_b}_x'], row[f'{joint_b}_y'], row[f'{joint_b}_z']])
                coords_c = np.array([row[f'{joint_c}_x'], row[f'{joint_c}_y'], row[f'{joint_c}_z']])

                vec_u = coords_b - coords_a
                vec_v = coords_c - coords_b

                quaternion = compute_quaternion(vec_u, vec_v)
                col_prefix = f"{joint_a}_to_{joint_b}_to_{joint_c}"
                quaternions[f"{col_prefix}_w"] = quaternion[0]
                quaternions[f"{col_prefix}_x"] = quaternion[1]
                quaternions[f"{col_prefix}_y"] = quaternion[2]
                quaternions[f"{col_prefix}_z"] = quaternion[3]
                
                del coords_a, coords_b, coords_c, vec_u, col_prefix
    return quaternions

# Function to compute distances between joints for a single frame
def compute_joint_distances(row, connections, way="Right"):
    distances = {}
    for joint_a, joint_b in connections.items():
        if joint_a.startswith(way[0]) and joint_b.startswith(way[0]):
            joint_c = connections.get(joint_b, None)
            if joint_c and joint_c.startswith(way[0]):
                coords_a = np.array([row[f'{joint_a}_x'], row[f'{joint_a}_y'], row[f'{joint_a}_z']])
                coords_b = np.array([row[f'{joint_b}_x'], row[f'{joint_b}_y'], row[f'{joint_b}_z']])
                coords_c = np.array([row[f'{joint_c}_x'], row[f'{joint_c}_y'], row[f'{joint_c}_z']])

                # Calculate distances (Euclidean distance)
                distance_ab = np.linalg.norm(coords_b - coords_a)
                distance_bc = np.linalg.norm(coords_c - coords_b)
                distances[f"{joint_a}_to_{joint_b}_distance"] = distance_ab
                distances[f"{joint_b}_to_{joint_c}_distance"] = distance_bc
                
                del coords_a, coords_b, coords_c, distance_ab, distance_bc
                
    return distances



def compute_three_joint_quaternions_body(row, connections):
    quaternions = {}
    for joint_a, joint_b in connections.items():
        # Find joint_c connected to joint_b
        joint_c = connections.get(joint_b, None)
        if joint_c is not None:
            coords_a = np.array([row[f'{joint_a}_x'], row[f'{joint_a}_y'], row[f'{joint_a}_z']])
            coords_b = np.array([row[f'{joint_b}_x'], row[f'{joint_b}_y'], row[f'{joint_b}_z']])
            coords_c = np.array([row[f'{joint_c}_x'], row[f'{joint_c}_y'], row[f'{joint_c}_z']])

            vec_u = coords_b - coords_a
            vec_v = coords_c - coords_b

            quaternion = compute_quaternion(vec_u, vec_v)
            col_prefix = f"{joint_a}_to_{joint_b}_to_{joint_c}"
            quaternions[f"{col_prefix}_w"] = quaternion[0]
            quaternions[f"{col_prefix}_x"] = quaternion[1]
            quaternions[f"{col_prefix}_y"] = quaternion[2]
            quaternions[f"{col_prefix}_z"] = quaternion[3]

            del coords_a, coords_b, coords_c, vec_u, vec_v, quaternion, col_prefix
    return quaternions

def compute_joint_distances_body(row, connections):
    distances = {}
    for joint_a, joint_b in connections.items():
        joint_c = connections.get(joint_b, None)
        if joint_c is not None:
            coords_a = np.array([row[f'{joint_a}_x'], row[f'{joint_a}_y'], row[f'{joint_a}_z']])
            coords_b = np.array([row[f'{joint_b}_x'], row[f'{joint_b}_y'], row[f'{joint_b}_z']])
            coords_c = np.array([row[f'{joint_c}_x'], row[f'{joint_c}_y'], row[f'{joint_c}_z']])

            distance_ab = np.linalg.norm(coords_b - coords_a)
            distance_bc = np.linalg.norm(coords_c - coords_b)

            distances[f"{joint_a}_to_{joint_b}_distance"] = distance_ab
            distances[f"{joint_b}_to_{joint_c}_distance"] = distance_bc

            del coords_a, coords_b, coords_c, distance_ab, distance_bc
    return distances
