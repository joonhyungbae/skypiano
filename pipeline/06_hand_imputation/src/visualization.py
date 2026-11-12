import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Select only the necessary joints (remove non-existent columns)
right_joint_cols = ['RID_x', 'RID_y', 'RID_z', 'RIF_x', 'RIF_y', 'RIF_z', 
                    'RIM_x', 'RIM_y', 'RIM_z', 'RIP_x', 'RIP_y', 'RIP_z', 
                    'RLD_x', 'RLD_y', 'RLD_z', 'RLF_x', 'RLF_y', 'RLF_z', 
                    'RLM_x', 'RLM_y', 'RLM_z', 'RLP_x', 'RLP_y', 'RLP_z', 
                    'RMD_x', 'RMD_y', 'RMD_z', 'RMF_x', 'RMF_y', 'RMF_z', 
                    'RMM_x', 'RMM_y', 'RMM_z', 'RMP_x', 'RMP_y', 'RMP_z', 
                    'RRD_x', 'RRD_y', 'RRD_z', 'RRF_x', 'RRF_y', 'RRF_z', 
                    'RRM_x', 'RRM_y', 'RRM_z', 'RRP_x', 'RRP_y', 'RRP_z', 
                    'RTC_x', 'RTC_y', 'RTC_z', 'RTD_x', 'RTD_y', 'RTD_z', 
                    'RTF_x', 'RTF_y', 'RTF_z', 'RTM_x', 'RTM_y', 'RTM_z', 
                    'RElbow_x', 'RElbow_y', 'RElbow_z', 'RSh_x', 'RSh_y', 'RSh_z', 
                    'RWrist_x', 'RWrist_y', 'RWrist_z',
                    'REnd_x', 'REnd_y', 'REnd_z']

left_joint_cols = ['LID_x', 'LID_y', 'LID_z', 'LIF_x', 'LIF_y', 'LIF_z', 
                'LIM_x', 'LIM_y', 'LIM_z', 'LIP_x', 'LIP_y', 'LIP_z', 
                'LLD_x', 'LLD_y', 'LLD_z', 'LLF_x', 'LLF_y', 'LLF_z', 
                'LLM_x', 'LLM_y', 'LLM_z', 'LLP_x', 'LLP_y', 'LLP_z', 
                'LMD_x', 'LMD_y', 'LMD_z', 'LMF_x', 'LMF_y', 'LMF_z', 
                'LMM_x', 'LMM_y', 'LMM_z', 'LMP_x', 'LMP_y', 'LMP_z', 
                'LRD_x', 'LRD_y', 'LRD_z', 'LRF_x', 'LRF_y', 'LRF_z', 
                'LRM_x', 'LRM_y', 'LRM_z', 'LRP_x', 'LRP_y', 'LRP_z', 
                'LTC_x', 'LTC_y', 'LTC_z', 'LTD_x', 'LTD_y', 'LTD_z', 
                'LTF_x', 'LTF_y', 'LTF_z', 'LTM_x', 'LTM_y', 'LTM_z', 
                'LElbow_x', 'LElbow_y', 'LElbow_z', 'LSh_x', 'LSh_y', 'LSh_z', 
                'LWrist_x', 'LWrist_y', 'LWrist_z',
                    'LEnd_x', 'LEnd_y', 'LEnd_z']

all_joint_cols = ['RID_x', 'RID_y', 'RID_z', 'RIF_x', 'RIF_y', 'RIF_z', 
                    'RIM_x', 'RIM_y', 'RIM_z', 'RIP_x', 'RIP_y', 'RIP_z', 
                    'RLD_x', 'RLD_y', 'RLD_z', 'RLF_x', 'RLF_y', 'RLF_z', 
                    'RLM_x', 'RLM_y', 'RLM_z', 'RLP_x', 'RLP_y', 'RLP_z', 
                    'RMD_x', 'RMD_y', 'RMD_z', 'RMF_x', 'RMF_y', 'RMF_z', 
                    'RMM_x', 'RMM_y', 'RMM_z', 'RMP_x', 'RMP_y', 'RMP_z', 
                    'RRD_x', 'RRD_y', 'RRD_z', 'RRF_x', 'RRF_y', 'RRF_z', 
                    'RRM_x', 'RRM_y', 'RRM_z', 'RRP_x', 'RRP_y', 'RRP_z', 
                    'RTC_x', 'RTC_y', 'RTC_z', 'RTD_x', 'RTD_y', 'RTD_z', 
                    'RTF_x', 'RTF_y', 'RTF_z', 'RTM_x', 'RTM_y', 'RTM_z', 
                    'RElbow_x', 'RElbow_y', 'RElbow_z', 'RSh_x', 'RSh_y', 'RSh_z', 
                    'RWrist_x', 'RWrist_y', 'RWrist_z',
                    'REnd_x', 'REnd_y', 'REnd_z','LID_x', 'LID_y', 'LID_z', 'LIF_x', 'LIF_y', 'LIF_z', 
                'LIM_x', 'LIM_y', 'LIM_z', 'LIP_x', 'LIP_y', 'LIP_z', 
                'LLD_x', 'LLD_y', 'LLD_z', 'LLF_x', 'LLF_y', 'LLF_z', 
                'LLM_x', 'LLM_y', 'LLM_z', 'LLP_x', 'LLP_y', 'LLP_z', 
                'LMD_x', 'LMD_y', 'LMD_z', 'LMF_x', 'LMF_y', 'LMF_z', 
                'LMM_x', 'LMM_y', 'LMM_z', 'LMP_x', 'LMP_y', 'LMP_z', 
                'LRD_x', 'LRD_y', 'LRD_z', 'LRF_x', 'LRF_y', 'LRF_z', 
                'LRM_x', 'LRM_y', 'LRM_z', 'LRP_x', 'LRP_y', 'LRP_z', 
                'LTC_x', 'LTC_y', 'LTC_z', 'LTD_x', 'LTD_y', 'LTD_z', 
                'LTF_x', 'LTF_y', 'LTF_z', 'LTM_x', 'LTM_y', 'LTM_z', 
                'LElbow_x', 'LElbow_y', 'LElbow_z', 'LSh_x', 'LSh_y', 'LSh_z', 
                'LWrist_x', 'LWrist_y', 'LWrist_z',
                    'LEnd_x', 'LEnd_y', 'LEnd_z']

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

right_body_joint_cols = ['RWrist_x', 'RWrist_y', 'RWrist_z',
                         'RIM_x', 'RIM_y', 'RIM_z', 
                        'RLM_x', 'RLM_y', 'RLM_z', 
                        'RElbow_x', 'RElbow_y', 'RElbow_z',
                        'RSh_x', 'RSh_y', 'RSh_z']

left_body_joint_cols = ['LWrist_x', 'LWrist_y', 'LWrist_z',
                         'LIM_x', 'LIM_y', 'LIM_z', 
                        'LLM_x', 'LLM_y', 'LLM_z', 
                        'LElbow_x', 'LElbow_y', 'LElbow_z',
                        'LSh_x', 'LSh_y', 'LSh_z']

body_joint_cols =['RWrist_x', 'RWrist_y', 'RWrist_z',
                'RIM_x', 'RIM_y', 'RIM_z', 
                'RLM_x', 'RLM_y', 'RLM_z', 
                'RElbow_x', 'RElbow_y', 'RElbow_z',
                'RSh_x', 'RSh_y', 'RSh_z',
                'LWrist_x', 'LWrist_y', 'LWrist_z',
                'LIM_x', 'LIM_y', 'LIM_z', 
                'LLM_x', 'LLM_y', 'LLM_z', 
                'LElbow_x', 'LElbow_y', 'LElbow_z',
                'LSh_x', 'LSh_y', 'LSh_z']

right_body_connections = {
    'RIM': 'RWrist',
    'RLM': 'RWrist',
    'RWrist': 'RElbow', 'RElbow': 'RSh',
}

left_body_connections = {
    'LIM': 'LWrist',
    'LLM': 'LWrist',
    'LWrist': 'LElbow', 'LElbow': 'LSh',
}

body_connections_dict={'RIM': 'RWrist',
                'RLM': 'RWrist',
                'RWrist': 'RElbow', 'RElbow': 'RSh',
                'LIM': 'LWrist',
                'LLM': 'LWrist',
                'LWrist': 'LElbow', 'LElbow': 'LSh',}

# Anatomical order: M (MCP) → P (PIP) → D (DIP) → F (tip) per finger.
# Thumb: TC (CMC) → TM (MCP) → TD (IP) → TF (tip).
# Finger iteration order matches s07_slice_hands.hand_joint_cols(): I, L, M, R, T.
# MUST match s07's column layout — mismatch causes silent marker-label swapping.
right_hand_joint_cols = ['RWrist_x', 'RWrist_y', 'RWrist_z',
                    'RIM_x', 'RIM_y', 'RIM_z', 'RIP_x', 'RIP_y', 'RIP_z',
                    'RID_x', 'RID_y', 'RID_z', 'RIF_x', 'RIF_y', 'RIF_z',
                    'RLM_x', 'RLM_y', 'RLM_z', 'RLP_x', 'RLP_y', 'RLP_z',
                    'RLD_x', 'RLD_y', 'RLD_z', 'RLF_x', 'RLF_y', 'RLF_z',
                    'RMM_x', 'RMM_y', 'RMM_z', 'RMP_x', 'RMP_y', 'RMP_z',
                    'RMD_x', 'RMD_y', 'RMD_z', 'RMF_x', 'RMF_y', 'RMF_z',
                    'RRM_x', 'RRM_y', 'RRM_z', 'RRP_x', 'RRP_y', 'RRP_z',
                    'RRD_x', 'RRD_y', 'RRD_z', 'RRF_x', 'RRF_y', 'RRF_z',
                    'RTC_x', 'RTC_y', 'RTC_z', 'RTM_x', 'RTM_y', 'RTM_z',
                    'RTD_x', 'RTD_y', 'RTD_z', 'RTF_x', 'RTF_y', 'RTF_z']

left_hand_joint_cols = ['LWrist_x', 'LWrist_y', 'LWrist_z',
                'LIM_x', 'LIM_y', 'LIM_z', 'LIP_x', 'LIP_y', 'LIP_z',
                'LID_x', 'LID_y', 'LID_z', 'LIF_x', 'LIF_y', 'LIF_z',
                'LLM_x', 'LLM_y', 'LLM_z', 'LLP_x', 'LLP_y', 'LLP_z',
                'LLD_x', 'LLD_y', 'LLD_z', 'LLF_x', 'LLF_y', 'LLF_z',
                'LMM_x', 'LMM_y', 'LMM_z', 'LMP_x', 'LMP_y', 'LMP_z',
                'LMD_x', 'LMD_y', 'LMD_z', 'LMF_x', 'LMF_y', 'LMF_z',
                'LRM_x', 'LRM_y', 'LRM_z', 'LRP_x', 'LRP_y', 'LRP_z',
                'LRD_x', 'LRD_y', 'LRD_z', 'LRF_x', 'LRF_y', 'LRF_z',
                'LTC_x', 'LTC_y', 'LTC_z', 'LTM_x', 'LTM_y', 'LTM_z',
                'LTD_x', 'LTD_y', 'LTD_z', 'LTF_x', 'LTF_y', 'LTF_z']

# Define connections for the right hand only
right_hand_connections = {
    # Right thumb
    'RTF': 'RTD', 'RTD': 'RTM', 'RTM': 'RTC', 'RTC': 'RWrist',
    # Right index finger
    'RIF': 'RID', 'RID': 'RIP', 'RIP': 'RIM', 'RIM': 'RWrist',
    # Right middle finger
    'RMF': 'RMD', 'RMD': 'RMP', 'RMP': 'RMM', 'RMM': 'RWrist',
    # Right ring finger
    'RRF': 'RRD', 'RRD': 'RRP', 'RRP': 'RRM', 'RRM': 'RWrist',
    # Right little finger
    'RLF': 'RLD', 'RLD': 'RLP', 'RLP': 'RLM', 'RLM': 'RWrist',
}

left_hand_connections = {
    # Right thumb
    'LTF': 'LTD', 'LTD': 'LTM', 'LTM': 'LTC', 'LTC': 'LWrist',
    # Right index finger
    'LIF': 'LID', 'LID': 'LIP', 'LIP': 'LIM', 'LIM': 'LWrist',
    # Right middle finger
    'LMF': 'LMD', 'LMD': 'LMP', 'LMP': 'LMM', 'LMM': 'LWrist',
    # Right ring finger
    'LRF': 'LRD', 'LRD': 'LRP', 'LRP': 'LRM', 'LRM': 'LWrist',
    # Right little finger
    'LLF': 'LLD', 'LLD': 'LLP', 'LLP': 'LLM', 'LLM': 'LWrist',
}


# Define joint connections
connections = {
    # Right hand thumb
    'RTF': 'RTD', 'RTD': 'RTM','RTM': 'RTC', 'RTC': 'RWrist',
    # Right hand index finger
    'RIF': 'RID', 'RID': 'RIP', 'RIP': 'RIM', 'RIM': 'RWrist',
    # Right hand middle finger
    'RMF': 'RMD', 'RMD': 'RMP', 'RMP': 'RMM', 'RMM': 'RWrist',
    # Right hand ring finger
    'RRF': 'RRD', 'RRD': 'RRP', 'RRP': 'RRM', 'RRM': 'RWrist',
    # Right hand pinky finger
    'RLF': 'RLD', 'RLD': 'RLP', 'RLP': 'RLM', 'RLM': 'RWrist',
    # Right arm
    'RWrist': 'RElbow', 'RElbow': 'RSh',
    # Left hand thumb
    'LTF': 'LTD', 'LTD': 'LTM','LTM': 'LTC', 'LTC': 'LWrist',
    # Left hand index finger
    'LIF': 'LID', 'LID': 'LIP', 'LIP': 'LIM', 'LIM': 'LWrist',
    # Left hand middle finger
    'LMF': 'LMD', 'LMD': 'LMP', 'LMP': 'LMM', 'LMM': 'LWrist',
    # Left hand ring finger
    'LRF': 'LRD', 'LRD': 'LRP', 'LRP': 'LRM', 'LRM': 'LWrist',
    # Left hand pinky finger
    'LLF': 'LLD', 'LLD': 'LLP', 'LLP': 'LLM', 'LLM': 'LWrist',
    # Left arm
    'LWrist': 'LElbow', 'LElbow': 'LSh'
}


