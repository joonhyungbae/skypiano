from src.performers import PERFORMER_MAP, performer_id
# Merge restored left and right hand columns back into the body CSV to produce the final per-performer trial files.

# Import required libraries
import os
import pandas as pd
from tqdm import tqdm
from src.utils import *
from src.visualization import *

sec = 3
step = 1

motionFilePathLst=[]

motionExt = '.csv'

for (c_val, n_val), trials in sorted(PERFORMER_MAP.items()):
    pid = performer_id(c_val, n_val)

    src_dir = f'Data/Pipeline/06_hand_imputation/Imp/replace/Body/{pid}'
    if not os.path.isdir(src_dir):
        print(f"Skip {pid}: no Imp/replace/Body/ data")
        continue
    filePathLst = makeMotionFilePathLst(src_dir, motionExt)
    motionFilePathLst=motionFilePathLst+filePathLst

for filePath in tqdm(motionFilePathLst):

    norm = filePath.replace('\\', '/')
    parts = norm.split('/')
    pid = parts[-2]
    filename = parts[-1]

    data=pd.read_csv(filePath)

    '''
    colLst=list(data.columns)

    if 'LTC_x' in colLst:
        pass
    else:
        for col in ['LTC_x','LTC_y','LTC_z','RTC_x','RTC_y','RTC_z']:
            data[col] = np.nan
    '''

    rightPath=f'Data/Pipeline/06_hand_imputation/Imp/restore/Right/{pid}/{filename}'
    leftPath=f'Data/Pipeline/06_hand_imputation/Imp/restore/Left/{pid}/{filename}'

    if not (os.path.exists(rightPath) and os.path.exists(leftPath)):
        print(f"Skip (no restore data): {pid}/{filename}")
        continue

    rightData=pd.read_csv(rightPath)
    leftData=pd.read_csv(leftPath)

    right_colLst=list(rightData.columns)
    left_colLst=list(leftData.columns)

    for col in right_colLst:
        data[col]=rightData[col]

    for col in left_colLst:
        data[col]=leftData[col]
        
    os.makedirs(f'Data/Pipeline/06_hand_imputation/Final/{pid}',exist_ok=True)
    savePath=f'Data/Pipeline/06_hand_imputation/Final/{pid}/{filename}'

    data.to_csv(savePath,index=False) 