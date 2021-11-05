import mmcv 
import numpy as np
from numpy.lib.function_base import percentile 



test_data = '/mnt/lustre/liguankai/data/ski/test_A_data.npy'
data = np.load(test_data)  
n_samples = len(data)   # 628 
print(n_samples)

test_output_joint = '/mnt/lustre/liguankai/codebase/mmaction2/preds/1104pad/test_joint.pkl'
test_output_bone = '/mnt/lustre/liguankai/codebase/mmaction2/preds/1104pad/test_bone_4gpu.pkl' 
test_output_joint_mo = '/mnt/lustre/liguankai/codebase/mmaction2/preds/1104pad/test_motion_xy.pkl'
test_output_bone_xy = '/mnt/lustre/liguankai/codebase/mmaction2/preds/1104pad/test_bone_xy.pkl'# better than ski_bone_test.pkl


# test_output_joint_motion = '/mnt/lustre/liguankai/codebase/mmaction2/aagcn_motion_joint.pkl'
# test_output_bone_motion = '/mnt/lustre/liguankai/codebase/mmaction2/aagcn_motion_bone.pkl'

preds_joint = mmcv.load(test_output_joint)
preds_bone = mmcv.load(test_output_bone)
preds_joint_mo = mmcv.load(test_output_joint_mo)
preds_bone_xy = mmcv.load(test_output_bone_xy)
# n_preds = len(preds_joint)
print(len(preds_joint), len(preds_bone))  # 628
print(len(preds_joint_mo), len(preds_bone_xy)) # 628



output_file = 'submission.csv'

import csv

alpha=1


values = []

for i in range(len(preds_joint)):
    pred = preds_joint_mo[i] * alpha + preds_bone[i] * alpha + preds_joint_mo[i] * alpha
    # pred = preds_joint[i] + preds_bone[i] * alpha + preds_joint_mo[i] * alpha + preds_bone_xy[i] * alpha
    # pred = preds_joint[i]  + preds_joint_mo[i] * alpha  + preds_bone_xy[i] * alpha
    cate = np.argmax(pred)
    values.append((i, cate))

header=['sample_index','predict_category']
with open(output_file, 'w') as fp:
    writer = csv.writer(fp)
    writer.writerow(header)
    writer.writerows(values)

print('Finish~')


'''                                                       no norm               padding
joint + bone                                                65.4 
joint_motion + bone                                         66.4 
joint + bone_xy                                             67.5                  66.5
joint_motion + bone_xy                                      67.68                 66.7
joint + joint_motion + bone + bone_xy                       69.27   !!!!          68
joint + bone + bone_xy                                      67.03                 67.03
joint + bone + joint_motion                                 68.47                 69.9  !!!!
joint + joint_motion + bone_xy                              68.95                 68
'''