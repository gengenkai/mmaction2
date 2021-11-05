import mmcv 
import numpy as np
from numpy.lib.function_base import percentile 

alpha=1

# test_data = '/mnt/lustre/liguankai/data/ski/test_A_data.npy'
# data = np.load(test_data)  
# n_samples = len(data)   # 628 
# print(n_samples)

test_output_joint = '/mnt/lustre/liguankai/codebase/mmaction2/preds/1104pad/val_joint.pkl'
test_output_bone = '/mnt/lustre/liguankai/codebase/mmaction2/preds/1104pad/val_bone_4gpu.pkl' 
test_output_joint_mo = '/mnt/lustre/liguankai/codebase/mmaction2/preds/1104pad/val_motion_xy.pkl'
test_output_bone_mo = '/mnt/lustre/liguankai/codebase/mmaction2/preds/1104pad/val_bone_xy.pkl' # better than ski_bone_test.pkl

# test_output_joint_motion = '/mnt/lustre/liguankai/codebase/mmaction2/aagcn_motion_joint.pkl'
# test_output_bone_motion = '/mnt/lustre/liguankai/codebase/mmaction2/aagcn_motion_bone.pkl'

preds_joint = mmcv.load(test_output_joint)
preds_bone = mmcv.load(test_output_bone)
preds_joint_mo = mmcv.load(test_output_joint_mo)
preds_bone_mo = mmcv.load(test_output_bone_mo)
# preds_joint_motion = mmcv.load(test_output_joint_motion)
# preds_bone_motion = mmcv.load(test_output_bone_motion)
# n_preds = len(preds_joint)
print(len(preds_joint), len(preds_bone))  # 200 
print(len(preds_joint_mo), len(preds_bone_mo))

ann_file_val = '/mnt/lustre/liguankai/data/ski/bone_xy/val.pkl'
labels = mmcv.load(ann_file_val)

print('len(labels)--', len(labels))

right_num = total_num = right_num_5 = 0

for i in range(len(labels)):
    label = labels[i]['label']
    pred = preds_joint[i] + preds_bone[i] + preds_joint_mo[i]
    # pred = preds_joint[i] + preds_bone[i] * alpha + preds_joint_mo[i] + preds_bone_mo[i]
    # pred = preds_joint[i] + preds_bone[i] * alpha + preds_joint_mo[i]
    cate = np.argmax(pred)
    right_num += int(int(label)==cate)
    total_num += 1 

acc = right_num / total_num 
print(f'alpha={alpha}, acc = {acc}')
ss



# padding 
'''                                  no norm     padding
joint + bone                          0.68         0.725
joint_motion + bone                   0.71         0.695
joint_motion + bone_motion            0.69         0.725
joint + bone_motion                   0.71         0.745
all                                   0.705        0.74
joint + bone + bone_motion            0.75         0.755
joint + bone + joint_motion           0.69         0.725
'''


# no norm 
'''
joint + bone                          0.68 
joint_motion + bone                   0.71
joint_motion + bone_motion            0.69
joint + bone_motion                   0.71
all                                   0.705
joint + bone + bone_motion            0.75
joint + bone + joint_motion           0.69
'''


