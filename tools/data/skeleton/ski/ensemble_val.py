import mmcv 
import numpy as np
from numpy.lib.function_base import percentile 

alpha=1

# test_data = '/mnt/lustre/liguankai/data/ski/test_A_data.npy'
# data = np.load(test_data)  
# n_samples = len(data)   # 628 
# print(n_samples)

test_output_joint = '/mnt/lustre/liguankai/codebase/mmaction2/preds/2500_422/no_padding/joint_2s_val.pkl'
test_output_bone = '/mnt/lustre/liguankai/codebase/mmaction2/preds/2500_422/no_padding/bone_2s_val.pkl' 
test_output_joint_mo = '/mnt/lustre/liguankai/codebase/mmaction2/preds/2500_422/motion_xy_val.pkl'
test_output_bone_mo = '/mnt/lustre/liguankai/codebase/mmaction2/preds/2500_422/bone_xy_val.pkl'  # better than ski_bone_test.pkl

# test_output_joint_motion = '/mnt/lustre/liguankai/codebase/mmaction2/aagcn_motion_joint.pkl'
# test_output_bone_motion = '/mnt/lustre/liguankai/codebase/mmaction2/aagcn_motion_bone.pkl'

preds_joint = mmcv.load(test_output_joint)
preds_bone = mmcv.load(test_output_bone)
preds_joint_mo = mmcv.load(test_output_joint_mo)
preds_bone_mo = mmcv.load(test_output_bone_mo)
# preds_joint_motion = mmcv.load(test_output_joint_motion)
# preds_bone_motion = mmcv.load(test_output_bone_motion)
# n_preds = len(preds_joint)
print(len(preds_joint), len(preds_bone))  # 422
print(len(preds_joint_mo), len(preds_bone_mo))

# ann_file_val = '/mnt/lustre/liguankai/data/ski/bone_xy/val.pkl'
ann_file_val = '/mnt/lustre/liguankai/data/ski/2500_422/padding_sub/bone_xy/val.pkl'
labels = mmcv.load(ann_file_val)

print('len(labels)--', len(labels))

right_num = total_num = right_num_5 = 0

for i in range(len(labels)):
    label = labels[i]['label']
    pred = preds_joint[i] + preds_bone[i]
    # pred = preds_joint[i] + preds_bone[i] * alpha + preds_joint_mo[i] + preds_bone_mo[i]
    # pred = preds_joint[i] + preds_bone[i] * alpha + preds_joint_mo[i]
    cate = np.argmax(pred)
    right_num += int(int(label)==cate)
    total_num += 1 

acc = right_num / total_num 
print(f'alpha={alpha}, acc = {acc}')
ss


# 2500/422 

'''                                 no_padding
joint + bone                           0.679
'''



'''                                 padding_sub (train)    padding_sub (train+val)   
joint + bone                            0.6185                 0.8957
joint_motion + bone                     0.6137                 0.9171 !
joint_motion + bone_motion              0.6398  !!             0.7891
joint + bone_motion                     0.6374                 0.7607
all                                     0.6303                 0.8791
joint + bone + bone_motion              0.6398   !!            0.8839
joint + bone + joint_motion             0.6280                 0.8981
'''



# 2722/200
'''                                  no norm     padding
joint + bone                          0.68         0.725
joint_motion + bone                   0.71         0.695
joint_motion + bone_motion            0.69         0.725
joint + bone_motion                   0.71         0.745
all                                   0.705        0.74
joint + bone + bone_motion            0.75         0.755
joint + bone + joint_motion           0.69         0.725
'''




