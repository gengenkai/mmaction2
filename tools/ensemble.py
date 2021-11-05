import numpy as np
import mmcv



joint_pkl = '2s_joint_8682.pkl'
bone_pkl = '2s_bone_8784.pkl'

alpha = 1.05

joint_scores = mmcv.load(joint_pkl)
joint_scores = list(joint_scores)
print('len(joints)--', len(joint_scores))  # 16487

bone_scores = mmcv.load(bone_pkl)
bone_scores = list(bone_scores)
print('len(bones)--', len(bone_scores))  # 16487

label_pkl = '/mnt/lustre/liguankai/data/ntu/nturgb+d_skeletons_60_3d_nmtvc/xsub/val.pkl'
labels = mmcv.load(label_pkl)
l_label = len(labels)
print('l_label', l_label) # 16487

right_num = total_num = right_num_5 = 0

for i in range(l_label):
    label = labels[i]['label']
    joint_s = joint_scores[i]
    bone_s = bone_scores[i]
    score = joint_s + bone_s * alpha 
    rank_5 = score.argsort()[-5:]
    right_num_5 += int(int(label) in rank_5)
    rank_1 = np.argmax(score)
    right_num += int(int(label)==rank_1)
    total_num += 1 

acc = right_num / total_num 
acc5 = right_num_5 / total_num 
print('alpha=',alpha, acc, acc5)


'''
alpha= 0.8  0.8936131497543519 0.9802874992418269
alpha= 0.9  0.8945836113301389 0.9799842299993935
alpha= 1    0.894765572875599 0.9803481530903135
alpha= 1.05 0.8945229574816522 0.9802268453933403
alpha= 1.1  0.894947534421059 0.9799235761509067  !!!!
alpha= 1.15 0.8948262267240856 0.9799235761509067
alpha= 1.2  0.8948262267240856 0.9801055376963669

'''



'''
alpha= 0.8  0.8888214957239037 0.9794383453630133
alpha= 0.9  0.8891247649663371 0.9794989992115
alpha= 0.95 0.8896099957542306 0.97931703766604
alpha= 1.0  0.889731303451204 0.9793776915145266  !!!
alpha= 1.05 0.8894280342087706 0.97931703766604
alpha= 1.1  0.8891854188148238 0.9792563838175532
alpha= 1.15 0.8893673803602838 0.9790137684236064


'''





'''
alpha= 0.7  0.8825134954812883 0.9790137684236064
alpha= 0.8  0.8823921877843149 0.9791350761205798
alpha= 0.85 0.8829987262691817 0.9791350761205798  !!!
alpha= 0.9  0.8828774185722084 0.97931703766604 
alpha= 1    0.8826954570267483 0.9790744222720932
alpha= 1.05 0.8826348031782617 0.9790744222720932
alpha= 1.1  0.8823921877843149 0.9790137684236064

'''
 
