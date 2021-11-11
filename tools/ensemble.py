import numpy as np
import mmcv



joint_pkl = '2s_joint_8682.pkl'
bone_pkl = '2s_bone_8784.pkl'
joint_motion_pkl = '2s_motion_8613.pkl'
bone_motion_pkl = '2s_motion_bone_8720.pkl'

a=1
b=1
c=0
d=0

alpha = 1

joint_scores = mmcv.load(joint_pkl)
joint_scores = list(joint_scores)
print('len(joints)--', len(joint_scores))  # 16487

bone_scores = mmcv.load(bone_pkl)
bone_scores = list(bone_scores)
print('len(bones)--', len(bone_scores))  # 16487

joint_mo_scores = mmcv.load(joint_motion_pkl)
joint_mo_scores = list(joint_mo_scores)
print('len(joints_mo)--', len(joint_mo_scores))  # 16487

bone_mo_scores = mmcv.load(bone_motion_pkl)
bone_mo_scores = list(bone_mo_scores)
print('len(bones_mo)--', len(bone_mo_scores))  # 16487


label_pkl = '/mnt/lustre/liguankai/data/ntu/nturgb+d_skeletons_60_3d_nmtvc/xsub/val.pkl'
labels = mmcv.load(label_pkl)
l_label = len(labels)
print('l_label', l_label) # 16487

right_num = total_num = right_num_5 = 0

for i in range(l_label):
    label = labels[i]['label']
    # joint_s = joint_scores[i]
    # bone_s = bone_scores[i]
    # score = joint_s + bone_s * alpha
    score = joint_scores[i] * a + bone_scores[i] * b + joint_mo_scores[i] * c + bone_mo_scores[i] * d
    rank_5 = score.argsort()[-5:]
    right_num_5 += int(int(label) in rank_5)
    rank_1 = np.argmax(score)
    right_num += int(int(label)==rank_1)
    total_num += 1 

acc = right_num / total_num 
acc5 = right_num_5 / total_num 
# print('alpha=',alpha, acc, acc5)
print(a, b, c, d, acc, acc5 )


'''  ensemble
1   1   1    1        0.9021653423909747 0.9824710377873476
1   1  1.5  1.5       0.9015588039061079 0.9830775762722145
1  1.1  1  1.1        0.9021653423909747 0.9824103839388609
1   1   0    1        0.8996178807545339 0.981803845453994
1   1   1    0        0.8979802268453934 0.9823497300903742
0   0   1    1        0.8910656881179111 0.9807120761812337
1   1   0    0        0.894765572875599  0.9803481530903135

'''


'''
alpha= 0.8  0.8936131497543519 0.9802874992418269
alpha= 0.9  0.8945836113301389 0.9799842299993935
alpha= 1    0.894765572875599 0.9803481530903135
alpha= 1.05 0.8945229574816522 0.9802268453933403
alpha= 1.1  0.894947534421059 0.9799235761509067  !!!!
alpha= 1.15 0.8948262267240856 0.9799235761509067
alpha= 1.2  0.8948262267240856 0.9801055376963669

'''







 
