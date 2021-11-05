import mmcv
import numpy as np
import random
import math 


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    if np.abs(v1).sum() < 1e-6 or np.abs(v2).sum() < 1e-6:
        return 0
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def rotation_matrix(axis, theta):
    """Return the rotation matrix associated with counterclockwise rotation
    about the given axis by theta radians."""
    if np.abs(axis).sum() < 1e-6 or np.abs(theta) < 1e-6:
        return np.eye(3)
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

# def pre_normalization(data, zaxis=[0, 1], xaxis=[8, 4]):
def pre_normalization(data, zaxis=[8, 1], xaxis=[2, 5]):
    N, C, T, V, M = data.shape
    print('data-shape', data.shape) # 2922 3 2500 25 1 
    s = np.transpose(data, [0, 4, 2, 3, 1])  # N C T V M -> N M T V C

    print('pad the null frames with the previous frames')
    prog_bar = mmcv.ProgressBar(len(s))
    for i_s, skeleton in enumerate(s):
        if skeleton.sum() == 0:
            print(i_s, ' has no skeleton')
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            if person[0].sum() == 0:
                index = (person.sum(-1).sum(-1) != 0)
                tmp = person[index].copy()
                person *= 0
                person[:len(tmp)] = tmp

            for i_f, frame in enumerate(person):
                if frame.sum() == 0:
                    if person[i_f:].sum() == 0:
                        rest = len(person) - i_f
                        num = int(np.ceil(rest / i_f))
                        pad = np.concatenate(
                            [person[0:i_f] for _ in range(num)], 0)[:rest]
                        s[i_s, i_p, i_f:] = pad
                        break
        prog_bar.update()

    print('sub the center joint #1 (spine joint in ntu and '
          'neck joint in kinetics)')
    prog_bar = mmcv.ProgressBar(len(s))
    for i_s, skeleton in enumerate(s):
        if skeleton.sum() == 0:
            continue
        main_body_center = skeleton[0][:, 1:2, :].copy() # 1 neck in ski
        # main_body_center = skeleton[0][:, 8:9. :].copy() # 8 spine point in ski
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            mask = (person.sum(-1) != 0).reshape(T, V, 1)
            s[i_s, i_p] = (s[i_s, i_p] - main_body_center) * mask
        prog_bar.update()

    # print('parallel the bone between hip(jpt 0) and '
    #       'spine(jpt 1) of the first person to the z axis') # hip 8 spine 1 
    # prog_bar = mmcv.ProgressBar(len(s)) 
    # for i_s, skeleton in enumerate(s): # N M T V C
    #     if skeleton.sum() == 0:
    #         continue
    #     joint_bottom = skeleton[0, 0, zaxis[0]]
    #     joint_top = skeleton[0, 0, zaxis[1]]
    #     axis = np.cross(joint_top - joint_bottom, [0, 0, 1])
    #     angle = angle_between(joint_top - joint_bottom, [0, 0, 1])
    #     matrix_z = rotation_matrix(axis, angle)
    #     for i_p, person in enumerate(skeleton):
    #         if person.sum() == 0:
    #             continue
    #         for i_f, frame in enumerate(person):
    #             if frame.sum() == 0:
    #                 continue
    #             for i_j, joint in enumerate(frame):
    #                 s[i_s, i_p, i_f, i_j] = np.dot(matrix_z, joint)
    #     prog_bar.update()

    print('parallel the bone between right shoulder(jpt 8) and '
          'left shoulder(jpt 4) of the first person to the x axis') # 2 right shoulder 5 left shoulder in ski
    prog_bar = mmcv.ProgressBar(len(s))
    for i_s, skeleton in enumerate(s):
        if skeleton.sum() == 0:
            continue
        joint_rshoulder = skeleton[0, 0, xaxis[0]]
        joint_lshoulder = skeleton[0, 0, xaxis[1]]
        axis = np.cross(joint_rshoulder - joint_lshoulder, [1, 0, 0])
        angle = angle_between(joint_rshoulder - joint_lshoulder, [1, 0, 0])
        matrix_x = rotation_matrix(axis, angle)
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            for i_f, frame in enumerate(person):
                if frame.sum() == 0:
                    continue
                for i_j, joint in enumerate(frame):
                    s[i_s, i_p, i_f, i_j] = np.dot(matrix_x, joint)
        prog_bar.update()

    data = np.transpose(s, [0, 4, 2, 3, 1])  # N M T V C -> N C T V M
    return data





data = '/mnt/lustre/liguankai/data/ski/test_A_data.npy'


data = np.load(data)
print('len-data', len(data))  # 628



    
results = []
prog_bar = mmcv.ProgressBar(len(data))
for i, keypoint in enumerate(data):
    anno = dict()
    anno['total_frames'] = 2500
    # anno['keypoint'] = keypoint.transpose(3, 1, 2, 0)  # C T V M -> M T V C
    anno['keypoint'] = keypoint # C T V M 
    anno['img_shape'] = (1080, 720)
    anno['original_shape'] = (1080, 720)
    anno['label'] = 0
    # print(int(label[i]))
    results.append(anno)
    prog_bar.update()
    

max_frame=2500
num_joint = 25
max_body_true = 1
print('Preforming pre_normalization \n') 
fp = np.zeros((len(results), 3, max_frame, num_joint, max_body_true), dtype=np.float32) 
for i, anno in enumerate(results):
    # fp.append(anno['keypoint']) # C T V M 
    fp[i, :, :, :, :] = anno['keypoint'] # C T V M 

fp = pre_normalization(fp)

for i, anno in enumerate(results):
    anno['keypoint'] = fp[i].transpose(3, 1, 2, 0)  # C T V M -> M T V C


output_pkl = '/mnt/lustre/liguankai/data/ski/norm124/test_norm.pkl'
mmcv.dump(results, output_pkl)

print('Finish!')