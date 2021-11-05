

from tqdm import tqdm 
import mmcv 
import os
import os.path as osp
import numpy as np



datasets = {
    'xview', 'xsub'
}
# datasets = {'xview'}
sets = {
    'train', 'val'
}

max_body_true = 2
max_body_kinect = 4
num_joint = 25
max_frame = 300

for dataset in datasets:  # benchmark
    for set in sets: # part
        print('ing--', dataset, set)

        results = []

        # path = os.path.join('/mnt/lustre/liguankai/data/ntu/nturgb+d_skeletons_60_3d_nmtvc', dataset)
        path = os.path.join('/mnt/lustre/liguankai/data/ntu/nturgb+d_skeletons_60_3d_bone', dataset)
        path = '{}/{}.pkl'.format(path, set)

        data = mmcv.load(path)
        prog_bar = mmcv.ProgressBar(len(data))
        for i, item in enumerate(data):
            keypoint = item['keypoint'] # CTVM -> MTVC
            M, T, V, C = keypoint.shape

            motion = np.zeros((M, T, V, C), dtype=np.float32)
            for t in range(T-1):
                motion[:, t, :, :] = keypoint[:, t+1, :, :] - keypoint[:, t, :, :]
            motion[:, T-1, :, :] = 0
            item['keyoint'] = motion 

            results.append(item)
            prog_bar.update()

        out_path = os.path.join('/mnt/lustre/liguankai/data/ntu/nturgb+d_skeletons_60_3d_bone_motion', dataset)
        if not osp.exists(out_path):
            os.makedirs(out_path)
        output_path = '{}/{}.pkl'.format(out_path, set)
    
        mmcv.dump(results, output_path)
        print(f'{dataset}--{set} finish!!!!~')




