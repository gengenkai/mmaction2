import mmcv
import numpy as np
import os



sets = ['train', 'val', 'test']

# generate joint/bone motion

for set in sets:

    results = []
    # path = '{}/{}.pkl'.format('/mnt/lustre/liguankai/data/ski', set) 
    path = '{}/{}.pkl'.format('/mnt/lustre/liguankai/data/ski/2500_422/padding_sub', set) 
    data = mmcv.load(path)
    print('len(data)', len(data)) # 2722

    prog_bar = mmcv.ProgressBar(len(data))
    for i, item in enumerate(data):
        keypoint = item['keypoint'] # CTVM -> MTVC
        M, T, V, C = keypoint.shape
        motion = np.zeros((M, T, V, C), dtype=np.float32)

        # for t in range(T-1):
        #     motion[:, t, :, :] = keypoint[:, t+1, :, :] - keypoint[:, t, :, :]
        for t in range(T-1):
            motion[:, t, :, :2] = keypoint[:, t+1, :, :2] - keypoint[:, t, :, :2]
            motion[:, t, :, 2] = (keypoint[:, t+1, :, 2] + keypoint[:, t, :, 2]) / 2

        motion[:, T-1, :, :] = 0
        item['keypoint'] =  motion 
        results.append(item)
        prog_bar.update()
    
    # out_path = '/mnt/lustre/liguankai/data/ski/motion_xy'
    out_path = '/mnt/lustre/liguankai/data/ski/2500_422/padding_sub/motion_xy'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    out_path = '{}/{}.pkl'.format(out_path, set)
    mmcv.dump(results, out_path)
    print(f'{out_path} finish!!!!~\n')

