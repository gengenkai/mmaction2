import mmcv 
import numpy as np 
import os

sets = ['train', 'val', 'test']

for set in sets:
    path = '{}/{}.pkl'.format('/mnt/lustre/liguankai/data/ski/2500_422', set)
    data = mmcv.load(path) 
    print('len(data)--', len(data))

    results = []

    prog_bar = mmcv.ProgressBar(len(data))
    for i, anno in enumerate(data):

        # print('anno-shape', anno['keypoint'].shape) # M T V C  (1, 2500, 25, 3)
    
        keypoint = anno['keypoint'] # M T V C 
        M, T, V, C = keypoint.shape
        # print(keypoint[0, 400:500, :])
        total_frames = anno['total_frames']
        
        for i_p, person in enumerate(keypoint): # keypoint M T V C 
            if person[0].sum() == 0: # person T V C 
                index = (person.sum(-1).sum(-1) != 0)
                print(f'{i_p} person frame  0 is zero, non_zero indexes are {index}') # 有3个sample totally(train val test)
                tmp = person[index].copy()
                person *= 0
                person[:len(tmp)] = tmp
            
            for i_f, frame in enumerate(person):  # frame  V C 
                if frame.sum() == 0:
                    if person[i_f:].sum() == 0:
                        print(f'start from {i_f} is zero')
                        total_frames = i_f + 1
                        # rest = len(person) - i_f 
                        # num =  int(np.ceil(rest / i_f))
                        # pad = np.concatenate(
                        #         [person[0:i_f] for _ in range(num)], 0)[:rest]
                        # keypoint[i_p, i_f:] = pad
                        break 
        
        # sub the center joint spine joint (1)
        for v in range(V):
            keypoint[:, :, v, :2] = keypoint[:, :, v, :2] - keypoint[:, :, 1, :2] # xy coordinate
        
        anno['keypoint'] = keypoint[:, :total_frames, :, :]
        anno['total_frames'] = total_frames
        results.append(anno)
        prog_bar.update()
    
        

    out_dir = '/mnt/lustre/liguankai/data/ski/2500_422/no_padding'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_path = '{}/{}.pkl'.format(out_dir, set)
    mmcv.dump(results, out_path)
    print(f'{set} finish no padding and save real total_frames!!!!~')


        