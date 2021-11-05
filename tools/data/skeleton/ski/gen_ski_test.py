import numpy as np
import mmcv

# output_pkl = '/mnt/lustre/liguankai/data/ski/test.pkl'
output_pkl = '/mnt/lustre/liguankai/data/ski/2500_422/test.pkl'

test_data = '/mnt/lustre/liguankai/data/ski/test_A_data.npy'


data = np.load(test_data)  
n_samples = len(data)   # 628 
print(n_samples)

results = []
prog_bar = mmcv.ProgressBar(n_samples)

for i, keypoint in enumerate(data):
    anno = dict()
    anno['total_frames'] = 2500
    anno['keypoint'] = keypoint.transpose(3, 1, 2, 0)  # C T V M -> M T V C
    anno['img_shape'] = (1080, 720)
    anno['original_shape'] = (1080, 720)
    anno['label'] = 0
    # print(int(label[i]))
    results.append(anno)
    prog_bar.update()

mmcv.dump(results, output_pkl)
print('Finish!')
    

