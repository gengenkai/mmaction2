import mmcv
import numpy as np
import random

random.seed(0)

data = '/mnt/lustre/liguankai/data/ski/train_data.npy'
label = '/mnt/lustre/liguankai/data/ski/train_label.npy'

label  = np.load(label)
print('len-label', len(label)) # 2922
data = np.load(data)
print('len-data', len(data))  # 2922

output_train_pkl = '/mnt/lustre/liguankai/data/ski/posec3d/train.pkl'
output_val_pkl = '/mnt/lustre/liguankai/data/ski/posec3d/val.pkl'

results = []
prog_bar = mmcv.ProgressBar(len(data))
lower_bound = -1.0
upper_bound = 1.0
size = 100
for i, keypoint in enumerate(data):
    anno = dict()
    anno['total_frames'] = 2500
    keypoint = keypoint.transpose(3, 1, 2, 0)  # C T V M -> M T V C
    anno['keypoint'] = (keypoint[:,:,:,:-1] - lower_bound) / (upper_bound - lower_bound) * size
    # print('key-point', anno['keypoint']) 
    anno['keypoint_score'] = keypoint[:,:,:,-1]
    anno['img_shape'] = (size, size)
    anno['original_shape'] = (size, size)
    anno['label'] = int(label[i])
    results.append(anno)
    prog_bar.update()



for anno in results[:15]:
    print(anno['label'])

random.shuffle(results)

print('After shuffle')
for anno in results[:15]:
    print(anno['label'])


# total =2922 split into train(2722) test(200)
train_list = results[:2722]
val_list = results[2722:]

print(f'len(train)={len(train_list)}, len(val)={len(val_list)}')

mmcv.dump(train_list, output_train_pkl)
mmcv.dump(val_list, output_val_pkl)

print('Finish!')