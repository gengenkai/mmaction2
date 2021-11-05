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


output_train_pkl = '/mnt/lustre/liguankai/data/ski/2500_422/train.pkl'
output_val_pkl = '/mnt/lustre/liguankai/data/ski/2500_422/val.pkl'

n_samples = len(label)


# for i, keypoint in enumerate(data):
    # print(keypoint.shape)  # 3 2500 25 1 C T V M     numpy.ndarray float32

    
results = []
prog_bar = mmcv.ProgressBar(n_samples)



for i, keypoint in enumerate(data):
    anno = dict()
    anno['total_frames'] = 2500
    anno['keypoint'] = keypoint.transpose(3, 1, 2, 0)  # C T V M -> M T V C
    anno['img_shape'] = (1080, 720)
    anno['original_shape'] = (1080, 720)
    anno['label'] = int(label[i])
    # print(int(label[i]))
    results.append(anno)
    prog_bar.update()
    
for anno in results[:15]:
    print(anno['label'])

random.shuffle(results)
print('After shuffle')
for anno in results[:15]:
    print(anno['label'])

# mmcv.dump(results, output_pkl)
# print('Finish!')



# total =2922 split into train(2722) val(200)
# train_list = results[:2722]
# val_list = results[2722:]



# plan b --- total = 2922 split into train(2500) val(422)
train_list = results[:2500]
val_list = results[2500:]
 

print(f'len(train)={len(train_list)}, len(val)={len(val_list)}')

mmcv.dump(train_list, output_train_pkl)
mmcv.dump(val_list, output_val_pkl)

print('Finish!')