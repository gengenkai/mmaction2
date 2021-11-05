import mmcv 
import numpy as np
from numpy.lib.function_base import percentile 



test_data = '/mnt/lustre/liguankai/data/ski/test_A_data.npy'
data = np.load(test_data)  
n_samples = len(data)   # 628 
print(n_samples)

# test_output = '/mnt/lustre/liguankai/codebase/mmaction2/ski_test.pkl'
test_output = '/mnt/lustre/liguankai/codebase/mmaction2/ski_bone_test.pkl'
preds = mmcv.load(test_output)
n_preds = len(preds)
print(n_preds)  # 628



output_file = 'submission.csv'

import csv


values = []
for i,pred in enumerate(preds):
    # print(i, pred)
    cate = np.argmax(pred)
    values.append((i, cate))

header=['sample_index','predict_category']
with open(output_file, 'w') as fp:
    writer = csv.writer(fp)
    writer.writerow(header)
    writer.writerows(values)

print('Finish~')