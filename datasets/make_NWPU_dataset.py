import h5py
import glob
import os
import scipy.io as io
from matplotlib import pyplot as plt
import numpy as np
import json

root = '/drive/datasets/NWPU-Crowd'

#train = os.path.join(root,'images')
#val = os.path.join(root,'images')
#path_sets = [train, val]
phases = ['val']
img_paths = []
#for path in path_sets:
#    for img_path in glob.glob(os.path.join(path, '*.jpg')):
#        img_paths.append(img_path)
for phase in phases:
    with open(os.path.join(root, '{}.txt'.format(phase)), 'r') as f:
        data = f.read()
    for d in data.split('\n'):
        index = d.split(' ')[0]
        img_paths.append(os.path.join(root, "images/{}.jpg".format(index)))

for im_n, img_path in enumerate(img_paths):
    print(img_path)
    json_path = img_path.replace('.jpg','.json').replace('images','jsons')
    with open(json_path) as f:
        data = json.load(f)
    img = plt.imread(img_path)
    k = np.zeros((img.shape[0],img.shape[1]))
    gt = data['points']

    for i in range(0, len(gt)):
        if int(gt[i][1])<img.shape[0] and int(gt[i][0])<img.shape[1]:
            k[int(gt[i][1]), int(gt[i][0])] = 1

    with h5py.File(img_path.replace('.jpg','_nofilter.h5').replace('images','ground_truth'), 'w') as hf:
        hf['density'] = k
