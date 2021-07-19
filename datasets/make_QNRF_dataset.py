import h5py
import glob
import os
import scipy.io as io
from matplotlib import pyplot as plt
import numpy as np
import json

root = '/home/16amf8/data/datasets/UCF-QNRF_ECCV18'

#train = os.path.join(root,'images')
#val = os.path.join(root,'images')
#path_sets = [train, val]
phase = 'Test'
img_paths = []
#for path in path_sets:
#    for img_path in glob.glob(os.path.join(path, '*.jpg')):
#        img_paths.append(img_path)
#for phase in phases:
img_paths = [os.path.join(root, phase, p) for p in os.listdir('/home/16amf8/data/datasets/UCF-QNRF_ECCV18/{}'.format(phase)) if '.jpg' in p]

for im_n, img_path in enumerate(img_paths):
    print(img_path)
    mat_path = img_path.replace('.jpg','_ann.mat')
    mat = io.loadmat(mat_path)
    img = plt.imread(img_path)
    k = np.zeros((img.shape[0],img.shape[1]))
    gt = mat['annPoints']

    for i in range(0, len(gt)):
        if int(gt[i][1])<img.shape[0] and int(gt[i][0])<img.shape[1]:
            k[int(gt[i][1]), int(gt[i][0])] = 1

    with h5py.File(img_path.replace('.jpg','_nofilter.h5').replace(phase, 'ground_truth/{}'.format(phase)), 'w') as hf:
        hf['density'] = k
