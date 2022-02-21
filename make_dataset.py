#import packages
import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import glob

from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter 
from scipy.spatial import KDTree
from matplotlib import cm as CM

ShanghaiTech = True
UCF_QNRF = False
CSRNet_EXP =True
CARLA_SIM = False


# Set the root to the dataset
root = "<set to root folder>"

part_A_train = os.path.normpath(os.path.join(root,'part_A_final/train_data','images'))
part_A_test = os.path.normpath(os.path.join(root,'part_A_final/test_data','images'))
part_B_train = os.path.normpath(os.path.join(root,'part_B_final/train_data','images'))
part_B_test = os.path.normpath(os.path.join(root,'part_B_final/test_data','images'))
    


path_sets = [part_A_train, part_A_test]
img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)


for im_n, img_path in enumerate(img_paths):
    print(img_path)
    mat = io.loadmat(img_path.replace('.jpg','.mat').replace('images','ground_truth').replace('IMG_','GT_IMG_'))
    img = plt.imread(img_path)
    k = np.zeros((img.shape[0],img.shape[1]))
    gt = mat["image_info"][0,0][0,0][0]

    for i in range(0, len(gt)):
        if int(gt[i][1])<img.shape[0] and int(gt[i][0])<img.shape[1]:
            k[int(gt[i][1]), int(gt[i][0])] = 1

    with h5py.File(img_path.replace('.jpg','_nofilter.h5').replace('images','ground_truth'), 'w') as hf:
        hf['density'] = k


#now generate the ShanghaiB's ground truth
path_sets = [part_B_train,part_B_test]
img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)

for im_n, img_path in enumerate(img_paths):
    print(img_path)
    mat = io.loadmat(img_path.replace('.jpg','.mat').replace('images','ground_truth').replace('IMG_','GT_IMG_'))
    img = plt.imread(img_path)
    k = np.zeros((img.shape[0],img.shape[1]))
    gt = mat["image_info"][0,0][0,0][0]
    for i in range(0,len(gt)):
        if int(gt[i][1])<img.shape[0] and int(gt[i][0])<img.shape[1]:
            k[int(gt[i][1]), int(gt[i][0])] = 1

    with h5py.File(img_path.replace('.jpg','_nofilter.h5').replace('images','ground_truth'), 'w') as hf:
        hf['density'] = k
        
  