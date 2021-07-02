import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import glob
import sys
np.set_printoptions(threshold=sys.maxsize)

from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter 
import scipy
from scipy.spatial import KDTree
import json
from matplotlib import cm as CM
from image import *
from model import CSRNet
import torch

#set the root to the Shanghai dataset you download
root = 'D:/queens/codes/capsnet_crowd/dataset'

#now generate the ShanghaiA's ground truth
part_A_train = os.path.join(root,'part_A_final/train','images')
part_A_test = os.path.join(root,'part_A_final/test','images')
part_B_train = os.path.join(root,'part_B_final/train','images')
part_B_test = os.path.join(root,'part_B_final/test','images')

#this is borrowed from https://github.com/davideverona/deep-crowd-counting_crowdnet


path_sets = [part_A_train,part_A_test]
img_paths = []
# print(path_sets)
print(glob.glob(os.path.join(path_sets[0], '*.jpg')))
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        
        print(img_path)
        img_paths.append(img_path)


for img_path in img_paths:
    mat = io.loadmat(img_path.replace('.jpg','.mat').replace('images','ground_truth').replace('IMG_','GT_IMG_'))
    img= plt.imread(img_path)
    k = np.zeros((img.shape[0],img.shape[1]))
    gt = mat["image_info"][0,0][0,0][0]
    for i in range(0,len(gt)):
        if int(gt[i][1])<img.shape[0] and int(gt[i][0])<img.shape[1]:
            k[int(gt[i][1]),int(gt[i][0])]=1
    # plt.imshow(k)
    # plt.savefig(img_path.replace('images','gt_image_noblur')) 
    # print(k)
    k = gaussian_filter(k,15)
    print(k.shape)
    with h5py.File(img_path.replace('.jpg','.h5').replace('images','ground_truth'), 'w') as hf:
            hf['density'] = k
    # plt.imshow(k)
    # plt.savefig(img_path.replace('images','gt_image')) 



# now see a sample from ShanghaiA
# plt.subplot(1,2,1).imshow(Image.open(img_paths[0]))
# gt_file = h5py.File(img_paths[0].replace('.jpg','.h5').replace('images','ground_truth'),'r')
# groundtruth = np.asarray(gt_file['density'])
# plt.subplot(1,2,2).imshow(groundtruth,cmap=CM.jet)
# plt.show()
# count = np.sum(groundtruth)# don't mind this slight variation
# print('Part A is done')


#now generate the ShanghaiB's ground truth
print('this is part b')
path_sets = [part_B_train,part_B_test]
img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)

for img_path in img_paths:
    # print(img_path)
    mat = io.loadmat(img_path.replace('.jpg','.mat').replace('images','ground_truth').replace('IMG_','GT_IMG_'))
    img= plt.imread(img_path)
    k = np.zeros((img.shape[0],img.shape[1]))
    gt = mat["image_info"][0,0][0,0][0]
    for i in range(0,len(gt)):
        if int(gt[i][1])<img.shape[0] and int(gt[i][0])<img.shape[1]:
            k[int(gt[i][1]),int(gt[i][0])]=1
    
    # plt.imshow(k)
    # plt.savefig(img_path.replace('images','gt_image_noblur')) 
    k = gaussian_filter(k,15)
    # np.savetxt(img_path.replace('images','gt_txt').replace('.jpg','.txt'),k, fmt='%3.3f')
    with h5py.File(img_path.replace('.jpg','.h5').replace('images','ground_truth'), 'w') as hf:
            hf['density'] = k
    # print(k)
    # plt.imshow(k)
    # plt.savefig(img_path.replace('images','gt_image')) 
    print('generate density...')        
print('we are done')