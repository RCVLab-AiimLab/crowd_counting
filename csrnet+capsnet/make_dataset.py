import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import glob

from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter 
import scipy
from scipy.spatial import KDTree
import json
from matplotlib import cm as CM
from image import *
from model import CSRNet
import torch

CSRNet_EXP = False
Queens_Geotab_EXP = not CSRNet_EXP

#set the root to the Shanghai dataset you download
if CSRNet_EXP:
    root = '/media/mohsen/myDrive/datasets/ShanghaiTech_Crowd_Counting_Dataset'

elif Queens_Geotab_EXP: 
    root = '/media/mohsen/myDrive/datasets/Queens_Geotab/Crowd_Counting_Sim_Dataset_01/Crowd_Counting_Sim_Dataset'
    #root = '/media/mohsen/myDrive/datasets/Queens_Geotab/Crowd_Counting_Sim_Dataset_02/Crowd_Counting_Sim_Dataset'

part_A_train = os.path.join(root,'part_A_final/train_data','images')
part_A_test = os.path.join(root,'part_A_final/test_data','images')
part_B_train = os.path.join(root,'part_B_final/train_data','images')
part_B_test = os.path.join(root,'part_B_final/test_data','images')

#this is borrowed from https://github.com/davideverona/deep-crowd-counting_crowdnet
def gaussian_filter_density(gt):
    print(gt.shape)
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density

    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))
    leafsize = 2048
    # build kdtree
    tree = KDTree(pts.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(pts, k=4)

    print('generate density...')
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1],pt[0]] = 1.
        if gt_count > 1:
            sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
        else:
            sigma = np.average(np.array(gt.shape))/2./2. #case: 1 point
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    print('done.')
    return density


path_sets = [part_A_train,part_A_test]
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
    if gt.shape[0] < 4:
        continue
    for i in range(0,len(gt)):
        if int(gt[i][1])<img.shape[0] and int(gt[i][0])<img.shape[1]:
            k[int(gt[i][1]),int(gt[i][0])]=1
    plt.subplot(1,3,2).imshow(k)
    plt.title('Original people count: ' + str(np.sum(k)))
    k = gaussian_filter_density(k)
    with h5py.File(img_path.replace('.jpg','.h5').replace('images','ground_truth'), 'w') as hf:
            hf['density'] = k

    #now see a sample
    plt.subplot(1,3,1).imshow(Image.open(img_paths[im_n]))
    gt_file = h5py.File(img_paths[im_n].replace('.jpg','.h5').replace('images','ground_truth'),'r')
    groundtruth = np.asarray(gt_file['density'])
    plt.subplot(1,3,3).imshow(groundtruth,cmap=CM.jet)
    count = np.sum(groundtruth)# don't mind this slight variation
    plt.title('People count after filter: ' + str(count))
    plt.show()

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
            k[int(gt[i][1]),int(gt[i][0])]=1
    k = gaussian_filter(k,15)
    with h5py.File(img_path.replace('.jpg','.h5').replace('images','ground_truth'), 'w') as hf:
            hf['density'] = k
    
    #now see a sample 
    plt.subplot(1,2,1).imshow(Image.open(img_paths[im_n]))
    gt_file = h5py.File(img_paths[im_n].replace('.jpg','.h5').replace('images','ground_truth'),'r')
    groundtruth = np.asarray(gt_file['density'])
    plt.subplot(1,2,2).imshow(groundtruth,cmap=CM.jet)
    count = np.sum(groundtruth)# don't mind this slight variation
    plt.title('People count: ' + str(np.round(count)))
    plt.show()