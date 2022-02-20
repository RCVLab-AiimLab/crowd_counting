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


# Set the root to the datasets
if CSRNet_EXP:
    root = r"C:\Users\mahdi\Desktop\SASNet_ROOT\ShanghaiTech"

elif UCF_QNRF:
    root = '/home/mohsen/Desktop/UCF-QNRF_ECCV18'

elif CARLA_SIM:
    root = '/media/mohsen/myDrive/datasets/Crowd_Counting_Sim_Dataset_10k'

if ShanghaiTech:
    part_A_train = os.path.normpath(os.path.join(root,'part_A_final/train_data','images'))
    part_A_test = os.path.normpath(os.path.join(root,'part_A_final/test_data','images'))
    part_B_train = os.path.normpath(os.path.join(root,'part_B_final/train_data','images'))
    part_B_test = os.path.normpath(os.path.join(root,'part_B_final/test_data','images'))
    
elif UCF_QNRF:
    train = os.path.join(root,'Train')
    test = os.path.join(root,'Test')

    path_sets = [train, test]
    img_paths = []
    for path in path_sets:
        for img_path in glob.glob(os.path.join(path, '*.jpg')):
            img_paths.append(img_path)

    for im_n, img_path in enumerate(img_paths):
        print(img_path)
        mat = io.loadmat(img_path.replace('.jpg','_ann.mat'))
        img = plt.imread(img_path)
        k = np.zeros((img.shape[0],img.shape[1]))
        gt = mat["annPoints"]

        for i in range(0, len(gt)):
            if int(gt[i][1])<img.shape[0] and int(gt[i][0])<img.shape[1]:
                k[int(gt[i][1]), int(gt[i][0])] = 1

        with h5py.File(img_path.replace('.jpg','_nofilter.h5'), 'w') as hf:
            hf['density'] = k

elif CARLA_SIM:
    train = os.path.join(root,'train_data', 'images')
    test = os.path.join(root,'test_data', 'images')

    path_sets = [train, test]
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

        with h5py.File(img_path.replace('.jpg','.h5').replace('images','ground_truth'), 'w') as hf:
            hf['density'] = k

    
        plt.subplot(1,3,2).imshow(k)
        plt.title('Original people count: ' + str(np.sum(k)))

        #now see a sample
        plt.subplot(1,3,1).imshow(Image.open(img_paths[im_n]))
        #gt_file = h5py.File(img_paths[im_n].replace('.jpg','_nofilter.h5'),'r')
        #groundtruth = np.asarray(gt_file['density'])
        #plt.subplot(1,3,3).imshow(groundtruth,cmap=CM.jet)
        #count = np.sum(groundtruth)# don't mind this slight variation
        #plt.title('People count after filter: ' + str(count))
        plt.show()

if ShanghaiTech:
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
        
  