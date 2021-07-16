import os
import random
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F

class listDataset(Dataset):
    def __init__(self, root, depthroot, shape=None, depth=False, shuffle=True, transform1=None, transform2=None, train=False, seen=0, batch_size=1, num_workers=4):
        if shuffle==True:
            if depth:
                main_root = list(zip(root, depthroot))
                random.shuffle(main_root)
                root, depthroot = zip(*main_root)
                depthroot = depthroot *4
                self.depthlines = depthroot
        if train:
            root = root * 4
        
        self.nSamples = len(root)
        self.lines = root
        self.transform1 = transform1
        self.transform2 = transform2
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.depth = depth
        
    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error' 
        
        img_path = self.lines[index]
        if self.depth:
            img_depth_path = self.depthlines[index]
        else:
            img_depth_path = None

        
        img, target, img_depth = load_data(img_path, img_depth_path, self.train, depth=self.depth)
        
        #img = 255.0 * F.to_tensor(img)
        
        #img[0,:,:]=img[0,:,:]-92.8207477031
        #img[1,:,:]=img[1,:,:]-95.2757037428
        #img[2,:,:]=img[2,:,:]-104.877445883

        if self.transform1 is not None:
            img = self.transform1(img)
        if self.depth:
            if self.transform2 is not None:
                img_depth = self.transform2(img_depth)

        return img, target, img_depth


def load_data(img_path, depth_path=None, train=True, depth=False):
    gt_path = img_path.replace('.jpg','_nofilter.h5').replace('images','ground_truth')
    img = Image.open(img_path).convert('RGB')
    gt_file = h5py.File(gt_path)
    target = np.asarray(gt_file['density'])
    img_depth = torch.zeros(1)
    if depth:
        h5 = h5py.File(depth_path,'r')
        img_depth = h5['depth'][:]
    
    return img, target, img_depth
    