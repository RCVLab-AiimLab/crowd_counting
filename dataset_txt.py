import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from image_txt import *
import torchvision.transforms.functional as F
# from torchvision import datasets, transforms

class listDataset(Dataset):
    def __init__(self, root, depthroot, shape=None, depth=False, shuffle=True, transform1=None, transform2=None, train=False, seen=0, batch_size=1, num_workers=4, img_size=512):
        
        if shuffle==True:
            main_root = list(zip(root, depthroot))
            random.shuffle(main_root)
            root, depthroot = zip(*main_root)
        if train:
            root = root *4
            depthroot = depthroot *4
            
        self.nSamples = len(root)
        self.lines = root
        self.depthlines = depthroot
        self.transform1 = transform1
        self.transform2 = transform2
        self.train = train
        self.depth = depth
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        
    def __len__(self):
        return self.nSamples
    def __getitem__(self, index):
        assert index <= len(self), 'index range error' 
        
        img_path = self.lines[index]
        img_depth_path = self.depthlines[index]

        img, target = load_data(img_path, self.train, False, self.img_size)

        img_depth = load_data(img_depth_path, self.train, True, self.img_size)
        
        #img = 255.0 * F.to_tensor(img)
        
        #img[0,:,:]=img[0,:,:]-92.8207477031
        #img[1,:,:]=img[1,:,:]-95.2757037428
        #img[2,:,:]=img[2,:,:]-104.877445883

        if self.transform1 is not None:
            img = self.transform1(img)
        if self.transform2 is not None:
            img_depth = self.transform2(img_depth)
        
        return img, target, img_depth
