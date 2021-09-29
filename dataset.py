import os
import random
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F

class listDataset(Dataset):
    def __init__(self, root, depthroot, shape=None, density=False, depth=False, augment=False, shuffle=True, transform1=None, transform2=None, train=False, seen=0, batch_size=1, num_workers=4):
        #if shuffle==True:
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
        self.density = density
        self.depth = depth
        self.augment = augment
        
    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error' 
        
        img_path = self.lines[index]
        if self.depth:
            img_depth_path = self.depthlines[index]
        else:
            img_depth_path = None

        
        img, target, img_depth = load_data(img_path, img_depth_path, self.train, density=self.density, depth=self.depth, augment=self.augment)
        
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


def load_data(img_path, depth_path=None, train=True, density=False, depth=False, augment=False):
    if density:
        gt_path = img_path.replace('.jpg','.h5').replace('images','ground_truth')
    else:
        gt_path = img_path.replace('.jpg','_nofilter.h5').replace('images','ground_truth')
        
    img = Image.open(img_path).convert('RGB')
    gt_file = h5py.File(gt_path)
    target = np.asarray(gt_file['density'])
    img_depth = torch.zeros(1)
    if depth:
        '''h5 = h5py.File(depth_path,'r')
        img_depth = h5['depth'][:]'''
        depth_path = depth_path.replace('depth_resized_h5', 'depth').replace('.h5', '.png')
        img_depth = Image.open(depth_path)
        img_depth = np.array(img_depth, dtype=float)
    
    if augment:
        crop_size = (img.size[0]//2, img.size[1]//2)
        if random.randint(0,9)<= -1:
            dx = int(random.randint(0, 1) * img.size[0] * 1./2)
            dy = int(random.randint(0, 1) * img.size[1] * 1./2)
        else:
            dx = int(random.random() * img.size[0] * 1./2)
            dy = int(random.random() * img.size[1] * 1./2)
        
        img = img.crop((dx, dy, crop_size[0] + dx, crop_size[1] + dy))
        target = target[dy:crop_size[1] + dy, dx:crop_size[0] + dx]
        
        if random.random() > 0.8:
            target = np.fliplr(target)
            target = torch.from_numpy(target.copy())
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

    return img, target, img_depth
    