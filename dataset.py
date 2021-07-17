import os
import random
import h5py
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F

class listDataset(Dataset):
    def __init__(self, root, shape=None, shuffle=True, transform=None,  train=False, seen=0, batch_size=1, num_workers=4, exp='shanghai'):
        if train:
            root = root *4
        random.shuffle(root)
        
        self.nSamples = len(root)
        self.lines = root
        self.transform = transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.exp = exp
        
    def __len__(self):
        return self.nSamples
    def __getitem__(self, index):
        assert index <= len(self), 'index range error' 
        
        img_path = self.lines[index]
        
        img, target = load_data(img_path, self.exp, self.train)
        
        #img = 255.0 * F.to_tensor(img)
        
        #img[0,:,:]=img[0,:,:]-92.8207477031
        #img[1,:,:]=img[1,:,:]-95.2757037428
        #img[2,:,:]=img[2,:,:]-104.877445883

        if self.transform is not None:
            img = self.transform(img)
        return img,target


def load_data(img_path, exp, train=True):
    if (exp == 'shanghai'):
        gt_path = img_path.replace('.jpg','_nofilter.h5').replace('images','ground_truth')
        img = Image.open(img_path).convert('RGB')
        gt_file = h5py.File(gt_path)
        target = np.asarray(gt_file['density'])
    elif (exp == 'NWPU'):
        gt_path = img_path.replace('.jpg', '_nofilter.h5').replace('images', 'ground_truth')
        img = Image.open(img_path).convert('RGB')
        gt_file = h5py.File(gt_path)
        target = np.asarray(gt_file['density'])
    elif (exp == 'QNRF'):
        phase = ''
        if ('Train' in img_path):
            phase = 'Train'
        elif ('Test' in img_path):
            phase = 'Test'
        else:
            print('ERROR, invalid img_path')
            return
        gt_path = img_path.replace('.jpg', '_nofilter.h5').replace(phase, 'ground_truth/{}'.format(phase))
        img = Image.open(img_path).convert('RGB')
        gt_file = h5py.File(gt_path)
        target = np.asarray(gt_file['density'])

    return img, target
