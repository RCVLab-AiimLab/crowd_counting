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
        #if shuffle==True:
        # print('root',root)
        if depth:
            main_root = list(zip(root, depthroot))
            # random.shuffle(main_root)
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

        
        img, target, bxy, l, img_depth = load_data(img_path, img_depth_path, self.train, depth=self.depth)
        
        #img = 255.0 * F.to_tensor(img)
        
        #img[0,:,:]=img[0,:,:]-92.8207477031
        #img[1,:,:]=img[1,:,:]-95.2757037428
        #img[2,:,:]=img[2,:,:]-104.877445883

        if self.transform1 is not None:
            img = self.transform1(img)
        if self.depth:
            if self.transform2 is not None:
                img_depth = self.transform2(img_depth)

        return img, target, bxy, l, img_depth


def load_data(img_path, depth_path=None, train=True, depth=False):
    # print(img_path)
    gt_path = img_path

    hf = h5py.File(img_path)
    img = np.asarray(hf['image'])

    target = np.asarray(hf['target'])

    bxy = np.asarray(hf['bxy'])

    hf.close()
    # l = h5py.File(img_path)
    # l = np.asarray(l['l'])
    l = target.shape[-1]

    img_depth = torch.zeros(1)
    if depth:
        h5 = h5py.File(depth_path,'r')
        img_depth = h5['depth'][:]
    
    return img, target, bxy, l, img_depth
    
def get_list(args):
    catgrs = os.listdir('crowd_csr_grid/data_chopped/part_A_train/')
    direct = []
    for i, cat in enumerate(catgrs):
        direct.append([])
        filenames = os.listdir(os.path.join('crowd_csr_grid/data_chopped/part_A_train/',cat))
        # counter = 0
        for j in filenames:
            # print(j)
            # counter = counter + 1
            direct[i].append(os.path.join('crowd_csr_grid/data_chopped/part_A_train/',cat,j))
            # if counter == 16:
                # break
        random.shuffle(direct[i])
    

    # direct[0] = direct[0][0:34]
    # direct[1] = direct[1][0:67]
    # print('direct 0',len(direct[0]))
    # print('direct 1',len(direct[1]))

    # print(direct[i])
    train_list = []
    num_cat = len(catgrs)
    i = 0
    count_ctgr = np.zeros((len(catgrs),1))
    c = np.zeros((len(catgrs),1))
    remain = np.ones((len(catgrs),1))
    # print(len(direct[i]) - args.stack_size)
    while remain.any() == 1:
        # print('count_ctgr[i]', count_ctgr[i])
        # print('len(direct[i]) - count_ctgr[i]', len(direct[i]) - count_ctgr[i])
        # print('i',i)
        # print('remain',remain)
        # print('c', c)
        num = len(direct[i])//args.stack_size
        if c[i] == num:
            remain[i] = 0 
            i = i + 1
            
        else: 
            train_list.append(direct[i][int(count_ctgr[i])])
            count_ctgr[i] = count_ctgr[i] + 1
            
            if count_ctgr[i] % (args.stack_size)==0:
                c[i] = c[i] + 1 
                i = i + 1
                
                    
        if i == len(catgrs):
                i = 0     
            
    # print(train_list)

    if args.depth:
        train_list_depth = [st.replace('images', 'depth_resized_h5') for st in train_list]
        val_list_depth = [st.replace('images', 'depth_resized_h5') for st in val_list]
        train_list_depth = [st.replace('.jpg', '.h5') for st in train_list_depth]
        val_list_depth = [st.replace('.jpg', '.h5') for st in val_list_depth]
    else:
        train_list_depth = None
        val_list_depth = None

    return train_list, train_list_depth
