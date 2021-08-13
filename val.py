
import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter 
import scipy
import json
import torchvision.transforms.functional as F
from matplotlib import cm as CM
from image import *
from model_csr import CSRNet
import torch
import torch.nn as nn
# %matplotlib inline

from torchvision import datasets, transforms
transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ])
				   
root = 'crowd_csr_grid/datasets/shanghai'
#now generate the ShanghaiA's ground truth
part_A_train = os.path.join(root,'part_A_final/train_data','images')
part_A_test = os.path.join(root,'part_A_final/test_data','images')
part_B_train = os.path.join(root,'part_B_final/train_data','images')
part_B_test = os.path.join(root,'part_B_final/test_data','images')
path_sets = [part_A_train]

img_paths = []
for path in path_sets:
    # print(path)
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        # print(img_path)
        img_paths.append(img_path)
model = CSRNet()

model = model.cuda()

checkpoint = torch.load('crowd_counting/runs/weights/csr/checkpoint.pth.tar')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

model.load_state_dict(checkpoint['state_dict'])
mae = 0
sum_mae = 0
num = 1
outpath = 'crowd_csr_grid/datasets/shanghai/part_A_final/train_data/out_data'
for i in range(len(img_paths)):
    
    print(img_paths[i])
    # img = 255.0 * F.to_tensor(Image.open(img_paths[i]).convert('RGB'))
    # img[0,:,:]=img[0,:,:]-92.8207477031
    # img[1,:,:]=img[1,:,:]-95.2757037428
    # img[2,:,:]=img[2,:,:]-104.877445883
    # img = img.cuda()
    img = transform(Image.open(img_paths[i]).convert('RGB')).cuda()
    density = model(img.unsqueeze(0))
    print(img.shape)
    density_up = nn.Upsample((img.shape[-2], img.shape[-1]), mode='nearest')(density)
    density_non_zero = 1*(density_up > 0.001)
    # print(output.shape)
    # density = nn.Upsample((output.shape[-2], output.shape[-1]), mode='bilinear')(density)
    # density_non_zero = 1*(density > 0.01)
    plt.imshow(density_non_zero[0,0,:,:].detach().cpu().numpy())
    plt.savefig(img_paths[i].replace('images','out_data_jpg/density_non_zero')) 
    plt.imshow(density[0,0,:,:].detach().cpu().numpy())
    plt.savefig(img_paths[i].replace('images','out_data_jpg/density')) 
    plt.imshow((img.permute(1,2,0).detach().cpu().numpy() * 255).astype(np.uint8))
    plt.savefig(img_paths[i].replace('images','out_data_jpg/img')) 
    # plt.savefig(img_paths[i].replace('images','out_data_jpg'),density[0,0,:,:].detach().cpu().numpy()) 
    # hf = h5py.File(img_paths[i].replace('images','out_data_h5').replace('.jpg','.h5'), 'w')
    # hf.create_dataset('out_data', data=output[0,0,:,:].detach().cpu().numpy())
    # hf.close()

