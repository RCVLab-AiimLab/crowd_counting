############## srun -p Aurora python Crowdcounting_CARLA/code/val.py
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
from model import CSRNet
import torch
# %matplotlib inline

from torchvision import datasets, transforms
transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ])
				   
root = 'Crowdcounting_CARLA/dataset'
#now generate the ShanghaiA's ground truth
part_A_train = os.path.join(root,'part_A_final/train_data','images')
part_A_test = os.path.join(root,'part_A_final/test_data','images')
part_B_train = os.path.join(root,'part_B_final/train_data','images')
part_B_test = os.path.join(root,'part_B_final/test_data','images')
path_sets = [part_B_test]

img_paths = []
for path in path_sets:
    # print(path)
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        # print(img_path)
        img_paths.append(img_path)
model = CSRNet()

model = model.cuda()

checkpoint = torch.load('Crowdcounting_CARLA/code/runs/weights/model_best.pth.tar')

model.load_state_dict(checkpoint['state_dict'])
mae = 0
mape = 0
sum_mae = 0
sum_mape = 0
num = 1
outpath = 'Crowdcounting_CARLA/dataset/part_B_final/test_data/out_data'
for i in range(len(img_paths)):
    
    print(img_paths[i])
    # img = 255.0 * F.to_tensor(Image.open(img_paths[i]).convert('RGB'))
    # img[0,:,:]=img[0,:,:]-92.8207477031
    # img[1,:,:]=img[1,:,:]-95.2757037428
    # img[2,:,:]=img[2,:,:]-104.877445883
    # img = img.cuda()
    img = transform(Image.open(img_paths[i]).convert('RGB')).cuda()
    gt_file = h5py.File(img_paths[i].replace('.jpg','.h5').replace('images','ground_truth'),'r')
    groundtruth = np.asarray(gt_file['density'])
    output = model(img.unsqueeze(0))
    print(output[0,0,:,:].detach().cpu().numpy().shape)
    plt.imshow(output[0,0,:,:].detach().cpu().numpy())
    plt.savefig(img_paths[i].replace('images','out_data')) 
    mae = abs(output.detach().cpu().sum().numpy()-np.sum(groundtruth))
    mape = abs(output.detach().cpu().sum().numpy()-np.sum(groundtruth))/abs(np.sum(groundtruth))
    sum_mae += mae
    sum_mape += mape
    print(i,mape)
    num += 1
print("percent is",(sum_mape/len(img_paths))*100)
