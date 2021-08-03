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
import math
from train import args, zeropad


from torchvision import datasets, transforms
transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])


root = '/media/mohsen/myDrive/datasets/ShanghaiTech_Crowd_Counting_Dataset'
if args.use_gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    torch.cuda.manual_seed(args.seed)
    CUDA =True
else:
    CUDA = False

#now generate the ShanghaiA's ground truth
part_A_train = os.path.join(root,'part_A_final/train_data','images')
part_A_test = os.path.join(root,'part_A_final/test_data','images')
part_B_train = os.path.join(root,'part_B_final/train_data','images')
part_B_test = os.path.join(root,'part_B_final/test_data','images')
path_sets = [part_A_test]

img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)

model = CSRNet(args.depth, reconstruction_type=args.decoder, imsize=args.input_size//8, 
                routing_iterations=args.routing, primary_caps_gridsize=8,
                img_channel=3, batchnorm=args.batch_norm, num_primary_capsules=32,
                loss=args.loss, leaky_routing=args.leaky)

model = model.cuda()
chkpnt_dir = '../runs/weights/checkpoint.pth.tar'
checkpoint = torch.load(chkpnt_dir)

model.load_state_dict(checkpoint['state_dict'])

sum_mae = 0
length = 32
imgs, targets = [], []
b_num = 0
dataset_length = len(img_paths)
for bi in range(len(img_paths)):
    img_big = 255.0 * F.to_tensor(Image.open(img_paths[bi]).convert('RGB'))

    img_big[0,:,:]=img_big[0,:,:]-92.8207477031
    img_big[1,:,:]=img_big[1,:,:]-95.2757037428
    img_big[2,:,:]=img_big[2,:,:]-104.877445883
    #img_big = img_big.cuda()
    #img_big = transform(Image.open(img_paths[bi]).convert('RGB')).cuda()
    gt_file = h5py.File(img_paths[bi].replace('.jpg','_nofilter.h5').replace('images','ground_truth'),'r')
    groundtruth = np.asarray(gt_file['density'])

    ni = int(math.ceil(img_big.shape[1] / length)) 
    nj = int(math.ceil(img_big.shape[2] / length))  
    for i in range(ni):  
        for j in range(nj):  
            y2 = min((i + 1) * length, img_big.shape[1])
            y1 = y2 - length
            x2 = min((j + 1) * length, img_big.shape[2])
            x1 = x2 - length

            img_chip = img_big[:, y1:y2, x1:x2]
            img_chip = zeropad(img_chip.squeeze(0).permute(1,2,0).numpy(), length - img_chip.shape[1], length - img_chip.shape[2])
            img_chip = torch.from_numpy(img_chip).permute(2,0,1)
            target_chip = groundtruth[y1:y2, x1:x2]
            target_chip = zeropad(target_chip, length - img_chip.shape[1], length - img_chip.shape[2], target=True)
            target_chip = torch.from_numpy(target_chip)
            assert img_chip.shape[1] == img_chip.shape[2] == length, 'image size error'
            assert target_chip.shape[0] == target_chip.shape[1] == length, 'target size error'

            '''
            imtest = img_chip[0,...].permute(1, 2, 0).cpu()
            imtest = cv2.normalize(np.float32(imtest), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            imtest = imtest.astype(np.uint8)
            plt.subplot(121).imshow(imtest)
            plt.subplot(122).imshow(target_chip.squeeze(0), cmap=CM.jet)
            count = np.sum(target_chip.numpy().squeeze(0))
            plt.title('People count: ' + str(count))
            plt.show()
            '''
            count = np.sum(target_chip.numpy())
            count = np.round(count)
            if count >= 10:
                #continue
                count = 9
            target = torch.zeros(10)
            target[int(count)] = 1

            img = torch.clone(img_chip)
            target = torch.clone(target)
            
            imgs.append(img)
            targets.append(target)
            b_num += 1

            if i == (ni-1) and j == (nj-1):
                if b_num > 800:
                    dataset_length -= 1
                    imgs = []
                    targets = []
                    b_num = 0
                    continue
                img = torch.stack(imgs, dim=0).squeeze(1)
                target = torch.stack(targets, dim=0)

                if CUDA:
                    img = img.cuda()
                    target = target.cuda()

                with torch.no_grad():
                    _, _, predictions = model(img)
                    predictions = np.argmax(predictions.cpu(), axis=1) 
                    target = np.argmax(target.cpu(), axis=1) 

                    mae = abs(predictions.detach().cpu().sum().numpy() - np.sum(groundtruth))
                    sum_mae += mae
                    print(bi, mae)
                    imgs = []
                    targets = []
                    b_num = 0
    
print(sum_mae/dataset_length)