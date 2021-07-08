
import os
import glob
import pathlib
from tqdm import tqdm
import argparse
import numpy as np
import h5py
import time
import math
from torchvision import transforms
import json
from model import ComputeLoss
import torch 
import torchvision.transforms.functional as F
import PIL.Image as Image
from utils import zeropad, vis_input
from model import Model



path = pathlib.Path(__file__).parent.absolute()
parser = argparse.ArgumentParser(description='RCVLab-AiimLab Crowd counting')

parser.add_argument('--model_desc', default='shanghaiB, darknet, lr=1e-3/', help="Set model description")
parser.add_argument('--dataset_path', default='/media/mohsen/myDrive/datasets/ShanghaiTech_Crowd_Counting_Dataset', help='path to dataset')
parser.add_argument('--exp_sets', default='part_B_final/test_data')
parser.add_argument('--use_gpu', default=True, action="store_false", help="Indicates whether or not to use GPU")
parser.add_argument('--device', default='0', type=str, help='GPU id to use.')
parser.add_argument('--checkpoint_path', default='../runs/weights', type=str, help='checkpoint path')

# MODEL
parser.add_argument('--model_file', default='model.yaml')
parser.add_argument('--cell_size', default=64, type=int, help="cell size")
parser.add_argument('--threshold', default=0.01, type=int, help="threshold for the classification output")

parser.add_argument('--vis', default=False, type=bool, help='visualize the inputs') 




def test():
    args = parser.parse_args()
    
    if args.use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device
        torch.cuda.manual_seed(time.time())
        CUDA =True
    else:
        CUDA = False

    path_sets = [os.path.join(args.dataset_path, args.exp_sets,'images')]
    img_paths = []
    for path in path_sets:
        for img_path in glob.glob(os.path.join(path, '*.jpg')):
            img_paths.append(img_path)

    
    args.checkpoint_path += ('/'+args.model_desc)
    args.checkpoint_path += 'model_best.pth.tar' #'checkpoint.pth.tar'
    
    model = Model(args.model_file)

    if CUDA:
        model = model.cuda()
    
    checkpoint = torch.load(args.checkpoint_path)

    model.load_state_dict(checkpoint['state_dict'])

    imgs, targets = [], []
    length = args.cell_size
    b_num = 0
    sum_mae = 0
    dataset_length = len(img_paths)
    
    pbar = enumerate(img_paths)
    pbar = tqdm(pbar, total=len(img_paths))

    for bi, img_path in pbar: 
        img_big = 255.0 * F.to_tensor(Image.open(img_path).convert('RGB'))
        img_big[0,:,:]=img_big[0,:,:]-92.8207477031
        img_big[1,:,:]=img_big[1,:,:]-95.2757037428
        img_big[2,:,:]=img_big[2,:,:]-104.877445883

        gt_file = h5py.File(img_path.replace('.jpg','_nofilter.h5').replace('images','ground_truth'),'r')
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

                if args.vis:
                    vis_input(img_chip, target_chip)

                coord = (target_chip).nonzero(as_tuple=False)
                bxy = [[b_num, yb/length, xb/length] for (yb, xb) in coord]
                targets.append(torch.tensor(bxy))

                img = torch.clone(img_chip)
                imgs.append(img)
                
                b_num += 1

                if i == (ni-1) and j == (nj-1):
                    imgs = torch.stack(imgs, dim=0).squeeze(1)
                    targets = [ti for ti in targets if len(ti) != 0]
                    targets = torch.cat(targets)

                    if CUDA:
                        imgs = imgs.cuda()
                        targets = targets.cuda()

                    with torch.no_grad():
                        pred = model(imgs, training=False)
                        pred = pred > args.threshold
                        pred = pred.sum()

                        targets = targets.shape[0]
                        mae = abs(pred - targets)
                        sum_mae += mae

                        s = str((bi, 'MAE: ', mae, 'pred: ', pred, 'target: ', targets))
                        pbar.set_description(s)

                        imgs = []
                        targets = []
                        b_num = 0


    print(sum_mae/dataset_length)





if __name__ == '__main__':
    test()