
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
from torchvision import transforms
import PIL.Image as Image
from utils import zeropad, vis_input
from model import Model



path = pathlib.Path(__file__).parent.absolute()
parser = argparse.ArgumentParser(description='RCVLab-AiimLab Crowd counting')

parser.add_argument('--model_desc', default='shanghaiA, cell128/', help="Set model description")
parser.add_argument('--dataset_path', default='/media/mohsen/myDrive/datasets/ShanghaiTech_Crowd_Counting_Dataset', help='path to dataset')
parser.add_argument('--exp_sets', default='part_A_final/test_data')
parser.add_argument('--use_gpu', default=True, help="indicates whether or not to use GPU")
parser.add_argument('--device', default='0', type=str, help='GPU id to use.')
parser.add_argument('--checkpoint_path', default='../runs/weights', type=str, help='checkpoint path')
parser.add_argument('--log_dir', default='../runs/log', type=str, help='log dir')

# MODEL
parser.add_argument('--model_file', default='model.yaml')
parser.add_argument('--cell_size', default=128, type=int, help="cell size")
parser.add_argument('--threshold', default=0.1, type=int, help="threshold for the classification output")

parser.add_argument('--best', default=False, type=bool, help='best or last saved checkpoint?') 
parser.add_argument('--vis_patch', default=False, type=bool, help='visualize the patches') 
parser.add_argument('--vis_image', default=False, type=bool, help='visualize the whole image') 
parser.add_argument('--prob_map', default=False, type=bool, help='using threshold or probability map?') 




def test():
    args = parser.parse_args()
    args.log_dir += ('/'+args.model_desc)
    
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
    if args.best:
        args.checkpoint_path += 'model_best.pth.tar'
    else:
        args.checkpoint_path += 'checkpoint.pth.tar'

    model = Model(args.model_file)

    if CUDA:
        model = model.cuda()
    
    checkpoint = torch.load(args.checkpoint_path)

    model.load_state_dict(checkpoint['state_dict'])

    #for thresh in [0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.3, 0.5, 0.7, 0.9]:
    for thresh in [0.1]:
        print('thresh: ', thresh)
        args.threshold = thresh
        imgs, targets, target_chips = [], [], []
        length = args.cell_size
        b_num = 0
        sum_mae_prob, sum_mae_th, sum_mse, sum_mae_cell = 0.0, 0.0, 0.0, 0.0
        dataset_length = len(img_paths)
        
        pbar = enumerate(img_paths)
        pbar = tqdm(pbar, total=len(img_paths))

        with open(args.log_dir + 'results_test.txt', 'w') as f:
            for bi, img_path in pbar: 
                transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
                img_big = Image.open(img_path).convert('RGB')
                img_big = transform(img_big)
                #img_big = 255.0 * F.to_tensor(Image.open(img_path).convert('RGB'))
                
                '''
                img_big[0,:,:] = img_big[0,:,:] - 92.8207477031
                img_big[1,:,:] = img_big[1,:,:] - 95.2757037428
                img_big[2,:,:] = img_big[2,:,:] - 104.877445883
                '''
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
                        target_chip = zeropad(target_chip, length - target_chip.shape[0], length - target_chip.shape[1], target=True)
                        target_chip = torch.from_numpy(target_chip)
                        assert img_chip.shape[1] == img_chip.shape[2] == length, 'image size error'
                        assert target_chip.shape[0] == target_chip.shape[1] == length, 'target size error'

                        coord = (target_chip).nonzero(as_tuple=False)
                        bxy = [[b_num, yb/length, xb/length, i, j] for (yb, xb) in coord]
                        targets.append(torch.tensor(bxy))

                        img = torch.clone(img_chip)
                        imgs.append(img)

                        target_chips.append(target_chip)
                        
                        b_num += 1

                        if i == (ni-1) and j == (nj-1):
                            imgs = torch.stack(imgs, dim=0).squeeze(1)
                            targets = [ti for ti in targets if len(ti) != 0]
                            targets = torch.cat(targets)

                            target_chips = torch.stack(target_chips, dim=0).squeeze(1)

                            if CUDA:
                                imgs = imgs.cuda()
                                targets = targets.cuda()

                            with torch.no_grad():
                                predictions = model(imgs, training=False)

                                if args.vis_image:
                                    vis_image(args, img_big, imgs, target_chips, ni, nj, predictions)
                                
                                if args.vis_patch:
                                    im_i = imgs.size(0)//4
                                    vis_input(imgs[im_i, ...], target_chips[im_i, ...], predicted=predictions[im_i, 0, ...], thresholded=predictions[im_i, 0, ...] > args.threshold)
                            
                                targets = targets.shape[0]
                                pred_prob = predictions[..., 0].sum()
                                pred_thresh = (predictions[..., 0] > args.threshold).sum()
                                pred_cell = (predictions[..., 1]).view(predictions.size(0), -1).mean(1, keepdim=True)
                                pred_cell = pred_cell.sum()

                                mae_prob = abs(pred_prob - targets)
                                mae_thresh = abs(pred_thresh - targets)
                                mae_cell = abs(pred_cell - targets)

                                mse = (pred_prob - targets)**2
                                
                                sum_mae_prob += mae_prob
                                sum_mae_th += mae_thresh
                                sum_mse += mse
                                sum_mae_cell += mae_cell

                                s = str((bi, 'MAE: ', mae_prob.item(), 'pred: ', pred_prob.item(), 'target: ', targets))
                                pbar.set_description(s)

                                s = '*Target {targets:.2f}\t *Pred_Prob {pred_prob:.4f}\t *Pred_Thresh {pred_thresh:.4f}\t *Pred_Cell {pred_cell:.4f}\t \
                                    *MAE_Prob {mae_prob:.4f}\t *MAE_Thresh {mae_thresh:.4f}\t *MAE_Cell {mae_cell:.4f} \n'.\
                                    format(targets=targets, pred_prob=pred_prob, pred_thresh=pred_thresh, pred_cell=pred_cell, \
                                        mae_prob=(pred_prob-targets), mae_thresh=(pred_thresh-targets), mae_cell=(pred_cell-targets))

                                f.writelines(s)

                                imgs = []
                                targets = []
                                target_chips = []
                                b_num = 0

        print(' * MAE_Prob {mae_prob:.3f} \n  * MAE_Thresh {mae_th:.3f} \n * MSE_Prob {mse:.3f} \n * MAE_Cell {mae_cell:.3f} \n '.\
            format(mae_prob=(sum_mae_prob/dataset_length).item(), mae_th=(sum_mae_th/dataset_length).item(), mse=(sum_mse/dataset_length).sqrt().item(), \
                mae_cell=(sum_mae_cell/dataset_length).item()))


def vis_image(args, img_big, imgs, target_chips, ni, nj, predictions):
    import torch.nn as nn 
    import matplotlib.pyplot as plt
    import cv2 

    upsample = nn.Upsample(scale_factor=4, mode='nearest')
    img = torch.zeros_like(img_big)
    pred_prob = torch.zeros_like(img_big[0,:,:])
    pred_cell = torch.zeros_like(img_big[0,:,:])
    target = torch.zeros_like(img_big[0,:,:])
    k = -1
    length = args.cell_size
    for ii in range(ni):  
        for jj in range(nj):  
            y2 = min((ii + 1) * length, img_big.shape[1])
            y1 = y2 - length
            x2 = min((jj + 1) * length, img_big.shape[2])
            x1 = x2 - length

            k += 1
            img[:, y1:y2, x1:x2] = imgs[k, ...]
            target[y1:y2, x1:x2] = target_chips[k, ...]
            pred_prob[y1:y2, x1:x2] = upsample(predictions[k, :, :, 0].unsqueeze(0).unsqueeze(0))
            pred_cell[y1:y2, x1:x2] = upsample(predictions[k, :, :, 1].unsqueeze(0).unsqueeze(0))

    img = img.permute(1, 2, 0).cpu()
    img = cv2.normalize(np.float32(img), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    img = img.astype(np.uint8)
    plt.subplot(3,2,1).imshow(img)
    plt.subplot(3,2,2).imshow(target)
    count_gt = target.sum()
    plt.title('People count: ' + str(count_gt.item()))
    pred_prob = pred_prob / 16
    plt.subplot(3,2,3).imshow(pred_prob)
    count_prob = pred_prob.sum()
    plt.title('Pred_prob: ' + str(count_prob.round().item()) + '  MAE: ' + str((count_prob-count_gt).round().item()))
    pred_thresh = pred_prob > args.threshold
    count_thresh = pred_thresh.sum()
    plt.subplot(3,2,4).imshow(pred_thresh)
    plt.title('Pred_thresh: ' + str(count_thresh.item()) + '  MAE: ' + str((count_thresh-count_gt).round().item()))
    plt.subplot(3,2,5).imshow(pred_cell)
    count_cell = (predictions[..., 1].view(predictions.size(0), -1).mean(1, keepdim=True)).sum()
    plt.title('Pred_cell: ' + str(count_cell.item()) + '  MAE: ' + str((count_cell-count_gt).round().item()))
    plt.show()


if __name__ == '__main__':
    test()
