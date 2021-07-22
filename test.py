
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
from model import Model, CSRNet



path = pathlib.Path(__file__).parent.absolute()
parser = argparse.ArgumentParser(description='RCVLab-AiimLab Crowd counting')

parser.add_argument('--model_desc', default='shanghaiA, cell128, lr6_FC2/', help="Set model description")
parser.add_argument('--dataset_path', default='/media/mohsen/myDrive/datasets/ShanghaiTech_Crowd_Counting_Dataset', help='path to dataset')
parser.add_argument('--exp_sets', default='part_B_final/test_data')
parser.add_argument('--use_gpu', default=True, help="indicates whether or not to use GPU")
parser.add_argument('--device', default='0', type=str, help='GPU id to use.')
parser.add_argument('--checkpoint_path', default='../runs/weights', type=str, help='checkpoint path')
parser.add_argument('--log_dir', default='../runs/log', type=str, help='log dir')
parser.add_argument('--depth', default=False, type=bool, help='using depth?')

# MODEL
parser.add_argument('--model_file', default=path/'model.yaml')
parser.add_argument('--cell_size', default=128, type=int, help="cell size")
parser.add_argument('--threshold', default=[0.9], help="[0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5], threshold for the classification output")

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

    model = CSRNet(in_size=args.cell_size)

    if CUDA:
        model = model.cuda()
    
    checkpoint = torch.load(args.checkpoint_path)

    model.load_state_dict(checkpoint['state_dict'])

    for thresh in args.threshold:
        print('thresh: ', thresh)
        imgs, targets, target_chips = [], [], []
        length = args.cell_size
        b_num = 0
        sum_mae_prob, sum_mae_th, sum_mse, sum_mae_cell, sum_best = 0.0, 0.0, 0.0, 0.0, 0.0
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

                if args.depth:
                    depth_file = h5py.File(img_path.replace('.jpg','.h5').replace('images','depth_resized_h5'),'r')
                    img_big_depth = depth_file['depth'][:] 

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
                        
                        if args.depth:
                            img_chip_depth = img_big_depth[y1:y2, x1:x2]
                            img_chip_depth  = zeropad(img_chip_depth, length - img_chip_depth.shape[0], length - img_chip_depth.shape[1])
                            img_chip_depth = torch.from_numpy(img_chip_depth).unsqueeze(0)
                            assert img_chip_depth.shape[1] == img_chip_depth.shape[2] == length, 'image size error'

                        target_chip = groundtruth[y1:y2, x1:x2]
                        target_chip = zeropad(target_chip, length - target_chip.shape[0], length - target_chip.shape[1], target=True)
                        target_chip = torch.from_numpy(target_chip)
                        
                        assert img_chip.shape[1] == img_chip.shape[2] == length, 'image size error'
                        assert target_chip.shape[0] == target_chip.shape[1] == length, 'target size error'

                        coord = (target_chip).nonzero(as_tuple=False)
                        bxy = [[b_num, yb/length, xb/length, i, j] for (yb, xb) in coord]
                        targets.append(torch.tensor(bxy))

                        if args.depth:
                            img = torch.cat((img_chip ,img_chip_depth), dim=0)
                        else:
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
                                predictions0 = model(imgs, training=False)

                                img_name = img_path.replace('.jpg','').replace('/media/mohsen/myDrive/datasets/ShanghaiTech_Crowd_Counting_Dataset/' + args.exp_sets + '/images/','')
                
                                if args.vis_image:
                                    vis_image(args, img_name, img_big, imgs, target_chips, ni, nj, predictions0, predictions0, thresh)
                                
                                if args.vis_patch:
                                    im_i = imgs.size(0)//4
                                    vis_input(imgs[im_i, ...], target_chips[im_i, ...], predicted=predictions0[im_i, :, :], thresholded=predictions0[im_i, :, :] > thresh)
                            
                                targets = targets.shape[0]
                                #pred_prob = predictions0.view(predictions0.size(0), -1).mean(1, keepdim=True)
                                #pred_prob = pred_prob.sum()
                                pred_prob = predictions0.sum()
                                pred_thresh = (predictions0 > thresh).sum()
                                pred_cell = predictions0.sum()

                                mae_prob = abs(pred_prob - targets)
                                mae_thresh = abs(pred_thresh - targets)
                                mae_cell = abs(pred_cell - targets)

                                mse = (pred_prob - targets)**2
                                
                                sum_mae_prob += mae_prob
                                sum_mae_th += mae_thresh
                                sum_mse += mse
                                sum_mae_cell += mae_cell

                                sum_best += mae_prob if mae_prob < mae_cell else mae_cell

                                s = str((bi, 'MAE: ', mae_prob.item(), 'pred: ', pred_prob.item(), 'target: ', targets))
                                pbar.set_description(s)

                                s = '{img:}\t *Target {targets:.0f}\t *Pred_Prob {pred_prob:.3f}\t *Pred_Thresh {pred_thresh:.3f}\t *Pred_Cell {pred_cell:.3f}\t *MAE_Prob {mae_prob:.3f}\t *MAE_Thresh {mae_thresh:.3f}\t *MAE_Cell {mae_cell:.3f} \n'.\
                                    format(img=img_name, targets=targets, pred_prob=pred_prob, pred_thresh=0, pred_cell=0, \
                                        mae_prob=(pred_prob-targets), mae_thresh=(0), mae_cell=(0))

                                f.writelines(s)

                                imgs = []
                                targets = []
                                target_chips = []
                                b_num = 0

        print(' * MAE_Prob {mae_prob:.3f} \n  * MAE_Thresh {mae_th:.3f} \n * MSE_Prob {mse:.3f} \n * MAE_Cell {mae_cell:.3f} \n '.\
            format(mae_prob=(sum_mae_prob/dataset_length).item(), mae_th=(0), mse=(sum_mse/dataset_length).sqrt().item(), \
                mae_cell=(0)))
        
        print(' * MAE_Best {mae_best:.3f}'.format(mae_best=(sum_best/dataset_length).item()))


def vis_image(args, img_name, img_big, imgs, target_chips, ni, nj, predictions0, predictions1, thresh):
    import torch.nn as nn 
    import matplotlib.pyplot as plt
    import cv2 

    in_size = imgs.size(2)
    out_size = predictions0.size(1)

    upsample = nn.Upsample(scale_factor=in_size//out_size, mode='nearest')
    upsample1 = nn.Upsample(scale_factor=in_size, mode='nearest')
    img = torch.zeros_like(img_big)
    pred_prob = torch.zeros_like(img_big[0,:,:])
    pred_cell = torch.zeros_like(img_big[0,:,:])
    target = torch.zeros_like(img_big[0,:,:])
    k = -1
    length = args.cell_size
    if imgs.size(1) > 3:
        imgs = imgs[:, :3, :, :]
    for ii in range(ni):  
        for jj in range(nj):  
            y2 = min((ii + 1) * length, img_big.shape[1])
            y1 = y2 - length
            x2 = min((jj + 1) * length, img_big.shape[2])
            x1 = x2 - length

            k += 1
            img[:, y1:y2, x1:x2] = imgs[k, ...]
            target[y1:y2, x1:x2] = target_chips[k, ...]
            #pred_prob[y1:y2, x1:x2] = upsample(predictions0[k, :, :].unsqueeze(0).unsqueeze(0))
            pred_prob[y1:y2, x1:x2] = upsample1(predictions1[k, :].unsqueeze(0).unsqueeze(0))

    img = img.permute(1, 2, 0).cpu()
    img = cv2.normalize(np.float32(img), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    img = img.astype(np.uint8)
    plt.subplot(3,2,1).imshow(img)
    count_gt = target.sum()
    plt.title(img_name + ' - GT count: ' + str(count_gt.item()))
    plt.subplot(3,2,2).imshow(target)
    plt.title('GT')
    #pred_prob = pred_prob / 64
    plt.subplot(3,2,3).imshow(pred_prob)
    #count_prob = predictions0.view(predictions0.size(0), -1).sum(1, keepdim=True)
    count_prob = pred_prob.sum() / (in_size**2)
    plt.title('Pred count: ' + str(count_prob.round().item()) + '  MAE: ' + str((count_prob-count_gt).round().item()))
    #pred_thresh = pred_prob > thresh
    #count_thresh = pred_thresh.sum() / 64
    #plt.subplot(3,2,4).imshow(pred_thresh)
    #plt.title('Pred_thresh: ' + str(count_thresh.item()) + '  MAE: ' + str((count_thresh-count_gt).round().item()))
    #plt.subplot(3,2,5).imshow(pred_cell)
    #count_cell = predictions1.sum()      
    #plt.title('Pred_cell: ' + str(count_cell.round().item()) + '  MAE: ' + str((count_cell-count_gt).round().item()))
    plt.show()


if __name__ == '__main__':
    test()
