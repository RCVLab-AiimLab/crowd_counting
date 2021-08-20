
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
from model import CSRNet, ComputeLoss 
import torch 
import torchvision.transforms.functional as F
from torchvision import transforms
import PIL.Image as Image
from utils import zeropad, vis_input



path = pathlib.Path(__file__).parent.absolute()
parser = argparse.ArgumentParser(description='RCVLab-AiimLab Crowd counting')

parser.add_argument('--model_desc', default='shanghaiA, 128, 7, trial14/', help="Set model description")
parser.add_argument('--dataset_path', default='/home/16amf8/data/datasets/ShanghaiTech', help='path to dataset')
parser.add_argument('--exp_sets', default='part_A_final/test_data')
parser.add_argument('--use_gpu', default=True, help="indicates whether or not to use GPU")
parser.add_argument('--device', default='0', type=str, help='GPU id to use.')
parser.add_argument('--checkpoint_path', default='/home/16amf8/data/work_dirs/crowd_counting_shanghai_a', type=str, help='checkpoint path')
parser.add_argument('--log_dir', default='/home/16amf8/data/work_dirs/crowd_counting_shanghai_a/log', type=str, help='log dir')
parser.add_argument('--depth', default=False, type=bool, help='using depth?')

# MODEL
parser.add_argument('--model_file', default=path/'model.yaml')
parser.add_argument('--cell_size', default=128, type=int, help="cell size")
parser.add_argument('--threshold', default=0.01, help="[0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5], threshold for the classification output")

parser.add_argument('--best', default=True, type=bool, help='best or last saved checkpoint?') 
parser.add_argument('--vis_patch', default=False, type=bool, help='visualize the patches') 
parser.add_argument('--vis_image', default=True, type=bool, help='visualize the whole image') 
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
    model.eval()

    imgs, targets, target_chips = [], [], []
    length = args.cell_size
    b_num = 0
    sum_mae_count_0, sum_mse, sum_mae_count_1, sum_mae_count_2, sum_best = 0.0, 0.0, 0.0, 0.0, 0.0
    one = 0
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

                    #vis_input(img_big, groundtruth)
                    #vis_input(img_chip, target_chip)

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
                        if not targets:
                            targets.append(torch.tensor([[-1, 0, 0, 0, 0]]))
                        targets = torch.cat(targets)

                        target_chips = torch.stack(target_chips, dim=0).squeeze(1)

                        if CUDA:
                            imgs = imgs.cuda()
                            targets = targets.cuda()

                        with torch.no_grad():
                            predictions0, predictions1, predictions2 = model(imgs, training=False)
                            predictions2 = predictions2[0]

                            img_name = img_path.replace('.jpg','').replace('/home/16amf8/data/datasets/ShanghaiTech/' + args.exp_sets + '/images/','')
            
                            if args.vis_image:
                                vis_image(args, img_name, img_big, imgs, target_chips, ni, nj, predictions0, predictions1, predictions2, thresh=args.threshold)
                            
                            if args.vis_patch:
                                p_i = imgs.size(0)//4
                                vis_input(imgs[p_i, ...], target_chips[p_i, ...], predictions0[p_i, ...], pred1=predictions1[p_i, :, :], pred2=predictions2[p_i, :, :])
                                                    
                            targets = targets.shape[0]
                            pred_count_0 = predictions0.sum()
                            pred_count_1 = predictions1.sum()
                            pred_count_2 = predictions2.sum()

                            mae_count_0 = abs(pred_count_0 - targets)
                            mae_count_1 = abs(pred_count_1 - targets)
                            mae_count_2 = abs(pred_count_2 - targets)

                            mse = (pred_count_0 - targets)**2
                            
                            sum_mae_count_0 += mae_count_0
                            sum_mse += mse
                            sum_mae_count_1 += mae_count_1
                            sum_mae_count_2 += mae_count_2

                            #sum_best += (abs(density - targets))
                            if mae_count_0 < mae_count_1 and mae_count_0 < mae_count_2:
                                sum_best += mae_count_0
                            elif mae_count_1 < mae_count_0 and mae_count_1 < mae_count_2:
                                sum_best += mae_count_1
                                one += 1
                            elif mae_count_2 < mae_count_0 and mae_count_2 < mae_count_1:
                                sum_best += mae_count_2
                            

                            #sum_best += mae_count_0 if mae_count_0 < mae_count_1 else mae_count_1
                            

                            s = str((bi, 'MAE: ', mae_count_0.item(), 'Pred: ', pred_count_0.item(), 'target: ', targets))
                            #pbar.set_description(s)

                            s = '*Target {targets:.0f}\t *Pred_0 {pred_0:.3f}\t *Pred_1 {pred_1:.3f}\t *Pred_2 {pred_2:.3f}\t *MAE_0 {mae_0:.3f}\t *MAE_1 {mae_1:.3f}\t *MAE_2 {mae_2:.3f} \n'.\
                                format(targets=targets, pred_0=pred_count_0, pred_1=pred_count_1, pred_2=pred_count_2, \
                                    mae_0=(pred_count_0-targets), mae_1=(pred_count_1-targets), mae_2=(pred_count_2-targets))

                            f.writelines(s)

                            imgs = []
                            targets = []
                            target_chips = []
                            b_num = 0

    print(' * MAE_Count_0 {mae_count_0:.3f} \n * MSE {mse:.3f} \n * MAE_Count_1 {mae_count_1:.3f} \n * MAE_Count_2 {mae_count_2:.3f} \n '.\
        format(mae_count_0=(sum_mae_count_0/dataset_length).item(), mse=(sum_mse/dataset_length).sqrt().item(), \
            mae_count_1=(sum_mae_count_1/dataset_length).item(), mae_count_2=(sum_mae_count_2/dataset_length).item()))
   
    print(sum_best)
    print(dataset_length)
    print(sum_best/dataset_length)
    print(' * MAE_Best {mae_best:.3f}'.format(mae_best=(sum_best/dataset_length).item()))


def vis_image(args, img_name, img_big, imgs, target_chips, ni, nj, predictions0, predictions1, predictions2=None, thresh=0.1):
    
    import torch.nn as nn 
    import matplotlib.pyplot as plt
    import cv2 

    in_size = imgs.size(2)
    if img_big.size(1) < in_size or img_big.size(2) < in_size:
        return
    
    out_size0 = predictions0.size(1) #
    out_size1 = predictions1.size(1) #
    out_size2 = predictions2.size(1) if predictions2 is not None else 0

    upsample0 = nn.Upsample(scale_factor=in_size//out_size0, mode='nearest')
    upsample1 = nn.Upsample(scale_factor=in_size//out_size1, mode='nearest')
    upsample2 = nn.Upsample(scale_factor=in_size//out_size2, mode='nearest') if predictions2 is not None else None
    upsample3 = nn.Upsample(scale_factor=in_size, mode='nearest')
    img = torch.zeros_like(img_big)
    pred_count_im0 = torch.zeros_like(img_big[0,:,:])
    pred_count_im1 = torch.zeros_like(img_big[0,:,:])
    pred_count_im2 = torch.zeros_like(img_big[0,:,:])
    target_im = torch.zeros_like(img_big[0,:,:])
    target_count_im = torch.zeros_like(img_big[0,:,:])
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
            target_im[y1:y2, x1:x2] = target_chips[k, ...]
            #target_count_im[y1:y2, x1:x2] = upsample3(target_chips[k, ...].sum().unsqueeze(0).unsqueeze(0).unsqueeze(0))
            #pred_count_im[y1:y2, x1:x2] = upsample1(predictions0[k, :].unsqueeze(0).unsqueeze(0))
            pred_count_im0[y1:y2, x1:x2] = upsample0(predictions0[k, :, :].unsqueeze(0).unsqueeze(0))
            pred_count_im1[y1:y2, x1:x2] = upsample1(predictions1[k, :, :].unsqueeze(0).unsqueeze(0))
            if predictions2 is not None:
                pred_count_im2[y1:y2, x1:x2] = predictions2[k, 2, :, :]
                #pred_count_im2[y1:y2, x1:x2] = upsample2(predictions2[k, :, :].unsqueeze(0).unsqueeze(0))

    target = target_im.sum()
    img = img.permute(1, 2, 0).cpu()
    img = cv2.normalize(np.float32(img), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    img = img.astype(np.uint8)
    plt.subplot(2,3,1).imshow(img)
    plt.title(img_name + ' - GT count: ' + str(target.item()))
    
    kernel = np.ones((5,5), np.uint8)
    target_im = cv2.dilate(np.uint8(target_im), kernel, iterations=1)
    plt.subplot(2,3,2)
    plt.imshow(img)
    plt.imshow(target_im, alpha=0.7)
    plt.title('GT')
    
    plt.subplot(2,3,4)
    plt.imshow(img)
    plt.imshow(pred_count_im0, alpha=0.9)
    #pred_count, _ = predictions0.view(predictions0.size(0), -1).mode(1, keepdim=True)
    #pred_count = pred_count.sum() 
    pred_count0 = predictions0.sum() 
    plt.title('Pred: ' + str(pred_count0.round().item()) + '  MAE: ' + str((pred_count0-target).round().item()))
    
    plt.subplot(2,3,5)
    plt.imshow(img)
    plt.imshow(pred_count_im1, alpha=0.9)
    pred_count1 = predictions1.sum()     
    plt.title('Pred: ' + str(pred_count1.round().item()) + '  MAE: ' + str((pred_count1-target).round().item()))

    if predictions2 is not None:
        plt.subplot(2,3,6)
        plt.imshow(img)
        plt.imshow(pred_count_im2, alpha=0.9)
        pred_count2 = predictions2.sum()     
        plt.title('Pred: ' + str(pred_count2.round().item()) + '  MAE: ' + str((pred_count2-target).round().item()))

    #plt.subplot(3,2,5)
    #plt.imshow(target_count_im)
    #plt.title('GT')

    '''
    plt.subplot(3,2,5)
    plt.imshow(img)
    plt.imshow(pred_loc_im > thresh, alpha=0.7)
    pred_thresh = (predictions1 > thresh).sum().item()
    plt.title('Pred: ' + str(pred_thresh) + ' MAE: ' + str((pred_thresh-target).round().item()))

    
    plt.subplot(3,2,6)
    plt.imshow(img)
    plt.imshow(abs(pred_cell_im + pred_loc_im), alpha=0.7)
    plt.title('Pred Cell - Loc: ' + str((abs(pred_cell_im + pred_loc_im)>1.7).sum()))
    '''

    plt.show()


if __name__ == '__main__':
    test()


