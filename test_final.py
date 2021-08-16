
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
from model_final import CSRNet, ComputeLoss 
import torch 
import torchvision.transforms.functional as F
from torchvision import transforms
import PIL.Image as Image
from utils import zeropad, vis_input
import cv2
import torch.nn as nn 
import matplotlib.pyplot as plt




path = pathlib.Path(__file__).parent.absolute()
parser = argparse.ArgumentParser(description='RCVLab-AiimLab Crowd counting')

parser.add_argument('--model_desc', default='shanghaiA, 1/', help="Set model description")
parser.add_argument('--dataset_path', default='/media/mohsen/myDrive/datasets/ShanghaiTech_Crowd_Counting_Dataset', help='path to dataset')
parser.add_argument('--exp_sets', default='part_A_final/test_data')
parser.add_argument('--use_gpu', default=True, help="indicates whether or not to use GPU")
parser.add_argument('--device', default='0', type=str, help='GPU id to use.')
parser.add_argument('--checkpoint_path', default='./runs/weights', type=str, help='checkpoint path')
parser.add_argument('--log_dir', default='./runs/log', type=str, help='log dir')
parser.add_argument('--density', default=False, type=bool, help='using density map instead of head locations?')
parser.add_argument('--depth', default=True, type=bool, help='using depth?')

# MODEL
parser.add_argument('--model_file', default=path/'model.yaml')
parser.add_argument('--cell_size', default=128, type=int, help="cell size")
parser.add_argument('--threshold', default=0.3, help="[0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5], threshold for the classification output")

parser.add_argument('--best', default=False, type=bool, help='best or last saved checkpoint?') 
parser.add_argument('--vis_patch', default=False, type=bool, help='visualize the patches') 
parser.add_argument('--vis_image', default=True, type=bool, help='visualize the whole image') 
parser.add_argument('--vis_loc', default=False, type=bool, help='visualize the locations') 


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

    model = CSRNet()

    if CUDA:
        model = model.cuda()
    
    checkpoint = torch.load(args.checkpoint_path)

    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    compute_loss = ComputeLoss(model)

    imgs, imgs_depth, targets, target_bigs = [], [], [], []
    b_num = 0
    sum_mae_count_0, sum_mse, sum_mae_count_1, sum_mae_count_2, sum_mae_total_count, sum_best = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    dataset_length = len(img_paths)
    temp = 0
    
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
            if args.density:
                gt_file = h5py.File(img_path.replace('.jpg','.h5').replace('images','ground_truth'),'r')
            else:
                gt_file = h5py.File(img_path.replace('.jpg','_nofilter.h5').replace('images','ground_truth'),'r')
            target_big = np.asarray(gt_file['density'])
            target_big = torch.from_numpy(target_big)

            if args.depth:
                depth_file = h5py.File(img_path.replace('.jpg','.h5').replace('images','depth_resized_h5'),'r')
                img_big_depth = depth_file['depth'][:] 
                img_big_depth = torch.from_numpy(img_big_depth).unsqueeze(0)

            length_0 = img_big.size(1)
            length_1 = img_big.size(2)

            img_big_d = torch.zeros_like(img_big)
            img_big_depth = img_big_depth/65535.0
            img_big_d[0,:,:] = img_big_depth[0,:,:]
            img_big_d[1,:,:] = img_big_depth[0,:,:]
            img_big_d[2,:,:] = img_big_depth[0,:,:]

            #vis_input(img_big, groundtruth)
            #vis_input(img_chip, target_chip)

            if not args.density:
                coord = (target_big).nonzero(as_tuple=False)
                bxy = [[yb/length_0, xb/length_1] for (yb, xb) in coord]
                targets.append(torch.tensor(bxy))

            if args.depth:
                #img = torch.cat((img_big ,img_big_depth), dim=0)
                img = torch.clone(img_big)
                img_depth = torch.clone(img_big_d)
            else:
                img = torch.clone(img_big)

            imgs.append(img)
            imgs_depth.append(img_depth)

            target_bigs.append(target_big)

            if args.density:
                target_big = target_big.squeeze(0).numpy()
                target_big = cv2.resize(target_big,(target_big.shape[1]//8,target_big.shape[0]//8),interpolation = cv2.INTER_CUBIC)*64
                targets.append(torch.from_numpy(target_big))

            imgs = torch.stack(imgs, dim=0).squeeze(1)
            imgs_depth = torch.stack(imgs_depth, dim=0).squeeze(1)
            if not args.density:
                targets = [ti for ti in targets if len(ti) != 0]
                if not targets:
                    targets.append(torch.tensor([[-1, 0, 0, 0, 0]]))
                targets = torch.cat(targets)
            else:
                targets = torch.stack(targets, dim=0)

            target_bigs = torch.stack(target_bigs, dim=0).squeeze(1)

            if CUDA:
                imgs = imgs.cuda()
                imgs_depth = imgs_depth.cuda()
                targets = targets.cuda()

            with torch.no_grad():
                predictions0, predictions1, predictions2, total_count = model(imgs, imgs_depth, training=False)

                img_name = img_path.replace('.jpg','').replace('/media/mohsen/myDrive/datasets/ShanghaiTech_Crowd_Counting_Dataset/' + args.exp_sets + '/images/','')

                if args.vis_patch:
                    p_i = imgs.size(0)//4
                    vis_input(imgs[p_i, ...], target_bigs[p_i, ...], predictions0[p_i, ...], pred1=predictions1[p_i, :, :], pred2=predictions2[p_i, :, :])
                
                if args.vis_image:
                    vis_image(args, img_name, img_big, imgs, target_bigs, predictions0, predictions1, predictions2, vis_loc=False)

                if args.vis_loc:
                    vis_image(args, img_name, img_big, imgs, target_bigs, predictions0, predictions1, predictions2, vis_loc=True)

                if args.density:
                    targets = targets.sum()
                else:
                    targets = targets.shape[0]

                pred_count_0 = (predictions0).sum()
                pred_count_1 = (predictions1).sum()
                pred_count_2 = (predictions2).sum()
                total_count = total_count.sum()
                
                mae_count_0 = abs(pred_count_0 - targets)
                mae_count_1 = abs(pred_count_1 - targets)
                mae_count_2 = abs(pred_count_2 - targets)
                mae_count_T = abs(total_count - targets)

                '''
                if mae_count_0 > 20:
                    temp += 1
                    #print(temp, targets, pred_count_0.item(), mae_count_0.item(), img_name)
                '''

                mse = (pred_count_0 - targets)**2
                
                sum_mae_count_0 += mae_count_0
                sum_mse += mse
                sum_mae_count_1 += mae_count_1
                sum_mae_count_2 += mae_count_2
                sum_mae_total_count += mae_count_T

                sum_best += mae_count_0 if (mae_count_0 < mae_count_1) else mae_count_1

                s = str((bi, 'MAE: ', mae_count_0.item(), 'Pred: ', pred_count_0.item(), 'target: ', targets))
                #pbar.set_description(s)

                s = '*Target {targets:.0f}\t *Pred_0 {pred_0:.3f}\t *Pred_1 {pred_1:.3f}\t *Pred_2 {pred_2:.3f}\t *MAE_0 {mae_0:.3f}\t *MAE_1 {mae_1:.3f}\t *MAE_2 {mae_2:.3f} \n'.\
                    format(targets=targets, pred_0=pred_count_0, pred_1=pred_count_1, pred_2=pred_count_2, \
                        mae_0=(pred_count_0-targets), mae_1=(pred_count_1-targets), mae_2=(pred_count_2-targets))

                f.writelines(s)

                imgs = []
                imgs_depth = []
                targets = []
                target_bigs = []


    print(' * MAE_Count_0 {mae_count_0:.3f} \n * MSE {mse:.3f} \n * MAE_Count_1 {mae_count_1:.3f} \n * MAE_Count_2 {mae_count_2:.3f} \n * MAE_Count_total {mae_count_t:.3f} \n '.\
        format(mae_count_0=(sum_mae_count_0/dataset_length).item(), mse=(sum_mse/dataset_length).sqrt().item(), \
            mae_count_1=(sum_mae_count_1/dataset_length).item(), mae_count_2=(sum_mae_count_2/dataset_length).item(), mae_count_t=(sum_mae_total_count/dataset_length).item()))
    
    print(' * MAE_Best {mae_best:.3f}'.format(mae_best=(sum_best/dataset_length).item()))



def vis_locations(image, loc_im, gt):
    from math import sqrt
    from skimage import data
    from skimage.feature import blob_dog, blob_log, blob_doh
    from skimage.color import rgb2gray

    loc_im = loc_im.numpy().astype('double')

    blobs_log = blob_log(loc_im, max_sigma=10, num_sigma=10, threshold=.009)

    # Compute radii in the 3rd column.
    blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)


    blobs_list = [blobs_log]
    num_blobs = [len(blobs_log)]
    
    colors = ['yellow', 'lime', 'red']
    titles = ['GT:' + str(gt) + 'Num pred locations' + str(num_blobs[0])]
    sequence = zip(blobs_list, colors, titles)

    fig, axes = plt.subplots(1, 1, figsize=(10, 10), sharex=True, sharey=True)
    #ax = axes.ravel()
    ax = []
    ax.append(axes)

    for idx, (blobs, color, title) in enumerate(sequence):
        ax[idx].set_title(title)
        ax[idx].imshow(image)
        #ax[idx].imshow(loc_im)
        for blob in blobs:
            y, x, r = blob
            c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
            ax[idx].add_patch(c)
        ax[idx].set_axis_off()

    plt.tight_layout()
    plt.show()

    
    num_blobs = [len(blobs_log)]
    return num_blobs


def vis_image(args, img_name, img_big, imgs, target_chips, predictions0, predictions1, predictions2=None, vis_loc=False):

    in_size = imgs.size(2)
    if img_big.size(1) < in_size or img_big.size(2) < in_size:
        return
    
    out_size0 = predictions0.size(1) 
    out_size1 = predictions1.size(1) 
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
    
    if imgs.size(1) > 3:
        imgs = imgs[:, :3, :, :]

    img = imgs.squeeze(0)
    target_im = target_chips.squeeze(0)
    pred_count_im0 = upsample0(predictions0.unsqueeze(0)).squeeze(0).squeeze(0).cpu()
    pred_count_im1 = upsample1(predictions1.unsqueeze(0)).squeeze(0).squeeze(0).cpu()
    if predictions2 is not None:
        pred_count_im2 = upsample2(predictions2.unsqueeze(0)).squeeze(0).squeeze(0).cpu()


    target = target_im.sum()
    
    img = img.permute(1, 2, 0).cpu()
    img = cv2.normalize(np.float32(img), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    img = img.astype(np.uint8)

    if vis_loc:
        loc = vis_locations(img, pred_count_im1, target.item())
        return loc

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
    plt.imshow(pred_count_im0)#, alpha=0.9)
    #pred_count, _ = predictions0.view(predictions0.size(0), -1).sum(1, keepdim=True)
    #pred_count = pred_count.sum() 
    pred_count0 = predictions0.sum() 
    plt.title('Pred: ' + str(pred_count0.round().item()) + '  MAE: ' + str((pred_count0-target).round().item()))
    
    plt.subplot(2,3,5)
    plt.imshow(img)
    plt.imshow(pred_count_im1)#, alpha=0.9)
    pred_count1 = predictions1.sum()
    plt.title('Pred: ' + str(pred_count1.round().item()) + '  MAE: ' + str((pred_count1-target).round().item()))

    if predictions2 is not None:
        plt.subplot(2,3,6)
        plt.imshow(img)
        plt.imshow(pred_count_im2)#, alpha=0.9)
        pred_count2 = predictions2.sum()     
        plt.title('Pred: ' + str(pred_count2.round().item()) + '  MAE: ' + str((pred_count2-target).round().item()))

    plt.show()



if __name__ == '__main__':
    test()
