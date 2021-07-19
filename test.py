
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

parser.add_argument('--model_desc', default='UCF-QNRF, darknet, countInCell, lr=1e-5/', help="Set model description")
parser.add_argument('--dataset_path', default='/home/16amf8/data/datasets/UCF-QNRF_ECCV18', help='path to dataset')
parser.add_argument('--exp_sets', default='QNRF')
parser.add_argument('--use_gpu', default=True, help="indicates whether or not to use GPU")
parser.add_argument('--device', default='0', type=str, help='GPU id to use.')
parser.add_argument('--checkpoint_path', default='/home/16amf8/data/work_dirs/crowd_counting_UCF-QNRF', type=str, help='checkpoint path')
parser.add_argument('--log_dir', default='/home/16amf8/data/work_dirs/crowd_counting_UCF-QNRF/log', type=str, help='log dir')

# MODEL
parser.add_argument('--model_file', default='model.yaml')
parser.add_argument('--cell_size', default=128, type=int, help="cell size")
parser.add_argument('--threshold', default=0.5, type=int, help="threshold for the classification output")

parser.add_argument('--best', default=False, type=bool, help='best or last saved checkpoint?') 
parser.add_argument('--vis_patch', default=False, type=bool, help='visualize the patches') 
parser.add_argument('--vis_image', default=False, type=bool, help='visualize the whole image') 
parser.add_argument('--prob_map', default=True, type=bool, help='using threshold or probability map?') 




def test():
    args = parser.parse_args()
    args.log_dir += ('/'+args.model_desc)
    
    if args.use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device
        torch.cuda.manual_seed(time.time())
        CUDA =True
    else:
        CUDA = False

    if ('NWPU' in args.exp_sets):
        img_paths = []
        with open('datasets/NWPU/val.json') as f:
            data = json.load(f)
        img_paths = data
    elif ('QNRF' in args.exp_sets):
        img_paths = []
        with open('datasets/UCF-QNRF/Test.json') as f:
            data = json.load(f)
        img_paths = data
    else:
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

    imgs, targets, target_chips = [], [], []
    length = args.cell_size
    b_num = 0
    sum_mae, sum_mse, sum_mae_num = 0, 0, 0
    dataset_length = len(img_paths)
    
    pbar = enumerate(img_paths)
    pbar = tqdm(pbar, total=len(img_paths))

    with open(args.log_dir + 'results_test.txt', 'w') as f:
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

                    coord = (target_chip).nonzero(as_tuple=False)
                    bxy = [[b_num, yb/length, xb/length] for (yb, xb) in coord]
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
                                import torch.nn as nn 
                                import matplotlib.pyplot as plt
                                import cv2 
                                upsample = nn.Upsample(scale_factor=4, mode='nearest')
                                img_reconstracted = torch.zeros_like(img_big)
                                pred_reconstracted = torch.zeros_like(img_big[0,:,:])
                                target_reconstracted = torch.zeros_like(img_big[0,:,:])
                                k = -1
                                for ii in range(ni):  
                                    for jj in range(nj):  
                                        y2 = min((ii + 1) * length, img_big.shape[1])
                                        y1 = y2 - length
                                        x2 = min((jj + 1) * length, img_big.shape[2])
                                        x1 = x2 - length

                                        k += 1
                                        img_reconstracted[:, y1:y2, x1:x2] = imgs[k, ...]
                                        target_reconstracted[y1:y2, x1:x2] = target_chips[k, ...]
                                        pred_reconstracted[y1:y2, x1:x2] = upsample(predictions[k, 0, :, :].unsqueeze(0).unsqueeze(0))

                                im = img_reconstracted.permute(1, 2, 0).cpu()
                                im = cv2.normalize(np.float32(im), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                                im = im.astype(np.uint8)
                                plt.subplot(1,2,1).imshow(im)
                                count = target_reconstracted.sum()
                                plt.title('People count: ' + str(count.item()))
                                plt.subplot(1,2,2).imshow(pred_reconstracted)
                                count = (pred_reconstracted).sum()
                                #count = (predictions[:, 1, ...].view(predictions.size(0), -1).mean(1, keepdim=True)).sum()
                                plt.title('Predicted count: ' + str(count.item()))
                                #plt.subplot(1,4,3).imshow(target_reconstracted)
                                #plt.subplot(1,4,4).imshow(pred_reconstracted > args.threshold)
                                plt.show()
                            

                            if args.vis_patch:
                                im_i = imgs.size(0)//4
                                vis_input(imgs[im_i, ...], target_chips[im_i, ...], predicted=predictions[im_i, 0, ...], thresholded=predictions[im_i, 0, ...] > args.threshold)
                            
                            if not args.prob_map:
                                predictions[:, 0, ...] = predictions[:, 0, ...] > args.threshold
                            
                            pred = (predictions[:, 0, ...]).sum()
                            predByNum = predictions[:, 1, ...].view(predictions.size(0), -1).mean(1, keepdim=True)
                            predByNum = predByNum.sum()

                            targets = targets.shape[0]
                            mae = abs(pred - targets)
                            mae_num = abs(predByNum - targets)
                            mse = (pred -targets)**2
                            sum_mae += mae
                            sum_mse += mse
                            sum_mae_num += mae_num

                            s = str((bi, 'MAE: ', mae.item(), 'pred: ', pred.item(), 'target: ', targets))
                            pbar.set_description(s)

                            s = '*Target {targets:.2f}\t *Pred_prob {pred:.4f}\t *Pred_num {predByNum:.4f}\t MAE_prob {mae:.4f}\t *MAE_num {maeByNum:.4f}\n'.format(targets=targets, pred=pred, predByNum=predByNum, mae=(pred-targets), maeByNum=(predByNum-targets))
                            f.writelines(s)

                            imgs = []
                            targets = []
                            target_chips = []
                            b_num = 0

    print(' * MAE {mae:.3f} \n * MSE {mse:.3f} \n * MAEbyNum {mae_num:.3f}'.format(mae=(sum_mae/dataset_length).item(), mse=(sum_mse/dataset_length).sqrt().item(), mae_num=(sum_mae_num/dataset_length).item()))



if __name__ == '__main__':
    test()
