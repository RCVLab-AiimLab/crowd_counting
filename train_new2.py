# this file does chopping while training. 
## target needs not be read.
import os
import argparse
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch.autograd import Variable
import pathlib
from model_new import Model, ComputeLoss, CSRNet
from dataset_new2 import listDataset, get_list
from tqdm import tqdm
import math
from utils import save_checkpoint, AverageMeter, vis_input, zeropad
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import glob
import random

path = pathlib.Path(__file__).parent.absolute()
parser = argparse.ArgumentParser(description='RCVLab-AiimLab Crowd counting')

# GENERAL
parser.add_argument('--model_desc', default='test/', help="Set model description")
# parser.add_argument('--model_desc', default='shanghaiA_cell__128_128_64__lr10-5_ep50_nodepth_shuffle2/', help="Set model description")
parser.add_argument('--train_json', default=os.path.join(path,'datasets/shanghai/part_A_train.json'), help='path to train json')
parser.add_argument('--val_json', default=os.path.join(path,'datasets/shanghai/part_A_test.json'), help='path to test json')
parser.add_argument('--use_pre', default=False, type=bool, help='use the pretrained model?')
parser.add_argument('--use_gpu', default=True, action="store_false", help="Indicates whether or not to use GPU")
parser.add_argument('--device', default='6', type=str, help='GPU id to use.')
parser.add_argument('--checkpoint_path', default=os.path.join(path,'runs/weights'), type=str, help='checkpoint path')
parser.add_argument('--log_dir', default=os.path.join(path,'runs/log'), type=str, help='log dir')
parser.add_argument('--exp', default='shanghai', type=str, help='set dataset for training experiment')
parser.add_argument('--depth', default=False, type=bool, help='using depth?')
# MODEL
parser.add_argument('--model_file', default=path/'model.yaml')
parser.add_argument('--cell_size', default=[128,128,64], type=int, help="cell size")
parser.add_argument('--threshold', default=0.01, type=int, help="threshold for the classification output")

# TRAINING
parser.add_argument('--stack_size', default=16, type=int)
parser.add_argument('--batch_size', default=512, type=int)
parser.add_argument('--epochs', default=50, type=int, help="Number of epochs to train for")
parser.add_argument('--workers', default=4, type=int, help="Number of workers in loading dataset")
parser.add_argument('--start_epoch', default=0, type=int, help="start_epoch")
parser.add_argument('--vis', default=False, type=bool, help='visualize the inputs') 
parser.add_argument('--lr0', default=0.00001, type=float, help="initial learning rate")
parser.add_argument('--weight_decay', default=0.0005, type=float, help="weight_decay")
parser.add_argument('--momentum', default=0.937, type=float, help="momentum")
parser.add_argument('--adam', default=False, type=bool, help='use torch.optim.Adam() optimizer') 


def train(args, model, optimizer, tb_writer, CUDA, val_list, val_list_depth, num_imgs_train, num_imgs_val):

    # print(args.cell_size)   
    # files = glob.glob(os.path.normpath(os.path.join("crowd_csr_grid/img/",'*')))
    # for f in files:
    #     os.remove(f)
    for epoch in range(args.start_epoch, args.epochs):
        
        train_list, train_list_depth = get_list(args)
        train_loader = torch.utils.data.DataLoader(listDataset(train_list,
                       train_list_depth,
                       shuffle=False,
                       depth=args.depth,
                    #    transform1=transforms.Compose([transforms.ToTensor(),]), 
                    #    transform2=transforms.Compose([transforms.ToTensor(),]), 
                    #    transform1=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),]), 
                    #    transform2=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.502], std=[0.291]),]), 
                       train=True, 
                       seen=0,
                       batch_size=1,
                       num_workers=args.workers))

        losses = AverageMeter()
        losses_obj = AverageMeter()
        losses_cell = AverageMeter()
        losses_neighbor = AverageMeter()
        data_time = AverageMeter()
        batch_time = AverageMeter()
        end = time.time()

        model.train()

        pbar = enumerate(train_loader)
        pbar = tqdm(pbar, total=len(train_loader))  # progress bar

        optimizer.zero_grad()

        imgs, targets = [], []
        length = args.cell_size
        b_num = 0
        counter = 0
        k = 0
        imgs, targets = [], []
        # print('before')
        for bi, (img, target, bxy, l, depth) in pbar:  # batch ----------
            # print('I entered the enumeration')
            
            data_time.update(time.time() - end)
            # print('target',target.shape)
            target = target.squeeze(0)
            img = img.squeeze(0)
            bxy = bxy.squeeze(0)
            print(bxy)
            # l = l.squeeze(0)
            ####### edit i j later
            # i is the x index of patch
            # j is the y index of patch
            # print('bxy',bxy)
            targets.append(torch.tensor(bxy))
            # print('img',img.shape)

            if args.depth:
                img = torch.cat((img ,depth), dim=1)
            # else:
            #     img = torch.clone(img)

            img = Variable(img)
            imgs.append(img)

            b_num += 1
            # print('here')
            # print('k',k)
            # print('here')
            k = k+1
            #if b_num >= args.batch_size:
            if k == (args.stack_size):
                # print('I passed if')
                imgs = torch.stack(imgs, dim=0).squeeze(1)
                targets = [ti for ti in targets if len(ti) != 0]
                if not targets:
                    # print('Im here')
                    # print('*'*100)
                    targets.append(torch.tensor([[-1, 0, 0]]))
                    # print('*'*1000)
                targets = torch.cat(targets)
                # print('targets')
                if CUDA:
                    imgs = imgs.cuda()
                    # print('imgs',imgs.shape)
                    targets = targets.cuda()
                    # print('tar',targets.shape)
                
                if epoch <= 1:
                    tb_writer.add_graph(torch.jit.trace(model, imgs, strict=False), [])
                pred0 = model(imgs, training=True)  # forward
                print('pred',pred0)
                compute_loss = ComputeLoss(model, in_size=l)
                loss, lcell = compute_loss(pred0, targets) 

                losses.update(loss.item(), imgs.size(0))
                losses_cell.update(lcell.item(), imgs.size(0))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()    
                
                batch_time.update(time.time() - end)
                end = time.time()

                s = ('Epoch [{0}][{1}/{2}] '
                    'Time/b {batch_time.val:.2f} ({batch_time.avg:.2f}) '
                    'Loss {loss.val:.4f} ({loss.avg:.3f}) '
                    .format(epoch, bi, len(train_loader), batch_time=batch_time, loss=losses))
                
                pbar.set_description(s)

                imgs = []
                targets = []
                b_num = 0
                k = 0
            # end batch ------------

        pred_prob, _, _, val_losses = validate(args, val_list, val_list_depth, model, CUDA, num_imgs_val)

        is_best = pred_prob < args.best_pred
        args.best_pred = min(pred_prob, args.best_pred)
        print(' * Best MAE {mae:.3f} '.format(mae=args.best_pred))
        
        save_checkpoint({
            'epoch': epoch,
            'arch': args.checkpoint_path,
            'state_dict': model.state_dict(),
            'best_pred': args.best_pred,
            'optimizer' : optimizer.state_dict(),}, is_best, args.checkpoint_path)
        
        tb_writer.add_scalar('train loss/total', losses.avg, epoch)
        #tb_writer.add_scalar('train loss/obj', losses_obj.avg, epoch)
        #tb_writer.add_scalar('train loss/cell', losses_cell.avg, epoch)
        #tb_writer.add_scalar('train loss/neighbor', losses_neighbor.avg, epoch)
        tb_writer.add_scalar('val loss/total', val_losses.avg, epoch)
        tb_writer.add_scalar('MAE/by prob', pred_prob, epoch)
        #tb_writer.add_scalar('MAE/by threshold', pred_thresh, epoch)
        #tb_writer.add_scalar('MAE/cell', pred_cell, epoch)

        # end epoch ------------


def validate(args, val_list, val_list_depth, model, CUDA, num_imgs_val):
    print ('begin validation')
    val_loader = torch.utils.data.DataLoader(listDataset(val_list, 
                    val_list_depth,
                    shuffle=False, 
                    depth=args.depth,
                    # transform1=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),]), 
                    # transform2=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.502], std=[0.291]),]), 
                    train=False), 
                    batch_size=1)    
    
    model.eval()

    pbar = enumerate(val_loader)
    pbar = tqdm(pbar, total=len(val_loader))  # progress bar
    
    losses = AverageMeter()
    mae_prob, mae_thresh = 0, 0
    mae_cell = 0
    length = args.cell_size
    imgs, targets = [], []
    b_num = 0

    with open(os.path.join(args.log_dir,'results.txt'), 'w') as f:
        k = 0 
        imgs, targets = [], []
        for bi, (img, target, bxy, l, depth) in pbar:  # batch ----------
            
            
            target = target.squeeze(0)
            img = img.squeeze(0)
            bxy = bxy.squeeze(0)
            l = l.squeeze(0)
            # ni = int(math.ceil(img_big.shape[2] / length)) 
            # nj = int(math.ceil(img_big.shape[3] / length))  
            
            targets.append(torch.tensor(bxy))

            if args.depth:
                img = torch.cat((img_chip ,img_chip_depth), dim=1)
            # else:
            #     img = torch.clone(img_chip)

            imgs.append(img)

            b_num += 1
            k = k + 1
            
            if k == args.stack_size:

                imgs = torch.stack(imgs, dim=0).squeeze(1)
                targets = [ti for ti in targets if len(ti) != 0]
                if not targets:
                    targets.append(torch.tensor([[-1, 0, 0]]))
                targets = torch.cat(targets)

                if CUDA:
                    imgs = imgs.cuda()
                    targets = targets.cuda()

                with torch.no_grad():
                    predictions0 = model(imgs, training=False)
                    compute_loss = ComputeLoss(model, in_size=l)
                    loss, _ = compute_loss(predictions0, targets)  # loss scaled by batch_size

                    losses.update(loss.item(), imgs.size(0))
                    
                    #predictions0, predictions1 = predictions[..., 0], predictions[..., 1]
                    # print('before',targets.shape[0])
                    # x = torch.where(targets == torch.tensor([[-1, 0, 0]]))
                    # print('x', x)
                    # print('after',targets.shape[0])
                    targets = targets.shape[0]
                    #pred_prob, _ = predictions0.view(predictions0.size(0), -1).max(1, keepdim=True)
                    pred_prob = predictions0.sum()
                    #pred_prob = pred_prob.sum()
                    pred_thresh = (predictions0 > args.threshold).sum()
                    pred_cell = predictions0.sum()
                    # print('pred_prob', pred_prob)
                    # print('targets', targets)
                    mae_prob += abs(pred_prob - targets)
                    mae_thresh += abs(pred_thresh - targets)
                    mae_cell += abs(pred_cell - targets)

                    s = '*Target {targets:.0f}\t *Pred {pred_prob:.4f}\t *Pred_Thresh {pred_thresh:.4f}\t *Pred_Cell {pred_cell:.4f}\t *MAE {mae_prob:.4f}\t *MAE_Thresh {mae_thresh:.4f}\t *MAE_Cell {mae_cell:.4f} \n'.\
                        format(targets=targets, pred_prob=pred_prob, pred_thresh=0, pred_cell=0, \
                            mae_prob=(pred_prob-targets), mae_thresh=(0), mae_cell=(0))
                    
                    f.writelines(s)

                    imgs = []
                    targets = []
                    b_num = 0
                    k = 0
        
    mae_prob = mae_prob/num_imgs_val
    mae_thresh = mae_thresh/num_imgs_val
    mae_cell = mae_cell/num_imgs_val

    print(' * MAE_Prob {mae_prob:.3f} '.format(mae_prob=mae_prob))
    #print(' * MAE_Thresh {mae_thresh:.3f} ({thresh:.3f}) '.format(mae_thresh=mae_thresh, thresh=args.threshold))
    #print(' * MAE_Cell {mae_cell:.3f} '.format(mae_cell=mae_cell))

    return mae_prob, mae_thresh, mae_cell, losses       


def main():
    args = parser.parse_args()

    args.best_pred = 1e6

    args.log_dir = os.path.join(args.log_dir,args.model_desc)
    files = glob.glob(os.path.normpath(os.path.join(args.log_dir,'*')))
    for f in files:
        os.remove(f)
    tb_writer = SummaryWriter(args.log_dir)

    args.checkpoint_path += ('/'+args.model_desc)
    if not pathlib.Path(args.checkpoint_path).exists():
        os.mkdir(args.checkpoint_path)
    files = glob.glob(os.path.normpath(os.path.join(args.checkpoint_path,'*')))
    for f in files:
        os.remove(f)
    args.checkpoint_path = os.path.join(args.checkpoint_path,'checkpoint.pth.tar')


    with open(args.train_json, 'r') as outfile:        
        train_list_main = json.load(outfile)
    with open(args.val_json, 'r') as outfile:       
        val_list_main = json.load(outfile)

    train_list_img = [st.replace('/home/leeyh/Downloads/Shanghai', 'crowd_csr_grid/datasets/shanghai') for st in train_list_main]
    val_list_img = [st.replace('/home/leeyh/Downloads/Shanghai', 'crowd_csr_grid/datasets/shanghai') for st in val_list_main]
    num_imgs_train = len(train_list_img) 
    num_imgs_val = len(val_list_img) 

    catgrs = os.listdir('crowd_csr_grid/data_chopped/part_A_test/')
    direct = []
    for i, cat in enumerate(catgrs):
        direct.append([])
        filenames = os.listdir(os.path.join('crowd_csr_grid/data_chopped/part_A_test/',cat))
        for j in filenames:
            direct[i].append(os.path.join('crowd_csr_grid/data_chopped/part_A_test/',cat,j))

    val_list = []
    num_cat = len(catgrs)
    i = 0
    count_ctgr = np.zeros((len(catgrs),1))
    c = np.zeros((len(catgrs),1))
    remain = np.ones((len(catgrs),1))
    while remain.any() == 1:
        num = len(direct[i])//args.stack_size
        if c[i] == num:
            remain[i] = 0 
            i = i + 1
                
        else: 
            val_list.append(direct[i][int(count_ctgr[i])])
            count_ctgr[i] = count_ctgr[i] + 1
            
            if count_ctgr[i] % (args.stack_size)==0:
                c[i] = c[i] + 1 
                i = i + 1
                
                    
        if i == len(catgrs):
                i = 0     

    if args.depth:
        val_list_depth = [st.replace('images', 'depth_resized_h5') for st in val_list]
        val_list_depth = [st.replace('.jpg', '.h5') for st in val_list_depth]
    else:
        val_list_depth = None

    
    if args.use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device
        torch.cuda.manual_seed(time.time())
        CUDA =True
    else:
        CUDA = False


    model = CSRNet(in_size=args.cell_size)

    if CUDA:
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr0, betas=(args.momentum, 0.999))  # adjust beta1 to momentum
    
    if args.use_pre:
        if os.path.isfile(args.checkpoint_path):
            print("=> loading checkpoint '{}'".format(args.checkpoint_path))
            checkpoint = torch.load(args.checkpoint_path)
            args.start_epoch = checkpoint['epoch']+1
            args.best_pred = checkpoint['best_pred']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.checkpoint_path, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.checkpoint_path))


    train(args, model, optimizer, tb_writer, CUDA, val_list, val_list_depth, num_imgs_train, num_imgs_val) 


if __name__ == '__main__':
    main() 