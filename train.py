<<<<<<< Updated upstream
import sys
=======
# this file does chopping while training. 
>>>>>>> Stashed changes
import os

import warnings

from model import CSRNet

from utils import save_checkpoint

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from skimage.transform import resize, downscale_local_mean

import numpy as np
import argparse
import json
import cv2
<<<<<<< Updated upstream
import dataset
import time
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

=======
import matplotlib.pyplot as plt
from PIL import Image
import glob
>>>>>>> Stashed changes

import pathlib
path = pathlib.Path(__file__).parent.absolute()
print(os.getcwd())
print('path is',path)
parser = argparse.ArgumentParser(description='PyTorch CSRNet')

<<<<<<< Updated upstream
parser.add_argument('--train_json', metavar='TRAIN', default=os.path.join(path,'part_A_train.json'), help='path to train json')
parser.add_argument('--test_json', metavar='TEST', default=os.path.join(path,'part_A_val.json'), help='path to test json')
parser.add_argument('--pre', '-p', metavar='PRETRAINED', default=os.path.join(path,'runs/weights/checkpoint.pth.tar'), type=str, help='path to the pretrained model')
parser.add_argument('--gpu',metavar='GPU', default='0', type=str, help= 'GPU id to use.')
parser.add_argument('--checkpoint_path',metavar='CHECKPOINT', default= os.path.join(path,'runs/weights'), type=str, help='checkpoint path')
parser.add_argument('--log_dir',metavar='CHECKPOINT', default= os.path.join(path,'runs/logs'), type=str, help='log dir')
parser.add_argument('--gpu_log_dir',metavar='CHECKPOINT', default= os.path.join(path,'runs/gpulogs'), type=str, help='GPU log dir')


    
global args,best_prec1

best_prec1 = 1e6

args = parser.parse_args()
args.original_lr = 1e-7
args.lr = 1e-7
args.batch_size    = 1
args.momentum      = 0.95
args.decay         = 5*1e-4
args.start_epoch   = 0
args.epochs = 400
args.steps         = [-1,1,100,150]
args.scales        = [1,1,1,1]
args.workers = 4
args.seed = time.time()
args.print_freq = 30

tb_writer = SummaryWriter(args.log_dir)
gpu_time_writer = SummaryWriter(args.gpu_log_dir)
=======
# GENERAL
parser.add_argument('--model_desc', default='test/', help="Set model description")
# parser.add_argument('--model_desc', default='test/', help="Set model description")
parser.add_argument('--train_json', default=os.path.join(path,'datasets/shanghai/part_A_train.json'), help='path to train json')
parser.add_argument('--val_json', default=os.path.join(path,'datasets/shanghai/part_A_test.json'), help='path to test json')
parser.add_argument('--use_pre', default=False, type=bool, help='use the pretrained model?')
parser.add_argument('--use_gpu', default=True, action="store_false", help="Indicates whether or not to use GPU")
parser.add_argument('--device', default='1', type=str, help='GPU id to use.')
parser.add_argument('--checkpoint_path', default=os.path.join(path,'runs/weights'), type=str, help='checkpoint path')
parser.add_argument('--log_dir', default=os.path.join(path,'runs/log'), type=str, help='log dir')
parser.add_argument('--exp', default='shanghai', type=str, help='set dataset for training experiment')
parser.add_argument('--depth', default=False, type=bool, help='using depth?')

# MODEL
parser.add_argument('--model_file', default=path/'model.yaml')
parser.add_argument('--cell_size', default=[128,128,64,32], type=int, help="cell size")
parser.add_argument('--threshold', default=0.01, type=int, help="threshold for the classification output")

# TRAINING
parser.add_argument('--batch_size', default=512, type=int)
parser.add_argument('--epochs', default=70, type=int, help="Number of epochs to train for")
parser.add_argument('--workers', default=4, type=int, help="Number of workers in loading dataset")
parser.add_argument('--start_epoch', default=0, type=int, help="start_epoch")
parser.add_argument('--vis', default=False, type=bool, help='visualize the inputs') 
parser.add_argument('--lr0', default=0.000001, type=float, help="initial learning rate")
parser.add_argument('--weight_decay', default=0.0005, type=float, help="weight_decay")
parser.add_argument('--momentum', default=0.937, type=float, help="momentum")
parser.add_argument('--adam', default=False, type=bool, help='use torch.optim.Adam() optimizer') 
>>>>>>> Stashed changes

def main():

    
    best_prec1 = 1e6
    command = os.popen('nvidia-smi -L')
    GPU = command.read()
    if not Path(args.checkpoint_path).exists():
        os.mkdir(args.checkpoint_path)
    args.checkpoint_path = os.path.join(args.checkpoint_path,'checkpoint.pth.tar')

<<<<<<< Updated upstream
    with open(args.train_json, 'r') as outfile:        
        train_list = json.load(outfile)
    with open(args.test_json, 'r') as outfile:       
        val_list = json.load(outfile)
    train_list = [st.replace('/home/leeyh/Downloads/Shanghai', '/content/drive/My Drive/Queens/capsnet_crowd/dataset') for st in train_list]
    val_list = [st.replace('/home/leeyh/Downloads/Shanghai', '/content/drive/My Drive/Queens/capsnet_crowd/dataset') for st in val_list]

    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.cuda.manual_seed(args.seed)
    
    model = CSRNet()
    
    model = model.cuda()
    
    criterion = nn.MSELoss(size_average=False).cuda()
    criterion_val = nn.MSELoss(size_average=False).cuda()
    
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.decay)
    if args.pre:
        if os.path.isfile(args.pre):
            print("=> loading checkpoint '{}'".format(args.pre))
            checkpoint = torch.load(args.pre)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.pre, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.pre))
            
=======
    print(args.cell_size)   
    # files = glob.glob(os.path.normpath(os.path.join("crowd_csr_grid/img/",'*')))
    # for f in files:
    #     os.remove(f)
>>>>>>> Stashed changes
    for epoch in range(args.start_epoch, args.epochs):
        
        adjust_learning_rate(optimizer, epoch)
        
        train(train_list, model, criterion, optimizer, epoch, GPU)
        prec1 = validate(val_list, model, criterion_val, epoch)
        
        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)
        print(' * best MAE {mae:.3f} '
              .format(mae=best_prec1))
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.pre,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, args.checkpoint_path)

def train(train_list, model, criterion, optimizer, epoch, GPU):
    
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    
    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(train_list,
                       shuffle=True,
<<<<<<< Updated upstream
                       transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),]), 
=======
                       depth=args.depth,
                    #    transform1=transforms.Compose([transforms.ToTensor(),]), 
                    #    transform2=transforms.Compose([transforms.ToTensor(),]), 
                       transform1=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),]), 
                       transform2=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.502], std=[0.291]),]), 
>>>>>>> Stashed changes
                       train=True, 
                       seen=model.seen,
                       batch_size=args.batch_size,
                       num_workers=args.workers),
                    batch_size=args.batch_size)
    print('epoch %d, processed %d samples, lr %.10f' % (epoch, epoch * len(train_loader.dataset), args.lr))
    
    model.train()
    end = time.time()
    
    for i,(img, target)in enumerate(train_loader):
        # image = downscale_local_mean(img, (1, 1))
        # target_d = downscale_local_mean(target, (1, 1))
        data_time.update(time.time() - end)
        
        img = img.cuda()
        img = Variable(img)
        output = model(img)
        
        # if epoch == 1:
        #     tb_writer.add_graph(torch.jit.trace(model, img, strict=False), [])

        
        target = target.type(torch.FloatTensor).unsqueeze(0).cuda()
        target = Variable(target)
        # target_fake = torch.rand(1,512,120,67).cuda()
        # target_fake = torch.rand(1,10,10).cuda()
        # print('target',target.shape)
        # print('output',output.shape)
        
        loss = criterion(output, target)

        losses.update(loss.item(), img.size(0))
        optimizer.zero_grad()
<<<<<<< Updated upstream
        loss.backward()
        optimizer.step()    
=======

        imgs, targets = [], []
        length = args.cell_size
        b_num = 0
        
        for bi, (img_big, target_big, img_big_depth) in pbar:  # batch ----------
            imgs, targets = [], []
            data_time.update(time.time() - end)

            # ni = int(math.ceil(img_big.shape[2] / length)) 
            # nj = int(math.ceil(img_big.shape[3] / length))  
            k = 0 
            ny = 0
            update_len = img_big.shape[2] 
            
            while update_len - length[k] > 0:
                ny = ny + 1
                update_len = update_len - length[k]
                if k < len(length)-1:
                    k = k+1
            ny = ny + 1
                # update_len = update_len - length[k]
            if bi == 0:
                print(img_big.shape[2])
                print('ny',ny)
            high_end = img_big.shape[2]
            for j in range(ny):  
                if j > len(length) - 1:
                    l = length[-1]
                else:
                    l = length[j]
                nx = int(math.ceil(img_big.shape[3] / l))  
                # print('row number:',j)
                print(nx)
                for i in range(nx):  
                    # print(j)
                    # y2 = img_big.shape[2]-(j*l)
                    # y1 = max(img_big.shape[2]-((j+1)*l),0)
                    # x2 = img_big.shape[3]-(i*l)
                    # x1 = max(img_big.shape[3]-((i+1)*l),0)

                    y2 = high_end
                    y1 = max(high_end-l,0)
                    x2 = img_big.shape[3]-(i*l)
                    x1 = max(img_big.shape[3]-((i+1)*l),0)

                    # print('y2 y1',y2,y1)
                    # print(y1)
                    img_chip = img_big[:, :, y1:y2, x1:x2]
                    # if bi == 0:
                        # print('y1',y1)
                        # print('y2',y2)
                        # print('x1',x1)
                        # print('x2',x2)
                        # print(img_chip.shape)
                        # im = Image.fromarray((img_chip.squeeze(0).numpy()*255).transpose(1,2,0).astype(np.uint8)) 
                        # im.save("crowd_csr_grid/img/"+str(j)+"_"+str(i)+".jpeg")
                        # im = Image.fromarray((img_big.squeeze(0).numpy()*255).transpose(1,2,0).astype(np.uint8)) 
                        # im.save("crowd_csr_grid/img/im.jpeg")


                    img_chip = zeropad(img_chip.squeeze(0).permute(1,2,0).numpy(), l - img_chip.shape[2], l - img_chip.shape[3])
                    img_chip = torch.from_numpy(img_chip).permute(2,0,1).unsqueeze(0)
                    
                    
                    if args.depth:
                        img_chip_depth = img_big_depth[:, :, y1:y2, x1:x2]
                        img_chip_depth  = zeropad(img_chip_depth.squeeze(0).permute(1,2,0).numpy(), l - img_chip_depth.shape[2], l - img_chip_depth.shape[3])
                        img_chip_depth = torch.from_numpy(img_chip_depth).unsqueeze(2)
                        img_chip_depth = img_chip_depth.permute(2,0,1).unsqueeze(0)
                        assert img_chip_depth.shape[2] == img_chip_depth.shape[3] == l, 'image size error'
                    
                    target_chip = target_big[:, y1:y2, x1:x2]
                    target_chip = zeropad(target_chip.squeeze(0).numpy(), l - target_chip.shape[1], l - target_chip.shape[2], target=True)
                    target_chip = torch.from_numpy(target_chip).unsqueeze(0)
                    
                    assert img_chip.shape[2] == img_chip.shape[3] == l, 'image size error'
                    assert target_chip.shape[1] == target_chip.shape[2] == l, 'target size error'
                    # print('target_chip', target_chip.shape)
                    # print(bi)
                    if args.vis:
                        vis_input(img_big.squeeze(0), target_big.squeeze(0), bi, path, 'big', args)
                        vis_input(img_chip.squeeze(0), target_chip.squeeze(0), bi, path, 'chip', args)
                        if args.depth:
                            vis_input(img_big_depth.squeeze(0), img_chip_depth.squeeze(0), bi, path, 'depth', args)

                    coord = (target_chip.squeeze(0)).nonzero(as_tuple=False)
                    # print('coord',coord)
                    bxy = [[b_num, yb/l, xb/l, i, j] for (yb, xb) in coord] #b_num is patch number, xb/l is the location relative to length, i and j are the indexes of the patch we are in.
                    # i is the x index of patch
                    # j is the y index of patch
                    # print('bxy',bxy)
                    targets.append(torch.tensor(bxy))
                    # print('targets',targets)

                    if args.depth:
                        img = torch.cat((img_chip ,img_chip_depth), dim=1)
                    else:
                        img = torch.clone(img_chip)

                    img = Variable(img)
                    imgs.append(img)

                    b_num += 1

                    #if b_num >= args.batch_size:
                    if i == (nx-1):
                        imgs = torch.stack(imgs, dim=0).squeeze(1)
                        targets = [ti for ti in targets if len(ti) != 0]
                        if not targets:
                            print(b_num)
                            targets.append(torch.tensor([[-1, 0, 0, 0, 0]]))
                            print('*'*1000)
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
                        # print('pred',pred0.shape)
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
                high_end = y1
            # end batch ------------

        pred_prob, _, _, val_losses = validate(args, val_list, val_list_depth, model, CUDA, compute_loss)

        is_best = pred_prob < args.best_pred
        args.best_pred = min(pred_prob, args.best_pred)
        print(' * Best MAE {mae:.3f} '.format(mae=args.best_pred))
>>>>>>> Stashed changes
        
        batch_time.update(time.time() - end)
        end = time.time()
        
<<<<<<< Updated upstream
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  .format(epoch, i, len(train_loader), batch_time=batch_time, data_time=data_time, loss=losses))
        
            tb_writer.add_scalar('train loss/iteration', losses.avg, epoch * len(train_loader.dataset) + i)
    gpu_time_writer.add_text('GPU_Model', GPU)
    gpu_time_writer.add_text('time', str(data_time.sum))
    tb_writer.add_scalar('train loss/epoch', losses.avg, epoch)


def validate(val_list, model, criterion_val, epoch):
    print ('begin test')
    losses_val = AverageMeter()
    test_loader = torch.utils.data.DataLoader(
    dataset.listDataset(val_list,
                   shuffle=False,
                   transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),]),  train=False),
    batch_size=args.batch_size)    
=======
        tb_writer.add_scalar('train loss/total', losses.avg, epoch)
        #tb_writer.add_scalar('train loss/obj', losses_obj.avg, epoch)
        #tb_writer.add_scalar('train loss/cell', losses_cell.avg, epoch)
        #tb_writer.add_scalar('train loss/neighbor', losses_neighbor.avg, epoch)
        tb_writer.add_scalar('val loss/total', val_losses.avg, epoch)
        tb_writer.add_scalar('MAE/by prob', pred_prob, epoch)
        #tb_writer.add_scalar('MAE/by threshold', pred_thresh, epoch)
        #tb_writer.add_scalar('MAE/cell', pred_cell, epoch)

        # end epoch ------------


def validate(args, val_list, val_list_depth, model, CUDA, compute_loss):
    print ('begin validation')
    val_loader = torch.utils.data.DataLoader(listDataset(val_list, 
                    val_list_depth,
                    shuffle=False, 
                    depth=args.depth,
                    # transform1=transforms.Compose([transforms.ToTensor(),]), 
                    # transform2=transforms.Compose([transforms.ToTensor(),]), 
                    transform1=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),]), 
                    transform2=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.502], std=[0.291]),]), 
                    train=False), 
                    batch_size=1)    
>>>>>>> Stashed changes
    
    model.eval()
    
<<<<<<< Updated upstream
    mae = 0
    
    for i,(img, target) in enumerate(test_loader):
        img = img.cuda()
        img = Variable(img)
        target = target.cuda()
        output = model(img).cuda()
        loss_val = criterion_val(output, target)

        losses_val.update(loss_val.item(), img.size(0))
        mae += abs(output.data.sum()-target.sum().type(torch.FloatTensor).cuda())
    tb_writer.add_scalar('valid loss/epoch', losses_val.avg, epoch)    
    mae = mae/len(test_loader)    
    print(' * MAE {mae:.3f} '
              .format(mae=mae))

    return mae    
        
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
=======
    losses = AverageMeter()
    mae_prob, mae_thresh = 0, 0
    mae_cell = 0
    length = args.cell_size
    imgs, targets = [], []
    b_num = 0

    with open(os.path.join(args.log_dir,'results.txt'), 'w') as f:
        for bi, (img_big, target_big, img_big_depth) in pbar:  # batch ----------
            imgs, targets = [], []

            # ni = int(math.ceil(img_big.shape[2] / length)) 
            # nj = int(math.ceil(img_big.shape[3] / length))  
            k = 0 
            ny = 0
            update_len = img_big.shape[2] 
            
            while update_len - length[k] > 0:
                ny = ny + 1
                update_len = update_len - length[k]
                if k < len(length)-1:
                    k = k+1
            ny = ny + 1
                # update_len = update_len - length[k]
            # if bi == 0:
            #     print(img_big.shape[2])
            #     print('ny',ny)
            high_end = img_big.shape[2]
            for j in range(ny):  
                if j > len(length) - 1:
                    l = length[-1]
                else:
                    l = length[j]
                nx = int(math.ceil(img_big.shape[3] / l))  
                # print('row number:',j)
                # print('l',l)
                for i in range(nx):  
                    # print(j)
                    # y2 = img_big.shape[2]-(j*l)
                    # y1 = max(img_big.shape[2]-((j+1)*l),0)
                    # x2 = img_big.shape[3]-(i*l)
                    # x1 = max(img_big.shape[3]-((i+1)*l),0)

                    y2 = high_end
                    y1 = max(high_end-l,0)
                    x2 = img_big.shape[3]-(i*l)
                    x1 = max(img_big.shape[3]-((i+1)*l),0)

                    # print('y2 y1',y2,y1)
                    # print(y1)
                    img_chip = img_big[:, :, y1:y2, x1:x2]
                    # if bi == 0:
                        # print('y1',y1)
                        # print('y2',y2)
                        # print('x1',x1)
                        # print('x2',x2)
                        # print(img_chip.shape)
                        # im = Image.fromarray((img_chip.squeeze(0).numpy()*255).transpose(1,2,0).astype(np.uint8)) 
                        # im.save("crowd_csr_grid/img/"+str(j)+"_"+str(i)+".jpeg")
                        # im = Image.fromarray((img_big.squeeze(0).numpy()*255).transpose(1,2,0).astype(np.uint8)) 
                        # im.save("crowd_csr_grid/img/im.jpeg")
                    
                    img_chip = zeropad(img_chip.squeeze(0).permute(1,2,0).numpy(), l - img_chip.shape[2], l - img_chip.shape[3])
                    img_chip = torch.from_numpy(img_chip).permute(2,0,1).unsqueeze(0)
                    
                    if args.depth:
                        img_chip_depth = img_big_depth[:, :, y1:y2, x1:x2]
                        img_chip_depth = zeropad(img_chip_depth.squeeze(0).permute(1,2,0).numpy(), l - img_chip_depth.shape[2], l - img_chip_depth.shape[3])
                        img_chip_depth = torch.from_numpy(img_chip_depth).unsqueeze(2)
                        img_chip_depth = img_chip_depth.permute(2,0,1).unsqueeze(0)
                        assert img_chip_depth.shape[2] == img_chip_depth.shape[3] == l, 'image size error'
                    
                    target_chip = target_big[:, y1:y2, x1:x2]
                    target_chip = zeropad(target_chip.squeeze(0).numpy(), l - target_chip.shape[1], l - target_chip.shape[2], target=True)
                    target_chip = torch.from_numpy(target_chip).unsqueeze(0)
                    
                    assert img_chip.shape[2] == img_chip.shape[3] == l, 'image size error'
                    assert target_chip.shape[1] == target_chip.shape[2] == l, 'target size error'

                    coord = (target_chip.squeeze(0)).nonzero(as_tuple=False)

                    bxy = [[b_num, yb/l, xb/l, i, j] for (yb, xb) in coord]
                    targets.append(torch.tensor(bxy))

                    if args.depth:
                        img = torch.cat((img_chip ,img_chip_depth), dim=1)
                    else:
                        img = torch.clone(img_chip)

                    imgs.append(img)

                    b_num += 1

                    if i == (nx-1):
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
                            targets = targets.shape[0]
                            #pred_prob, _ = predictions0.view(predictions0.size(0), -1).max(1, keepdim=True)
                            pred_prob = predictions0.sum()
                            #pred_prob = pred_prob.sum()
                            pred_thresh = (predictions0 > args.threshold).sum()
                            pred_cell = predictions0.sum()

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
                high_end = y1
    mae_prob = mae_prob/len(val_loader)
    mae_thresh = mae_thresh/len(val_loader)
    mae_cell = mae_cell/len(val_loader)

    print(' * MAE_Prob {mae_prob:.3f} '.format(mae_prob=mae_prob))
    #print(' * MAE_Thresh {mae_thresh:.3f} ({thresh:.3f}) '.format(mae_thresh=mae_thresh, thresh=args.threshold))
    #print(' * MAE_Cell {mae_cell:.3f} '.format(mae_cell=mae_cell))

    return mae_prob, mae_thresh, mae_cell, losses       


def main():
    args = parser.parse_args()

    args.best_pred = 1e6

    args.log_dir = os.path.join(args.log_dir,args.model_desc)
    # files = glob.glob(os.path.normpath(os.path.join(args.log_dir,'*')))
    # for f in files:
    #     os.remove(f)
    tb_writer = SummaryWriter(args.log_dir)

    args.checkpoint_path += ('/'+args.model_desc)
    if not pathlib.Path(args.checkpoint_path).exists():
        os.mkdir(args.checkpoint_path)
    # files = glob.glob(os.path.normpath(os.path.join(args.checkpoint_path,'*')))
    # for f in files:
    #     os.remove(f)
    args.checkpoint_path = os.path.join(args.checkpoint_path,'checkpoint.pth.tar')

    if args.exp == 'shanghai':

        with open(args.train_json, 'r') as outfile:        
            train_list_main = json.load(outfile)
        with open(args.val_json, 'r') as outfile:       
            val_list_main = json.load(outfile)

        train_list = [st.replace('/home/leeyh/Downloads/Shanghai', 'crowd_csr_grid/datasets/shanghai') for st in train_list_main]
        val_list = [st.replace('/home/leeyh/Downloads/Shanghai', 'crowd_csr_grid/datasets/shanghai') for st in val_list_main]

        if args.depth:
            train_list_depth = [st.replace('images', 'depth_resized_h5') for st in train_list]
            val_list_depth = [st.replace('images', 'depth_resized_h5') for st in val_list]
            train_list_depth = [st.replace('.jpg', '.h5') for st in train_list_depth]
            val_list_depth = [st.replace('.jpg', '.h5') for st in val_list_depth]
        else:
            train_list_depth = None
            val_list_depth = None

    if args.use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device
        torch.cuda.manual_seed(time.time())
        CUDA =True
    else:
        CUDA = False

    #model = Model(args.model_file, args.cell_size)

    model = CSRNet(in_size=args.cell_size)

    if CUDA:
        model = model.cuda()

    #optimizer = torch.optim.SGD(model.parameters(), args.lr0, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr0, betas=(args.momentum, 0.999))  # adjust beta1 to momentum
>>>>>>> Stashed changes
    
    args.lr = args.original_lr
    
    for i in range(len(args.steps)):
        
        scale = args.scales[i] if i < len(args.scales) else 1
        
        if epoch >= args.steps[i]:
            args.lr = args.lr * scale
            if epoch == args.steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr
        
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count    

    
if __name__ == '__main__':
    main() 