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
from model import ComputeLoss, CSRNet
from dataset import listDataset
from tqdm import tqdm
import math
from utils import save_checkpoint, AverageMeter, vis_input, zeropad
import cv2


path = pathlib.Path(__file__).parent.absolute()
parser = argparse.ArgumentParser(description='RCVLab-AiimLab Crowd counting')

# GENERAL
parser.add_argument('--model_desc', default='UCF-QNRF, cell128, lr7, 2ch/', help="Set model description")
parser.add_argument('--train_json', default=path/'datasets/UCF-QNRF/Train.json', help='path to train json')
parser.add_argument('--val_json', default=path/'datasets/UCF-QNRF/Test.json', help='path to test json')
parser.add_argument('--use_pre', default=True, type=bool, help='use the pretrained model?')
parser.add_argument('--use_gpu', default=True, action="store_false", help="Indicates whether or not to use GPU")
parser.add_argument('--device', default='0', type=str, help='GPU id to use.')
parser.add_argument('--checkpoint_path', default=path.parent/'/drive/work_dirs/crowd_counting_UCF-QNRF', type=str, help='checkpoint path')
parser.add_argument('--log_dir', default=path.parent/'/drive/work_dirs/crowd_counting_UCF-QNRF/log', type=str, help='log dir')
parser.add_argument('--exp', default='QNRF', type=str, help='set dataset for training experiment')
parser.add_argument('--depth', default=False, type=bool, help='using depth?')

# MODEL
parser.add_argument('--model_file', default=path/'model.yaml')
parser.add_argument('--cell_size', default=128, type=int, help="cell size")
parser.add_argument('--threshold', default=0.01, type=int, help="threshold for the classification output")

# TRAINING
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--epochs', default=100, type=int, help="Number of epochs to train for")
parser.add_argument('--workers', default=1, type=int, help="Number of workers in loading dataset")
parser.add_argument('--start_epoch', default=10, type=int, help="start_epoch")
parser.add_argument('--vis', default=False, type=bool, help='visualize the inputs') 
parser.add_argument('--lr0', default=0.0000001, type=float, help="initial learning rate")
parser.add_argument('--weight_decay', default=0.0005, type=float, help="weight_decay")
parser.add_argument('--momentum', default=0.937, type=float, help="momentum")
parser.add_argument('--adam', default=False, type=bool, help='use torch.optim.Adam() optimizer') 


def train(args, model, optimizer, train_list, val_list, train_list_depth, val_list_depth, tb_writer, CUDA):

    compute_loss = ComputeLoss(model, in_size=args.cell_size)

    for epoch in range(args.start_epoch, args.epochs):
        train_loader = torch.utils.data.DataLoader(listDataset(train_list,
                       train_list_depth,
                       shuffle=True,
                       depth=args.depth,
                       transform1=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),]), 
                       transform2=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.502], std=[0.291]),]), 
                       train=True, 
                       seen=0,
                       batch_size=1,
                       num_workers=args.workers,
                       exp=args.exp))

        losses = AverageMeter()
        losses_loc = AverageMeter()
        losses_cell = AverageMeter()
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

        for bi, (img_big, target_big, img_big_depth) in pbar:  # batch ----------
            data_time.update(time.time() - end)

            ni = int(math.ceil(img_big.shape[2] / length)) 
            nj = int(math.ceil(img_big.shape[3] / length))  
            for i in range(ni):  
                for j in range(nj):  
                    y2 = min((i + 1) * length, img_big.shape[2])
                    y1 = y2 - length
                    x2 = min((j + 1) * length, img_big.shape[3])
                    x1 = x2 - length

                    img_chip = img_big[:, :, y1:y2, x1:x2]
                    img_chip = zeropad(img_chip.squeeze(0).permute(1,2,0).numpy(), length - img_chip.shape[2], length - img_chip.shape[3])
                    img_chip = torch.from_numpy(img_chip).permute(2,0,1).unsqueeze(0)
                    
                    if args.depth:
                        img_chip_depth = img_big_depth[:, :, y1:y2, x1:x2]
                        img_chip_depth  = zeropad(img_chip_depth.squeeze(0).permute(1,2,0).numpy(), length - img_chip_depth.shape[2], length - img_chip_depth.shape[3])
                        img_chip_depth = torch.from_numpy(img_chip_depth).unsqueeze(2)
                        img_chip_depth = img_chip_depth.permute(2,0,1).unsqueeze(0)
                        assert img_chip_depth.shape[2] == img_chip_depth.shape[3] == length, 'image size error'
                    
                    target_chip = target_big[:, y1:y2, x1:x2]
                    target_chip = zeropad(target_chip.squeeze(0).numpy(), length - target_chip.shape[1], length - target_chip.shape[2], target=True)
                    target_chip = torch.from_numpy(target_chip).unsqueeze(0)
                    
                    assert img_chip.shape[2] == img_chip.shape[3] == length, 'image size error'
                    assert target_chip.shape[1] == target_chip.shape[2] == length, 'target size error'
            
                    if args.vis:
                        vis_input(img_big.squeeze(0), target_big.squeeze(0))
                        vis_input(img_chip.squeeze(0), target_chip.squeeze(0))
                        if args.depth:
                            vis_input(img_big_depth.squeeze(0), img_chip_depth.squeeze(0))

                    coord = (target_chip.squeeze(0)).nonzero(as_tuple=False)

                    bxy = [[b_num, yb/length, xb/length, i, j] for (yb, xb) in coord]
                    targets.append(torch.tensor(bxy))

                    if args.depth:
                        img = torch.cat((img_chip ,img_chip_depth), dim=1)
                    else:
                        img = torch.clone(img_chip)

                    img = Variable(img)
                    imgs.append(img)

                    b_num += 1

                    if b_num >= args.batch_size:
                    #if i == (ni-1) and j == (nj-1):
                        imgs = torch.stack(imgs, dim=0).squeeze(1)
                        targets = [ti for ti in targets if len(ti) != 0]
                        if not targets:
                            targets.append(torch.tensor([[-1, 0, 0]]))
                        targets = torch.cat(targets)

                        if CUDA:
                            imgs = imgs.cuda()
                            targets = targets.cuda()
                        
                        #if epoch <= 1:
                        #    tb_writer.add_graph(torch.jit.trace(model, imgs, strict=False), [])

                        pred0, pred1 = model(imgs, training=True)  # forward
                        loss, lloc, lcell = compute_loss(pred0, pred1, targets) 

                        losses.update(loss.item(), imgs.size(0))
                        losses_loc.update(lloc.item(), imgs.size(0))
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

            # end batch ------------

        mae_loc, mae_cell, val_losses = validate(args, val_list, val_list_depth, model, CUDA, compute_loss)

        is_best = mae_loc < args.best_mae
        args.best_mae = min(mae_loc, args.best_mae)
        print(' * Best MAE {mae:.3f} '.format(mae=args.best_mae))
        
        save_checkpoint({
            'epoch': epoch,
            'arch': args.checkpoint_path,
            'state_dict': model.state_dict(),
            'best_pred': args.best_mae,
            'optimizer' : optimizer.state_dict(),}, False, args.checkpoint_path)
        
        tb_writer.add_scalar('train loss/total', losses.avg, epoch)
        tb_writer.add_scalar('train loss/loc', losses_loc.avg, epoch)
        tb_writer.add_scalar('train loss/cell', losses_cell.avg, epoch)
        tb_writer.add_scalar('val loss/total', val_losses.avg, epoch)
        tb_writer.add_scalar('MAE/Loc', mae_loc, epoch)
        tb_writer.add_scalar('MAE/cell', mae_cell, epoch)

        # end epoch ------------


def validate(args, val_list, val_list_depth, model, CUDA, compute_loss):
    print ('begin validation')
    val_loader = torch.utils.data.DataLoader(listDataset(val_list, 
                    val_list_depth,
                    shuffle=False, 
                    depth=args.depth,
                    transform1=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),]), 
                    transform2=transforms.Compose([transforms.ToTensor(),]), 
                    train=False,
                    exp=args.exp), 
                    batch_size=1)    
    
    model.eval()

    pbar = enumerate(val_loader)
    pbar = tqdm(pbar, total=len(val_loader))  # progress bar
    
    losses = AverageMeter()
    mae_loc = 0
    mae_cell = 0
    length = args.cell_size
    imgs, targets = [], []
    b_num = 0

    with open(args.log_dir / 'results.txt', 'w') as f:
        for bi, (img_big, target_big, img_big_depth) in pbar:  
            ni = int(math.ceil(img_big.shape[2] / length))  
            nj = int(math.ceil(img_big.shape[3] / length)) 
            for i in range(ni):  
                for j in range(nj):  
                    y2 = min((i + 1) * length, img_big.shape[2])
                    y1 = y2 - length
                    x2 = min((j + 1) * length, img_big.shape[3])
                    x1 = x2 - length

                    img_chip = img_big[:, :, y1:y2, x1:x2]
                    img_chip = zeropad(img_chip.squeeze(0).permute(1,2,0).numpy(), length - img_chip.shape[2], length - img_chip.shape[3])
                    img_chip = torch.from_numpy(img_chip).permute(2,0,1).unsqueeze(0)
                    
                    if args.depth:
                        img_chip_depth = img_big_depth[:, :, y1:y2, x1:x2]
                        img_chip_depth = zeropad(img_chip_depth.squeeze(0).permute(1,2,0).numpy(), length - img_chip_depth.shape[2], length - img_chip_depth.shape[3])
                        img_chip_depth = torch.from_numpy(img_chip_depth).unsqueeze(2)
                        img_chip_depth = img_chip_depth.permute(2,0,1).unsqueeze(0)
                        assert img_chip_depth.shape[2] == img_chip_depth.shape[3] == length, 'image size error'
                    
                    target_chip = target_big[:, y1:y2, x1:x2]
                    target_chip = zeropad(target_chip.squeeze(0).numpy(), length - target_chip.shape[1], length - target_chip.shape[2], target=True)
                    target_chip = torch.from_numpy(target_chip).unsqueeze(0)
                    
                    assert img_chip.shape[2] == img_chip.shape[3] == length, 'image size error'
                    assert target_chip.shape[1] == target_chip.shape[2] == length, 'target size error'

                    coord = (target_chip.squeeze(0)).nonzero(as_tuple=False)

                    bxy = [[b_num, yb/length, xb/length, i, j] for (yb, xb) in coord]
                    targets.append(torch.tensor(bxy))

                    if args.depth:
                        img = torch.cat((img_chip ,img_chip_depth), dim=1)
                    else:
                        img = torch.clone(img_chip)

                    imgs.append(img)

                    b_num += 1

                    #if i == (ni-1) and j == (nj-1):
                    if b_num >= args.batch_size:
                        imgs = torch.stack(imgs, dim=0).squeeze(1)
                        targets = [ti for ti in targets if len(ti) != 0]
                        if (len(targets) <= 0):
                            b_num = 0
                            targets = []
                            imgs = []
                            continue
                        targets = torch.cat(targets)

                        if CUDA:
                            imgs = imgs.cuda()
                            targets = targets.cuda()

                        with torch.no_grad():
                            predictions0, predictions1 = model(imgs, training=False)
                            loss, _, _ = compute_loss(predictions0, predictions1, targets)  # loss scaled by batch_size

                            losses.update(loss.item(), imgs.size(0))
                            
                            targets = targets.shape[0]
                            pred_loc = predictions0.view(predictions0.size(0), -1).sum(1, keepdim=True)
                            pred_loc = pred_loc.sum()
                            pred_cell = predictions1.sum()

                            mae_loc += abs(pred_loc - targets)
                            mae_cell += abs(pred_cell - targets)

                            s = '*Target {targets:.0f}\t *Pred {pred_loc:.3f}\t *Pred_Cell {pred_cell:.3f}\t *MAE {mae_loc:.3f}\t *MAE_Cell {mae_cell:.3f} \n'.\
                                format(targets=targets, pred_loc=pred_loc, pred_cell=pred_cell, \
                                    mae_loc=(pred_loc-targets), mae_cell=(pred_cell-targets))
                            
                            f.writelines(s)

                            imgs = []
                            targets = []
                            b_num = 0
        
    mae_loc = mae_loc/len(val_loader)
    mae_cell = mae_cell/len(val_loader)

    print(' * MAE_Loc {mae_loc:.3f} '.format(mae_loc=mae_loc))
    print(' * MAE_Cell {mae_cell:.3f} '.format(mae_cell=mae_cell))

    return mae_loc, mae_cell, losses       


def main():
    args = parser.parse_args()

    args.best_mae = 1e6

    args.log_dir = args.log_dir / args.model_desc
    tb_writer = SummaryWriter(args.log_dir)

    args.checkpoint_path = args.checkpoint_path / args.model_desc
    if not pathlib.Path(args.checkpoint_path).exists():
        os.mkdir(args.checkpoint_path)
    args.checkpoint_path = args.checkpoint_path / 'checkpoint.pth.tar'

    if args.exp == 'shanghai':
        with open(args.train_json, 'r') as outfile:        
            train_list_main = json.load(outfile)
        with open(args.val_json, 'r') as outfile:       
            val_list_main = json.load(outfile)

        train_list = [st.replace('/home/leeyh/Downloads/Shanghai', '/media/mohsen/myDrive/datasets/ShanghaiTech_Crowd_Counting_Dataset') for st in train_list_main]
        val_list = [st.replace('/home/leeyh/Downloads/Shanghai', '/media/mohsen/myDrive/datasets/ShanghaiTech_Crowd_Counting_Dataset') for st in val_list_main]

        if args.depth:
            train_list_depth = [st.replace('images', 'depth_resized_h5') for st in train_list]
            val_list_depth = [st.replace('images', 'depth_resized_h5') for st in val_list]
            train_list_depth = [st.replace('.jpg', '.h5') for st in train_list_depth]
            val_list_depth = [st.replace('.jpg', '.h5') for st in val_list_depth]
        else:
            train_list_depth = None
            val_list_depth = None
    elif args.exp == 'QNRF':
        train_list_depth = None
        val_list_depth = None

        with open(args.train_json, 'r') as outfile:
            train_list = json.load(outfile)
        with open(args.val_json, 'r') as outfile:
            val_list = json.load(outfile)

    if args.use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device
        torch.cuda.manual_seed(time.time())
        CUDA =True
    else:
        CUDA = False

    model = CSRNet(in_size=args.cell_size)

    if CUDA:
        model = model.cuda()

    #optimizer = torch.optim.SGD(model.parameters(), args.lr0, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr0, betas=(args.momentum, 0.999))  # adjust beta1 to momentum
    
    if args.use_pre:
        if os.path.isfile(args.checkpoint_path):
            print("=> loading checkpoint '{}'".format(args.checkpoint_path))
            checkpoint = torch.load(args.checkpoint_path)
            args.start_epoch = checkpoint['epoch']+1
            args.best_mae = checkpoint['best_pred']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.checkpoint_path, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.checkpoint_path))


    train(args, model, optimizer, train_list, val_list, train_list_depth, val_list_depth, tb_writer, CUDA) 


if __name__ == '__main__':
    main() 
