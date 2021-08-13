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
from model_7_pre import ComputeLoss, CSRNet
from dataset_5 import listDataset
from tqdm import tqdm
import math
from utils import save_checkpoint, AverageMeter, vis_input, zeropad
import cv2
import glob
import wandb



config = {
'EPOCHS'             : 1000,
'LR'                 : 0.000001,
'MOMENTUM'           : 0.937,
'WANDB'              : True,
}
# wandb.init(project="trainn_7", reinit = True, config = config)

path = pathlib.Path(__file__).parent.absolute()
parser = argparse.ArgumentParser(description='RCVLab-AiimLab Crowd counting')

# GENERAL
parser.add_argument('--model_desc', default='train_7_pre/', help="Set model description")
parser.add_argument('--train_json', default=os.path.join(path,'datasets/shanghai/part_A_train.json'), help='path to train json')
parser.add_argument('--val_json', default=os.path.join(path,'datasets/shanghai/part_A_test.json'), help='path to test json')
parser.add_argument('--use_pre', default=False, type=bool, help='use the pretrained model?')
parser.add_argument('--use_gpu', default=True, action="store_false", help="Indicates whether or not to use GPU")
parser.add_argument('--device', default='2', type=str, help='GPU id to use.')
parser.add_argument('--checkpoint_path', default=os.path.join(path,'runs/weights'), type=str, help='checkpoint path')
parser.add_argument('--log_dir', default=os.path.join(path,'runs/log'), type=str, help='log dir')
parser.add_argument('--exp', default='shanghai', type=str, help='set dataset for training experiment')
parser.add_argument('--depth', default=False, type=bool, help='using depth?')

# MODEL
parser.add_argument('--model_file', default=path/'model.yaml')
parser.add_argument('--cell_size', default=128, type=int, help="cell size")
parser.add_argument('--threshold', default=0.01, type=int, help="threshold for the classification output")

# TRAINING
parser.add_argument('--batch_size', default=512, type=int)
parser.add_argument('--epochs', default=20, type=int, help="Number of epochs to train for")
parser.add_argument('--workers', default=4, type=int, help="Number of workers in loading dataset")
parser.add_argument('--start_epoch', default=0, type=int, help="start_epoch")
parser.add_argument('--vis', default=False, type=bool, help='visualize the inputs') 
parser.add_argument('--lr0', default=0.000001, type=float, help="initial learning rate")
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
                       num_workers=args.workers))

        losses = AverageMeter()
        losses_count_0 = AverageMeter()
        losses_count_1 = AverageMeter()
        losses_count_2 = AverageMeter()
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

                    #if b_num >= args.batch_size:
                    
                    if i == (ni-1) and j == (nj-1):
                        imgs = torch.stack(imgs, dim=0).squeeze(1)
                        targets = [ti for ti in targets if len(ti) != 0]
                        if not targets:
                            targets.append(torch.tensor([[-1, 0, 0, 0, 0]]))
                        targets = torch.cat(targets)

                        if CUDA:
                            imgs = imgs.cuda()
                            targets = targets.cuda()
                        
                        if epoch <= 1:
                           tb_writer.add_graph(torch.jit.trace(model, imgs, strict=False), [])

                        pred0, pred1, pred2, count = model(imgs, training=True)  # forward
                        loss, lcount_0, lcount_1, lcount_2 = compute_loss(pred0, pred1, pred2, targets, count) 
                        # wandb.log({'train_loss': loss, 'lcount_0': lcount_0, 'lcount_1': lcount_1, 'lcount_2': lcount_2})
                        losses.update(loss.item(), imgs.size(0))
                        losses_count_0.update(lcount_0.item(), imgs.size(0))
                        losses_count_1.update(lcount_1.item(), imgs.size(0))
                        losses_count_2.update(lcount_2.item(), imgs.size(0))

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

        mae_count_0, mae_count_1, mae_count_2, mae_count_T, val_losses = validate(args, val_list, val_list_depth, model, CUDA, compute_loss)
        # val_losses = validate(args, val_list, val_list_depth, model, CUDA, compute_loss)

        is_best_0 = mae_count_0 < args.best_mae_0
        args.best_mae_0 = min(mae_count_0, args.best_mae_0)
        print(' * Best MAE_0 {mae:.3f} '.format(mae=args.best_mae_0))
        # wandb.log({'best_mae_0': args.best_mae_0})
        is_best_1 = mae_count_1 < args.best_mae_1
        args.best_mae_1 = min(mae_count_1, args.best_mae_1)
        print(' * Best MAE_1 {mae:.3f} '.format(mae=args.best_mae_1))
        # wandb.log({'best_mae_1': args.best_mae_1})
        is_best_2 = mae_count_2 < args.best_mae_2
        args.best_mae_2 = min(mae_count_2, args.best_mae_2)
        # print(' * Best MAE_2 {mae:.3f} '.format(mae=args.best_mae_2))
        # wandb.log({'best_mae_2': args.best_mae_2})
        is_best_T = mae_count_T < args.best_mae_T
        args.best_mae_T = min(mae_count_T, args.best_mae_T)
        # print(' * Best MAE_T {mae:.3f} '.format(mae=args.best_mae_T))
        # wandb.log({'best_mae_3': args.best_mae_3})
        save_checkpoint({
            'epoch': epoch,
            'arch': args.checkpoint_path,
            'state_dict': model.state_dict(),
            'best_pred': args.best_mae_T,
            'optimizer' : optimizer.state_dict(),}, is_best_T, args.checkpoint_path)
        
        tb_writer.add_scalar('train loss/total', losses.avg, epoch)
        tb_writer.add_scalar('train loss/count_0', losses_count_0.avg, epoch)
        tb_writer.add_scalar('train loss/count_1', losses_count_1.avg, epoch)
        tb_writer.add_scalar('train loss/count_2', losses_count_2.avg, epoch)
        tb_writer.add_scalar('val loss/total', val_losses.avg, epoch)
        tb_writer.add_scalar('MAE/Count_0', mae_count_0, epoch)
        tb_writer.add_scalar('MAE/Count_1', mae_count_1, epoch)
        tb_writer.add_scalar('MAE/Count_2', mae_count_2, epoch)
        tb_writer.add_scalar('MAE/Count_T', mae_count_T, epoch)

        # end epoch ------------


def validate(args, val_list, val_list_depth, model, CUDA, compute_loss):
    print ('begin validation')
    val_loader = torch.utils.data.DataLoader(listDataset(val_list, 
                    val_list_depth,
                    shuffle=False, 
                    depth=args.depth,
                    transform1=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),]), 
                    transform2=transforms.Compose([transforms.ToTensor(),]), 
                    train=False), 
                    batch_size=1)    
    
    model.eval()

    pbar = enumerate(val_loader)
    pbar = tqdm(pbar, total=len(val_loader))  # progress bar
    
    losses = AverageMeter()
    mae_count_0, mae_count_1, mae_count_2, mae_count_T = 0, 0, 0, 0
    length = args.cell_size
    imgs, targets = [], []
    b_num = 0

    with open(os.path.join(args.log_dir,'results.txt'), 'w') as f:
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

                    if i == (ni-1) and j == (nj-1):
                        imgs = torch.stack(imgs, dim=0).squeeze(1)
                        targets = [ti for ti in targets if len(ti) != 0]
                        if not targets:
                            targets.append(torch.tensor([[-1, 0, 0, 0, 0]]))
                        targets = torch.cat(targets)

                        if CUDA:
                            imgs = imgs.cuda()
                            targets = targets.cuda()

                        with torch.no_grad():
                            predictions0, predictions1, predictions2, count = model(imgs, training=False)
                            loss, _, _, _ = compute_loss(predictions0, predictions1, predictions2, targets, count)  

                            losses.update(loss.item(), imgs.size(0))
                            
                            targets = targets.shape[0]
                            pred_count_0 = predictions0.sum()
                            pred_count_1 = predictions1.sum()
                            pred_count_2 = predictions2.sum()
                            count = count.sum()

                            mae_count_0 += abs(pred_count_0 - targets)
                            mae_count_1 += abs(pred_count_1 - targets)
                            mae_count_2 += abs(pred_count_2 - targets)
                            # print(count)
                            # print(targets)
                            mae_count_T += abs(count - targets)
                            # print(mae_count_T)
                            # print(type(count))
                            # print(type(pred_count_0))
                            s = '*Target {targets:.0f}\t *Pred_0 {pred_0:.3f}\t *Pred_1 {pred_1:.3f}\t *Pred_2 {pred_2:.3f}\t *MAE_0 {mae_0:.3f}\t *MAE_1 {mae_1:.3f}\t *MAE_2 {mae_2:.3f} \n'.\
                                format(targets=targets, pred_0=pred_count_0, pred_1=pred_count_1, pred_2=pred_count_2, \
                                    mae_0=(pred_count_0-targets), mae_1=(pred_count_1-targets), mae_2=(pred_count_2-targets) )
                            
                            f.writelines(s)

                            imgs = []
                            targets = []
                            b_num = 0
        
    mae_count_0 = mae_count_0/len(val_loader)
    mae_count_1 = mae_count_1/len(val_loader)
    mae_count_2 = mae_count_2/len(val_loader)
    mae_count_T = mae_count_T/len(val_loader)
    # wandb.log({'mae_count_0': mae_count_0})
    # wandb.log({'mae_count_1': mae_count_1})
    # wandb.log({'mae_count_2': mae_count_2})
    # wandb.log({'mae_count_3': mae_count_3})
    print(' * MAE_Count_0 {mae_0:.3f} '.format(mae_0=mae_count_0))
    print(' * MAE_Count_1 {mae_1:.3f} '.format(mae_1=mae_count_1))
    print(' * MAE_Count_2 {mae_2:.3f} '.format(mae_2=mae_count_2))
    print(' * MAE_Count_T {mae_T:.3f} '.format(mae_T=mae_count_T))

    return mae_count_0, mae_count_1, mae_count_2, mae_count_T, losses       
    return losses  


def main():
    args = parser.parse_args()

    args.best_mae = 1e6
    args.best_mae_0 = 1e6
    args.best_mae_1 = 1e6
    args.best_mae_2 = 1e6
    args.best_mae_T = 1e6

    args.log_dir = os.path.join(args.log_dir,args.model_desc)
    if not pathlib.Path(args.log_dir).exists():
        os.mkdir(args.log_dir)
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

    model = CSRNet(in_size=args.cell_size)

    if CUDA:
        model = model.cuda()

    #optimizer = torch.optim.SGD(model.parameters(), args.lr0, momentum=args.momentum, weight_decay=args.weight_decay)
    # wandb.watch(model, log='all')
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