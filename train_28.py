import os
import argparse
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch.autograd import Variable
import pathlib
from model_28 import ComputeLoss, CSRNet
from dataset_20 import listDataset
from tqdm import tqdm
import math
from utils import save_checkpoint, AverageMeter, vis_input, zeropad
import cv2
import glob
import wandb
from matplotlib import pyplot as plt



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
parser.add_argument('--model_desc', default='train_28/', help="Set model description")
parser.add_argument('--pre_model_desc', default='train_20/', help="Set model description")
parser.add_argument('--train_json', default=os.path.join(path,'datasets/shanghai/part_A_train.json'), help='path to train json')
parser.add_argument('--val_json', default=os.path.join(path,'datasets/shanghai/part_A_test.json'), help='path to test json')
parser.add_argument('--use_pre', default=False, type=bool, help='use the pretrained model?')
parser.add_argument('--use_gpu', default=True, action="store_false", help="Indicates whether or not to use GPU")
parser.add_argument('--device', default='6', type=str, help='GPU id to use.')
parser.add_argument('--checkpoint_path', default=os.path.join(path,'runs/weights'), type=str, help='checkpoint path')
parser.add_argument('--pre_checkpoint_path', default=os.path.join(path,'runs/weights'), type=str, help='checkpoint path')
parser.add_argument('--log_dir', default=os.path.join(path,'runs/log'), type=str, help='log dir')
parser.add_argument('--exp', default='shanghai', type=str, help='set dataset for training experiment')
parser.add_argument('--depth', default=True, type=bool, help='using depth?')

# MODEL
parser.add_argument('--model_file', default=path/'model.yaml')
parser.add_argument('--cell_size', default=128, type=int, help="cell size")
parser.add_argument('--threshold', default=0.01, type=int, help="threshold for the classification output")

# TRAINING
parser.add_argument('--batch_size', default=512, type=int)
parser.add_argument('--epochs', default=200, type=int, help="Number of epochs to train for")
parser.add_argument('--workers', default=4, type=int, help="Number of workers in loading dataset")
parser.add_argument('--start_epoch', default=0, type=int, help="start_epoch")
parser.add_argument('--vis', default=False, type=bool, help='visualize the inputs') 
parser.add_argument('--lr0', default=0.0001, type=float, help="initial learning rate")
parser.add_argument('--weight_decay', default=0.0005, type=float, help="weight_decay")
parser.add_argument('--momentum', default=0.937, type=float, help="momentum")
parser.add_argument('--adam', default=False, type=bool, help='use torch.optim.Adam() optimizer') 


def train(args, model, optimizer, train_list, val_list, train_list_depth, val_list_depth, tb_writer, CUDA):

    compute_loss = ComputeLoss(model, in_size=args.cell_size)

    n = 0
    for epoch in range(args.start_epoch, args.epochs):
        train_loader = torch.utils.data.DataLoader(listDataset(train_list,
                       train_list_depth,
                       shuffle=True,
                       depth=args.depth,
                       transform1=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),]), 
                       transform2=transforms.Compose([transforms.ToTensor(), transforms.ConvertImageDtype(torch.float),transforms.Normalize(mean=[19.6193], std=[855.685]),]), 
                       train=True, 
                       seen=0,
                       batch_size=args.batch_size,
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

        imgs, imgs_depth, targets, density = [], [], [], []
        length = args.cell_size
        b_num = 0

        for bi, (imgs, targets, imgs_depth, density) in pbar:  # batch ----------
            n = n + 1
            density = density.unsqueeze(0)
            targets = targets.unsqueeze(0)
            # print(density.sum())
            # print(targets.sum())
            if imgs.shape[-1]%8 != 0 or imgs.shape[-2]%8 != 0:
                z4 = nn.ZeroPad2d([0, 8-(imgs.shape[-1]%8), 0, 8-(imgs.shape[-2]%8)])
                imgs = z4(imgs)
                targets = z4(targets)
                imgs_depth = z4(imgs_depth)
                density = z4(density)

            # density = cv2.resize(density.numpy(),(density.shape[2]//8,density.shape[1]//8),interpolation = cv2.INTER_CUBIC)*64
            # density = density.unsqueeze(0)
            density = F.interpolate(density, size=(density.shape[2]//8,density.shape[3]//8), mode='bilinear', align_corners=False)*64
            # print(density.sum())
            data_time.update(time.time() - end)
            # print('density',density.shape)
            # print('imgs_depth',imgs_depth.shape)
            # print('targets',targets.shape)
            # print('imgs',imgs.shape)
            img_d = torch.zeros_like(imgs)
            imgs_depth = imgs_depth.unsqueeze(0)
            img_d[:,0,:,:] = imgs_depth[:,0,:,:]
            img_d[:,1,:,:] = imgs_depth[:,0,:,:]
            img_d[:,2,:,:] = imgs_depth[:,0,:,:]

            if CUDA:
                imgs = imgs.cuda()
                imgs_depth = img_d.cuda()
                targets = targets.cuda()
                density = density.cuda()



            # if epoch <= 1:
            #    tb_writer.add_graph(torch.jit.trace(model, imgs, imgs_depth, strict=False), [])

            pred0, count, density_out = model(imgs, imgs_depth, training=True)  # forward
            # print('pred0', pred0[0].sum())
            # print('pred1', pred1[0].sum())
            # print('pred2', pred2[0].sum())
            # print(density.shape)
            # print('d', density.sum())
            # print('do', density_out.sum())
            # print('tar', targets.sum())
            loss = compute_loss(pred0, targets, count, density, density_out) 
            # wandb.log({'train_loss': loss, 'lcount_0': lcount_0, 'lcount_1': lcount_1, 'lcount_2': lcount_2})
            losses.update(loss.item(), imgs.size(0))
            # losses_count_0.update(lcount_0.item(), imgs.size(0))
            # losses_count_1.update(lcount_1.item(), imgs.size(0))
            # losses_count_2.update(lcount_2.item(), imgs.size(0))



            optimizer.zero_grad()
            # grads = model.grad
            loss.backward()
            
            plot_grad_flow(model.named_parameters(), tb_writer, n)
            optimizer.step()    
            
            batch_time.update(time.time() - end)
            end = time.time()

            s = ('Epoch [{0}][{1}/{2}] '
                'Time/b {batch_time.val:.2f} ({batch_time.avg:.2f}) '
                'Loss {loss.val:.4f} ({loss.avg:.3f}) '
                .format(epoch, bi, len(train_loader), batch_time=batch_time, loss=losses))
            
            pbar.set_description(s)


            # end batch ------------

        mae_count_0, mae_count_T, mae_count_den, val_losses = validate(args, val_list, val_list_depth, model, CUDA, compute_loss)
        # val_losses = validate(args, val_list, val_list_depth, model, CUDA, compute_loss)

        is_best_0 = mae_count_0 < args.best_mae_0
        args.best_mae_0 = min(mae_count_0, args.best_mae_0)
        print(' * Best MAE_0 {mae:.3f} '.format(mae=args.best_mae_0))
        # wandb.log({'best_mae_0': args.best_mae_0})
        # wandb.log({'best_mae_2': args.best_mae_2})
        is_best_T = mae_count_T < args.best_mae_T
        args.best_mae_T = min(mae_count_T, args.best_mae_T)
        print(' * Best MAE_T {mae:.3f} '.format(mae=args.best_mae_T))
        # wandb.log({'best_mae_3': args.best_mae_3})
        is_best_den = mae_count_den < args.best_mae_den
        args.best_mae_den = min(mae_count_den, args.best_mae_den)
        print(' * Best MAE_den {mae:.3f} '.format(mae=args.best_mae_den))
        save_checkpoint({
            'epoch': epoch,
            'arch': args.checkpoint_path,
            'state_dict': model.state_dict(),
            'best_pred': args.best_mae_0,
            'optimizer' : optimizer.state_dict(),}, is_best_0, args.checkpoint_path)
        
        # tb_writer.add_scalar('train loss/total', losses.avg, epoch)
        # tb_writer.add_scalar('train loss/count_0', losses_count_0.avg, epoch)
        # tb_writer.add_scalar('train loss/count_1', losses_count_1.avg, epoch)
        # tb_writer.add_scalar('train loss/count_2', losses_count_2.avg, epoch)
        # tb_writer.add_scalar('val loss/total', val_losses.avg, epoch)
        tb_writer.add_scalar('MAE/Count_0', mae_count_0, epoch)
        # tb_writer.add_scalar('MAE/Count_1', mae_count_1, epoch)
        # tb_writer.add_scalar('MAE/Count_2', mae_count_2, epoch)
        # tb_writer.add_scalar('MAE/Count_T', mae_count_T, epoch)
        # tb_writer.add_scalar('MAE/Count_den', mae_count_den, epoch)

        # end epoch ------------

def plot_grad_flow(named_parameters, tb_writer, ep):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n) and torch.is_tensor(p.grad):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().cpu().numpy())

    # print(ave_grads)
    # plt.plot(ave_grads, alpha=0.3, color="b")
    # plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    # plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    # plt.xlim(xmin=0, xmax=len(ave_grads))
    # plt.xlabel("Layers")
    # plt.ylabel("average gradient")
    # plt.title("Gradient flow")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig('crowd_counting/pltforgrad.jpg')
    for lay, grad in zip(layers, ave_grads):
        names = lay.split('.')
        # print(grad)
        # print(ep)
        # print(names)
        if len(names) >2:
            tb_writer.add_scalar('grads/'+names[0]+'_'+names[1]+'_'+names[2], grad, ep)
        else:
            tb_writer.add_scalar('grads/'+names[0]+'_'+names[1], grad, ep)

def validate(args, val_list, val_list_depth, model, CUDA, compute_loss):
    print ('begin validation')
    val_loader = torch.utils.data.DataLoader(listDataset(val_list, 
                    val_list_depth,
                    shuffle=False, 
                    depth=args.depth,
                    transform1=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),]), 
                    transform2=transforms.Compose([transforms.ToTensor(), transforms.ConvertImageDtype(torch.float),transforms.Normalize(mean=[19.6193], std=[855.685]),]), 
                    train=False), 
                    batch_size=1)    
    
    model.eval()

    pbar = enumerate(val_loader)
    pbar = tqdm(pbar, total=len(val_loader))  # progress bar
    
    losses = AverageMeter()
    mae_count_0, mae_count_1, mae_count_2, mae_count_T, mae_count_den = 0, 0, 0, 0, 0
    length = args.cell_size
    imgs, imgs_depth, targets, density = [], [], [], []
    b_num = 0

    with open(os.path.join(args.log_dir,'results.txt'), 'w') as f:
        for bi, (imgs, targets, imgs_depth, density) in pbar:

            density = density.unsqueeze(0)
            targets = targets.unsqueeze(0)

            if imgs.shape[-1]%8 != 0 or imgs.shape[-2]%8 != 0:
                z4 = nn.ZeroPad2d([0, 8-(imgs.shape[-1]%8), 0, 8-(imgs.shape[-2]%8)])
                imgs = z4(imgs)
                targets = z4(targets)
                imgs_depth = z4(imgs_depth)
                density = z4(density)

            # density = cv2.resize(density.numpy(),(density.shape[2]//8,density.shape[1]//8),interpolation = cv2.INTER_CUBIC)*64
            # density = density.unsqueeze(0)
            density = F.interpolate(density, size=(density.shape[2]//8,density.shape[3]//8), mode='bicubic', align_corners=False)*64
            img_d = torch.zeros_like(imgs)

            img_d[:,0,:,:] = imgs_depth[:,0,:,:]
            img_d[:,1,:,:] = imgs_depth[:,0,:,:]
            img_d[:,2,:,:] = imgs_depth[:,0,:,:]


            if CUDA:
                imgs = imgs.cuda()
                imgs_depth = img_d.cuda()
                targets = targets.cuda()
                density = density.cuda()

            with torch.no_grad():
                predictions0, count, density_out = model(imgs, imgs_depth, training=False)
                loss = compute_loss(predictions0, targets, count, density, density_out)  

                losses.update(loss.item(), imgs.size(0))
                
                targets = targets.sum()
                pred_count_0 = predictions0.sum()

                count = count.sum()

                mae_count_0 += abs(pred_count_0.sum() - targets.sum())

                # print(count)
                # print(targets)
                mae_count_T += abs(count - density.sum())
                mae_count_den += abs(density_out.sum() - density.sum())
                # print(mae_count_T)
                # print(type(count))
                # print(type(pred_count_0))
                # s = '*Target {targets:.0f}\t *Pred_0 {pred_0:.3f}\t *Pred_1 {pred_1:.3f}\t *Pred_2 {pred_2:.3f}\t *MAE_0 {mae_0:.3f}\t *MAE_1 {mae_1:.3f}\t *MAE_2 {mae_2:.3f} \n'.\
                #     format(targets=targets, pred_0=pred_count_0, pred_1=pred_count_1, pred_2=pred_count_2, \
                #         mae_0=(pred_count_0-targets), mae_1=(pred_count_1-targets), mae_2=(pred_count_2-targets) )
                
                # f.writelines(s)

        
    mae_count_0 = mae_count_0/len(val_loader)
    mae_count_T = mae_count_T/len(val_loader)
    mae_count_den = mae_count_den/len(val_loader)
    # wandb.log({'mae_count_0': mae_count_0})
    # wandb.log({'mae_count_1': mae_count_1})
    # wandb.log({'mae_count_2': mae_count_2})
    # wandb.log({'mae_count_3': mae_count_3})
    print(' * MAE_Count_0 {mae_0:.3f} '.format(mae_0=mae_count_0))

    print(' * MAE_Count_T {mae_T:.3f} '.format(mae_T=mae_count_T))
    print(' * MAE_Count_den {mae_den:.3f} '.format(mae_den=mae_count_den))

    return mae_count_0, mae_count_T, mae_count_den, losses       
    # return losses  


def main():
    args = parser.parse_args()

    args.best_mae = 1e6
    args.best_mae_0 = 1e6
    args.best_mae_1 = 1e6
    args.best_mae_2 = 1e6
    args.best_mae_T = 1e6
    args.best_mae_den = 1e6

    args.log_dir = os.path.join(args.log_dir,args.model_desc)
    if not pathlib.Path(args.log_dir).exists():
        os.mkdir(args.log_dir)
    files = glob.glob(os.path.normpath(os.path.join(args.log_dir,'*')))
    for f in files:
        os.remove(f)
    tb_writer = SummaryWriter(args.log_dir)

    args.pre_checkpoint_path += ('/'+args.pre_model_desc)
    args.pre_checkpoint_path = os.path.join(args.pre_checkpoint_path,'model_best.pth.tar')

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
            train_list_depth = [st.replace('images', 'depth_boosted_leres') for st in train_list]
            val_list_depth = [st.replace('images', 'depth_boosted_leres') for st in val_list]
            train_list_depth = [st.replace('.jpg', '.png') for st in train_list_depth]
            val_list_depth = [st.replace('.jpg', '.png') for st in val_list_depth]
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
        if os.path.isfile(args.pre_checkpoint_path):
            print("=> loading checkpoint '{}'".format(args.pre_checkpoint_path))
            checkpoint = torch.load(args.pre_checkpoint_path)
            args.start_epoch = 0
            args.best_mae = checkpoint['best_pred']
            
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            # optimizer.load_state_dict(checkpoint['optimizer'], strict=False)
            print("=> loaded checkpoint '{}' (epoch {})".format(args.pre_checkpoint_path, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.pre_checkpoint_path))

    train(args, model, optimizer, train_list, val_list, train_list_depth, val_list_depth, tb_writer, CUDA) 


if __name__ == '__main__':
    main() 