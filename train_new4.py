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
from model_new4 import ComputeLoss, CSRNet
from dataset_new2 import listDataset
from tqdm import tqdm
import math
from utils_new2 import save_checkpoint, AverageMeter, vis_input, zeropad
import cv2
import glob


path = pathlib.Path(__file__).parent.absolute()
parser = argparse.ArgumentParser(description='RCVLab-AiimLab Crowd counting')

# GENERAL
parser.add_argument('--model_desc', default='train_new4/', help="Set model description")
parser.add_argument('--pre_model_desc', default='shanghaiA_pre/', help="Set model description")
parser.add_argument('--train_json', default=os.path.join(path,'datasets/shanghai/part_A_train.json'), help='path to train json')
parser.add_argument('--val_json', default=os.path.join(path,'datasets/shanghai/part_A_test.json'), help='path to test json')
parser.add_argument('--use_pre', default=False, type=bool, help='use the pretrained model?')
parser.add_argument('--use_gpu', default=True, action="store_false", help="Indicates whether or not to use GPU")
parser.add_argument('--device', default='1', type=str, help='GPU id to use.')
parser.add_argument('--checkpoint_path', default=os.path.join(path,'runs/weights'), type=str, help='checkpoint path')
parser.add_argument('--pre_checkpoint_path', default=os.path.join(path,'runs/weights'), type=str, help='checkpoint path')
parser.add_argument('--log_dir', default=os.path.join(path,'runs/log'), type=str, help='log dir')
parser.add_argument('--exp', default='shanghai', type=str, help='set dataset for training experiment')
parser.add_argument('--density', default=False, type=bool, help='using density map instead of head locations?')
parser.add_argument('--depth', default=True, type=bool, help='using depth?')

# MODEL
parser.add_argument('--model_file', default=path/'model.yaml')
parser.add_argument('--cell_size', default=128, type=int, help="cell size")
parser.add_argument('--threshold', default=0.01, type=int, help="threshold for the classification output")
parser.add_argument('--backend', default='vgg', type=str, help='vgg or resnet')

# TRAINING
parser.add_argument('--batch_size', default=512, type=int)
parser.add_argument('--epochs', default=300, type=int, help="Number of epochs to train for")
parser.add_argument('--workers', default=4, type=int, help="Number of workers in loading dataset")
parser.add_argument('--start_epoch', default=0, type=int, help="start_epoch")
parser.add_argument('--vis', default=False, type=bool, help='visualize the inputs') 
parser.add_argument('--lr0', default=0.00001, type=float, help="initial learning rate")
parser.add_argument('--weight_decay', default=0.0005, type=float, help="weight_decay")
parser.add_argument('--momentum', default=0.937, type=float, help="momentum")
parser.add_argument('--adam', default=True, type=bool, help='use torch.optim.Adam() optimizer') 
parser.add_argument('--dwa', default=True, type=bool, help='apply Dynamic Weight Average') 
parser.add_argument('--temp', default=2.0, type=float, help='temperature for DWA (must be positive)') 


def train(args, model, optimizer, train_list, val_list, train_list_depth, val_list_depth, tb_writer, CUDA):

    compute_loss = ComputeLoss(model)

    ##########
    T = args.temp
    avg_cost = np.zeros([args.epochs, 16], dtype=np.float64)
    lambda_weight = np.ones([4, args.epochs])
    ##########
    
    for epoch in range(args.start_epoch, args.epochs):

        train_loader = torch.utils.data.DataLoader(listDataset(train_list,
                       train_list_depth,
                       shuffle=True,
                       density=args.density,
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

        imgs, targets, imgs_depth = [], [], []

        #########
        cost = np.zeros(16, dtype=np.float32)

        if args.dwa:
            index = np.copy(epoch)
            if index == 0 or index == 1:
                lambda_weight[:, index] = 1.0
            else:
                w_1 = avg_cost[index - 1, 0] / avg_cost[index - 2, 0]
                w_2 = avg_cost[index - 1, 1] / avg_cost[index - 2, 1]
                w_3 = avg_cost[index - 1, 2] / avg_cost[index - 2, 2]
                w_4 = avg_cost[index - 1, 3] / avg_cost[index - 2, 3]
                lambda_weight[0, index] = 4 * np.exp(w_1 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T) + np.exp(w_3 / T) + np.exp(w_4 / T))
                lambda_weight[1, index] = 4 * np.exp(w_2 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T) + np.exp(w_3 / T) + np.exp(w_4 / T))
                lambda_weight[2, index] = 4 * np.exp(w_3 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T) + np.exp(w_3 / T) + np.exp(w_4 / T))
                lambda_weight[3, index] = 4 * np.exp(w_4 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T) + np.exp(w_3 / T) + np.exp(w_4 / T))

        #########
        
        for bi, (img_big, target_big, img_big_depth) in pbar:  # batch ----------
            data_time.update(time.time() - end)

            length_0 = img_big.size(2)
            length_1 = img_big.size(3)

            img_big_d = torch.zeros_like(img_big)
            img_big_d[:,0,:,:] = img_big_depth[:,0,:,:]
            img_big_d[:,1,:,:] = img_big_depth[:,0,:,:]
            img_big_d[:,2,:,:] = img_big_depth[:,0,:,:]
            
            if args.vis:
                vis_input(img_big.squeeze(0), target_big.squeeze(0))
                if args.depth:
                    vis_input(img_big.squeeze(0), img_big_depth.squeeze(0)[0, ...])

            if not args.density:
                coord = (target_big.squeeze(0)).nonzero(as_tuple=False)
                bxy = [[yb/length_0, xb/length_1] for (yb, xb) in coord]
                targets.append(torch.tensor(bxy))

            if args.depth:
                #img = torch.cat((img_big ,img_big_depth), dim=1)
                img = torch.clone(img_big)
                img_depth = torch.clone(img_big_d)

                img_depth = Variable(img_depth)
                imgs_depth.append(img_depth)
            else:
                img = torch.clone(img_big)

            img = Variable(img)
            imgs.append(img)

            if args.density:
                target_big = target_big.squeeze(0).numpy()
                target_big = cv2.resize(target_big, (target_big.shape[1]//8,target_big.shape[0]//8),interpolation = cv2.INTER_CUBIC)*64
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

            if CUDA:
                imgs = imgs.cuda()
                imgs_depth = imgs_depth.cuda()
                targets = targets.cuda()
            
            if epoch <= 1:
               tb_writer.add_graph(torch.jit.trace(model, imgs, strict=False), [])

            pred0, pred1, pred2, pred3 = model(imgs, imgs_depth, training=True)  # forward
            train_loss, lcount_0, lcount_1, lcount_2 = compute_loss(pred0, pred1, pred2, pred3, targets, epoch, eval=False) 

            #####  
            if args.dwa:
                loss = sum([lambda_weight[i, index] * train_loss[i] for i in range(len(train_loss))])
            else:
                logsigma = nn.Parameter(torch.FloatTensor([-0.5, -0.5, -0.5]))
                loss = sum(1 / (2 * torch.exp(logsigma[i])) * train_loss[i] + logsigma[i] / 2 for i in range(len(train_loss)))
            
            cost[0] = train_loss[0].item()
            cost[1] = train_loss[1].item()
            cost[2] = train_loss[2].item()
            cost[3] = train_loss[3].item()
            
            if args.density:
                targets = targets.sum()
            else:
                targets = targets.shape[0]

            pred_count_0 = pred0.sum()
            pred_count_1 = pred1.sum()
            pred_count_2 = pred2.sum()
            pred_count_3 = pred3.sum()

            mae_count_0 = abs(pred_count_0 - targets)
            mae_count_1 = abs(pred_count_1 - targets)
            mae_count_2 = abs(pred_count_2 - targets) 
            mae_count_3 = abs(pred_count_3 - targets)

            cost[4] = mae_count_0.item()
            cost[5] = mae_count_1.item()
            cost[6] = mae_count_2.item()
            cost[7] = mae_count_3.item()

            avg_cost[index, :8] += cost[:8] / len(train_loader)
            #####  
            
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
            imgs_depth = []
            targets = []

            # end batch ------------

        mae_count_0, mae_count_1, mae_count_2, mae_count_3, mae_best, val_losses = validate(args, val_list, val_list_depth, model, CUDA, compute_loss, cost, avg_cost, index, epoch)

        is_best = mae_count_3 < args.best_mae
        args.best_mae = min(mae_count_3, args.best_mae)
        print(' * * *')
        print(' * Best MAE {mae:.3f} '.format(mae=args.best_mae))
        print(' * * *')
        
        save_checkpoint({
            'epoch': epoch,
            'arch': args.checkpoint_path,
            'state_dict': model.state_dict(),
            'best_pred': args.best_mae,
            'optimizer' : optimizer.state_dict(),}, is_best, args.checkpoint_path)
        
        
        tb_writer.add_scalar('train loss/total', losses.avg, epoch)
        tb_writer.add_scalar('train loss/count_0', losses_count_0.avg, epoch)
        tb_writer.add_scalar('train loss/count_1', losses_count_1.avg, epoch)
        tb_writer.add_scalar('train loss/count_2', losses_count_2.avg, epoch)
        tb_writer.add_scalar('val loss/total', val_losses[0].avg, epoch)
        tb_writer.add_scalar('val loss/count_0', val_losses[1].avg, epoch)
        tb_writer.add_scalar('val loss/count_1', val_losses[2].avg, epoch)
        tb_writer.add_scalar('val loss/count_2', val_losses[3].avg, epoch)
        tb_writer.add_scalar('MAE/Count_0', mae_count_0, epoch)
        tb_writer.add_scalar('MAE/Count_1', mae_count_1, epoch)
        tb_writer.add_scalar('MAE/Count_2', mae_count_2, epoch)
        tb_writer.add_scalar('MAE/Count_total', mae_count_3, epoch)
        tb_writer.add_scalar('MAE/Count_bestOf3', mae_best, epoch)

        # end epoch ------------


def validate(args, val_list, val_list_depth, model, CUDA, compute_loss, cost, avg_cost, index, epoch):
    print ('begin validation')
    val_loader = torch.utils.data.DataLoader(listDataset(val_list, 
                    val_list_depth,
                    shuffle=False, 
                    density=args.density,
                    depth=args.depth,
                    transform1=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),]), 
                    transform2=transforms.Compose([transforms.ToTensor(),]), 
                    train=False), 
                    batch_size=1)    
    
    model.eval()

    pbar = enumerate(val_loader)
    pbar = tqdm(pbar, total=len(val_loader))  # progress bar
    
    losses = AverageMeter()
    losses_count_0 = AverageMeter()
    losses_count_1 = AverageMeter()
    losses_count_2 = AverageMeter()
    sum_mae_count_0, sum_mae_count_1, sum_mae_count_2, sum_mae_total_count, sum_best = 0, 0, 0, 0, 0
    imgs, targets, imgs_depth = [], [], []
    b_num = 0

    with open(os.path.join(args.log_dir,'results.txt'), 'w') as f:
        for bi, (img_big, target_big, img_big_depth) in pbar:  

            length_0 = img_big.size(2)
            length_1 = img_big.size(3)

            img_big_d = torch.zeros_like(img_big)
            img_big_depth = img_big_depth/65535.0
            img_big_d[:,0,:,:] = img_big_depth[:,0,:,:]
            img_big_d[:,1,:,:] = img_big_depth[:,0,:,:]
            img_big_d[:,2,:,:] = img_big_depth[:,0,:,:]

            if not args.density:
                coord = (target_big.squeeze(0)).nonzero(as_tuple=False)
                bxy = [[yb/length_0, xb/length_1] for (yb, xb) in coord]
                targets.append(torch.tensor(bxy))

            if args.depth:
                #img = torch.cat((img_chip ,img_chip_depth), dim=1)
                img = torch.clone(img_big)
                img_depth = torch.clone(img_big_d)
                imgs_depth.append(img_depth)
            else:
                img = torch.clone(img_big)

            imgs.append(img)

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

            if CUDA:
                imgs = imgs.cuda()
                imgs_depth = imgs_depth.cuda()
                targets = targets.cuda()

            with torch.no_grad():
                predictions0, predictions1, predictions2, predictions3 = model(imgs, imgs_depth, training=False)
                val_loss, lcount_0, lcount_1, lcount_2 = compute_loss(predictions0, predictions1, predictions2, predictions3, targets, epoch, eval=False)  

                
                losses.update(val_loss[0].item(), imgs.size(0))
                losses_count_0.update(lcount_0.item(), imgs.size(0))
                losses_count_1.update(lcount_1.item(), imgs.size(0))
                losses_count_2.update(lcount_2.item(), imgs.size(0))
                
                if args.density:
                    targets = targets.sum()
                else:
                    targets = targets.shape[0]

                pred_count_0 = predictions0.sum()
                pred_count_1 = predictions1.sum()
                pred_count_2 = predictions2.sum()
                pred_count_3 = predictions3.sum()
                #total_count = (pred_count_0+pred_count_1+pred_count_2)/3

                mae_count_0 = abs(pred_count_0 - targets)
                mae_count_1 = abs(pred_count_1 - targets)
                mae_count_2 = abs(pred_count_2 - targets) 
                mae_count_3 = abs(pred_count_3 - targets)

                #####
                cost[8] = val_loss[0].item()
                cost[9] = val_loss[1].item()
                cost[10] = val_loss[2].item()
                cost[11] = val_loss[3].item()
                
                cost[12] = mae_count_0.item()
                cost[13] = mae_count_1.item()
                cost[14] = mae_count_2.item()
                cost[15] = mae_count_3.item()
                avg_cost[index, 8:] += cost[8:] / len(val_loader)
                #####
                
                sum_mae_count_0 += mae_count_0
                sum_mae_count_1 += mae_count_1
                sum_mae_count_2 += mae_count_2
                sum_mae_total_count += mae_count_3

                if mae_count_0 < mae_count_1 and mae_count_0 < mae_count_2:
                    sum_best += mae_count_0
                elif mae_count_1 < mae_count_0 and mae_count_1 < mae_count_2:
                    sum_best += mae_count_1
                elif mae_count_2 < mae_count_0 and mae_count_2 < mae_count_1:
                    sum_best += mae_count_2
                else:
                    print('error')

                #sum_best += mae_count_0 if (mae_count_0 < mae_count_1) else mae_count_1

                s = '*Target {targets:.0f}\t *Pred_0 {pred_0:.3f}\t *Pred_1 {pred_1:.3f}\t *Pred_2 {pred_2:.3f}\t *MAE_0 {mae_0:.3f}\t *MAE_1 {mae_1:.3f}\t *MAE_2 {mae_2:.3f} \n'.\
                    format(targets=targets, pred_0=pred_count_0, pred_1=pred_count_1, pred_2=pred_count_2, \
                        mae_0=(pred_count_0-targets), mae_1=(pred_count_1-targets), mae_2=(pred_count_2-targets))
                
                f.writelines(s)

                imgs = []
                imgs_depth = []
                targets = []
        
    mae_count_0 = sum_mae_count_0/len(val_loader)
    mae_count_1 = sum_mae_count_1/len(val_loader)
    mae_count_2 = sum_mae_count_2/len(val_loader)
    mae_count_3 = sum_mae_total_count/len(val_loader)
    mae_best = sum_best/len(val_loader)

    print(' * MAE_Count_0 {mae_0:.3f} '.format(mae_0=mae_count_0))
    print(' * MAE_Count_1 {mae_1:.3f} '.format(mae_1=mae_count_1))
    print(' * MAE_Count_2 {mae_2:.3f} '.format(mae_2=mae_count_2))
    print(' * MAE_Count_3 {mae_T:.3f} '.format(mae_T=mae_count_3))
    print(' * MAE_BestOf3 {mae_best:.3f} '.format(mae_best=mae_best))
    print(' * * * COSTS (to adjust weights in multitasking loss)')
    print(' Epoch: {:04d} | TRAIN: {:.3f} {:.3f} {:.3f} {:.3f} || {:.3f} {:.3f} {:.3f} {:.3f} \n'
            '                TEST: {:.3f} {:.3f} {:.3f} {:.3f} || {:.3f} {:.3f} {:.3f} {:.3f} '
            .format(index, avg_cost[index, 0], avg_cost[index, 1], avg_cost[index, 2], avg_cost[index, 3],
                    avg_cost[index, 4], avg_cost[index, 5], avg_cost[index, 6], avg_cost[index, 7], avg_cost[index, 8],
                    avg_cost[index, 9], avg_cost[index, 10], avg_cost[index, 11], avg_cost[index, 12], avg_cost[index, 13],
                    avg_cost[index, 14], avg_cost[index, 15]))

    return mae_count_0, mae_count_1, mae_count_2, mae_count_3, mae_best, [losses, losses_count_0, losses_count_1, losses_count_2]


def main():
    args = parser.parse_args()

    args.best_mae = 1e6

    
    tb_writer = SummaryWriter(args.log_dir)

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

    model = CSRNet(backend=args.backend)

    if CUDA:
        model = model.cuda()

    if args.adam:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr0, betas=(args.momentum, 0.999))  # adjust beta1 to momentum
    else:
        optimizer = torch.optim.SGD(model.parameters(), args.lr0, momentum=args.momentum, weight_decay=args.weight_decay)
    
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