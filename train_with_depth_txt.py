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
from model import Model, ComputeLoss
from dataset_txt import listDataset
from tqdm import tqdm
import math
import itertools
from utils import save_checkpoint, AverageMeter, vis_input, zeropad


path = pathlib.Path(__file__).parent.absolute()
parser = argparse.ArgumentParser(description='RCVLab-AiimLab Crowd counting')

# GENERAL
parser.add_argument('--model_desc', default='shanghaiB_darknet_lr=1e-3/', help="Set model description")
parser.add_argument('--train_json', default=os.path.join(path,'datasets/shanghai/part_B_train.json'), help='path to train json')
parser.add_argument('--val_json', default=os.path.join(path,'datasets/shanghai/part_B_test.json'), help='path to test json')
parser.add_argument('--pre', '-p', default=os.path.join(path,'runs_txt_test/weights/checkpoint.pth.tar'), type=str, help='path to the pretrained model')
parser.add_argument('--use_pre', default=True, type=bool, help='use the pretrained model?')
parser.add_argument('--use_gpu', default=True, action="store_false", help="Indicates whether or not to use GPU")
parser.add_argument('--device', default='6', type=str, help='GPU id to use.')
parser.add_argument('--checkpoint_path', default=os.path.join(path,'runs_txt_test/weights'), type=str, help='checkpoint path')
parser.add_argument('--log_dir', default=os.path.join(path,'runs_txt_test/log'), type=str, help='log dir')
parser.add_argument('--exp', default='shanghai', type=str, help='set dataset for training experiment')

# MODEL
parser.add_argument('--model_file', default=os.path.join(path,'model.yaml'))
parser.add_argument('--cell_size', default=64, type=int, help="cell size")
parser.add_argument('--threshold', default=0.9, type=int, help="threshold for the classification output")


# TRAINING
parser.add_argument('--batch_size', default=512, type=int)
parser.add_argument('--epochs', default=1000, type=int, help="Number of epochs to train for")
parser.add_argument('--workers', default=4, type=int, help="Number of workers in loading dataset")
parser.add_argument('--start_epoch', default=0, type=int, help="start_epoch")
parser.add_argument('--vis', default=False, type=bool, help='visualize the inputs') 
parser.add_argument('--lr0', default=0.00001, type=float, help="initial learning rate")
parser.add_argument('--weight_decay', default=0.0005, type=float, help="weight_decay")
parser.add_argument('--momentum', default=0.937, type=float, help="momentum")
parser.add_argument('--adam', default=False, type=bool, help='use torch.optim.Adam() optimizer') 


def train(args, model, optimizer, train_list, val_list, train_list_depth, val_list_depth, tb_writer, CUDA):

    compute_loss = ComputeLoss(model)
    # print(train_list)
    for epoch in range(args.start_epoch, args.epochs):
        train_loader = torch.utils.data.DataLoader(listDataset(train_list,
                       train_list_depth,
                       shuffle=True,
                       depth=False,
                       transform1=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),]), 
                       transform2=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.502], std=[0.291]),]), 
                       train=True, 
                       seen=0,
                       batch_size=1,
                       num_workers=args.workers))

        losses = AverageMeter()
        losses_obj = AverageMeter()
        losses_num = AverageMeter()
        data_time = AverageMeter()
        batch_time = AverageMeter()
        end = time.time()

        model.train()

        pbar = enumerate(train_loader)
        pbar = tqdm(pbar, total = len(train_loader))  # progress bar

        optimizer.zero_grad()

        imgs, targets, imgs_depth = [], [], []
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

                    img_chip_depth = img_big_depth[:, :, y1:y2, x1:x2]
                    img_chip_depth  = zeropad(img_chip_depth.squeeze(0).permute(1,2,0).numpy(), length - img_chip_depth.shape[2], length - img_chip_depth.shape[3])
                    img_chip_depth = torch.from_numpy(img_chip_depth).unsqueeze(2)
                    img_chip_depth = img_chip_depth.permute(2,0,1).unsqueeze(0)

                    target_chip = target_big[:, y1:y2, x1:x2]
                    target_chip = zeropad(target_chip.squeeze(0).numpy(), length - img_chip.shape[2], length - img_chip.shape[3], target=True)
                    target_chip = torch.from_numpy(target_chip).unsqueeze(0)
                    assert img_chip_depth.shape[2] == img_chip_depth.shape[3] == length, 'image size error'
                    assert img_chip.shape[2] == img_chip.shape[3] == length, 'image size error'
                    assert target_chip.shape[1] == target_chip.shape[2] == length, 'target size error'
            
                    if args.vis:
                        vis_input(img_chip, target_chip)
                    coord = (target_chip.squeeze(0)).nonzero(as_tuple=False)

                    bxy = [[b_num, yb/length, xb/length] for (yb, xb) in coord]
                    targets.append(torch.tensor(bxy))
                    img = torch.cat((img_chip ,img_chip_depth), dim=1)
                    img = Variable(img)
                    imgs.append(img)
                    b_num += 1

                    if b_num >= args.batch_size:
                        
                        imgs = torch.stack(imgs, dim=0).squeeze(1)
                        targets = [ti for ti in targets if len(ti) != 0]
                        targets = torch.cat(targets)

                        if CUDA:
                            imgs = imgs.float()
                            imgs = imgs.cuda()
                            targets = targets.cuda()
                        
                        if epoch <= 1:
                            tb_writer.add_graph(torch.jit.trace(model, imgs, strict=False), [])

                        pred = model(imgs, training=True)  # forward
                        loss, lobj, lnum = compute_loss(pred, targets) 

                        losses.update(loss.item(), imgs.size(0))
                        losses_obj.update(lobj.item(), imgs.size(0))
                        losses_num.update(lnum.item(), imgs.size(0))

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

        pred, predByNum, val_losses = validate(args, val_list, val_list_depth, model, CUDA, compute_loss)

        is_best = pred < args.best_pred
        args.best_pred = min(pred, args.best_pred)
        print(' * best MAE {mae:.3f} '.format(mae=args.best_pred))
        
        save_checkpoint({
            'epoch': epoch,
            'arch': args.checkpoint_path,
            'state_dict': model.state_dict(),
            'best_pred': args.best_pred,
            'optimizer' : optimizer.state_dict(),}, is_best, args.checkpoint_path)
        
        tb_writer.add_scalar('train loss/total', losses.avg, epoch)
        tb_writer.add_scalar('train loss/obj', losses_obj.avg, epoch)
        tb_writer.add_scalar('train loss/num', losses_num.avg, epoch)
        tb_writer.add_scalar('val loss/total', val_losses.avg, epoch)
        tb_writer.add_scalar('MAE/average', pred, epoch)
        tb_writer.add_scalar('MAE/averageByNum', predByNum, epoch)

        # end epoch ------------


def validate(args, val_list, val_list_depth, model, CUDA, compute_loss):
    print ('begin validation')
    val_loader = torch.utils.data.DataLoader(listDataset(val_list, 
                    val_list_depth,
                    shuffle=False,
                    depth=False, 
                    transform1=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),]), 
                    transform2=transforms.Compose([transforms.ToTensor(),]), 
                    train=False), 
                    batch_size=1)    
    
    model.eval()

    pbar = enumerate(val_loader)
    pbar = tqdm(pbar, total=len(val_loader))  # progress bar
    
    losses = AverageMeter()
    mae = 0
    maeByNum = 0
    length = args.cell_size
    imgs, targets, imgs_depth = [], [], []
    b_num = 0

    with open(args.log_dir + 'results.txt', 'w') as f:
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

                    img_chip_depth = img_big_depth[:, :, y1:y2, x1:x2]
                    img_chip_depth  = zeropad(img_chip_depth.squeeze(0).permute(1,2,0).numpy(), length - img_chip_depth.shape[2], length - img_chip_depth.shape[3])
                    img_chip_depth = torch.from_numpy(img_chip_depth).unsqueeze(2)
                    img_chip_depth = img_chip_depth.permute(2,0,1).unsqueeze(0)
                    
                    target_chip = target_big[:, y1:y2, x1:x2]
                    target_chip = zeropad(target_chip.squeeze(0).numpy(), length - img_chip.shape[2], length - img_chip.shape[3], target=True)
                    target_chip = torch.from_numpy(target_chip).unsqueeze(0)
                    assert img_chip_depth.shape[2] == img_chip_depth.shape[3] == length, 'image size error'
                    assert img_chip.shape[2] == img_chip.shape[3] == length, 'image size error'
                    assert target_chip.shape[1] == target_chip.shape[2] == length, 'target size error'

                    coord = (target_chip.squeeze(0)).nonzero(as_tuple=False)

                    bxy = [[b_num, yb/length, xb/length] for (yb, xb) in coord]
                    targets.append(torch.tensor(bxy))
                    img = torch.cat((img_chip ,img_chip_depth), dim=1)
                    img = torch.clone(img)
                    imgs.append(img)


                    b_num += 1

                    if i == (ni-1) and j == (nj-1):
                        imgs = torch.stack(imgs, dim=0).squeeze(1)
                        targets = [ti for ti in targets if len(ti) != 0]
                        targets = torch.cat(targets)

                        if CUDA:
                            imgs = imgs.float()
                            imgs = imgs.cuda()
                            targets = targets.cuda()

                        with torch.no_grad():
                            predictions = model(imgs, training=False)
                            loss, _, _ = compute_loss(predictions, targets)  # loss scaled by batch_size

                            losses.update(loss.item(), imgs.size(0))
                            
                            #pred = pred > args.threshold
                            pred = predictions[:, 0, :, :].sum()
                            predByNum = predictions[:, 1, ...].view(predictions.size(0), -1).mean(1, keepdim=True)
                            predByNum = predByNum.sum()

                            targets = targets.shape[0]
                            
                            mae += abs(pred - targets)
                            maeByNum += abs(predByNum - targets)

                            s = '*Target {targets:.2f}\t *Pred_prob {pred:.4f}\t *Pred_num {predByNum:.4f}\t MAE_prob {mae:.4f}\t *MAE_num {maeByNum:.4f}\n'.format(targets=targets, pred=pred, predByNum=predByNum, mae=(pred-targets), maeByNum=(predByNum-targets))
                            f.writelines(s)

                            imgs = []
                            targets = []
                            b_num = 0
        
    mae = mae/len(val_loader)
    maeByNum = maeByNum/len(val_loader)
    print(' * MAE {mae:.3f} '.format(mae=mae))
    print(' * MAEByNum {mae:.3f} '.format(mae=maeByNum))

    return mae, maeByNum, losses       


def main():
    args = parser.parse_args()

    args.best_pred = 1e6
    args.log_dir = os.path.join(args.log_dir,args.model_desc)
    tb_writer = SummaryWriter(args.log_dir)

    args.checkpoint_path += ('/'+args.model_desc)
    if not pathlib.Path(args.checkpoint_path).exists():
        os.mkdir(args.checkpoint_path)
    args.checkpoint_path += 'checkpoint.pth.tar'

    if args.exp == 'shanghai':
        with open(args.train_json, 'r') as outfile:        
            train_list_fab = json.load(outfile)
        with open(args.val_json, 'r') as outfile:       
            val_list_fab = json.load(outfile)

        train_list = [st.replace('/home/leeyh/Downloads/Shanghai/part_B_final/train_data/images', 'crowdcounting_main/datasets/shanghai/part_B_final/train_data/images') for st in train_list_fab]
        val_list = [st.replace('/home/leeyh/Downloads/Shanghai/part_B_final/test_data/images', 'crowdcounting_main/datasets/shanghai/part_B_final/test_data/images') for st in val_list_fab]
        
        train_list_depth = [st.replace('/home/leeyh/Downloads/Shanghai/part_B_final/train_data/images', 'crowdcounting_main/datasets/shanghai/part_B_final/train_data/depth_resized_h5') for st in train_list_fab]
        val_list_depth = [st.replace('/home/leeyh/Downloads/Shanghai/part_B_final/test_data/images', 'crowdcounting_main/datasets/shanghai/part_B_final/test_data/depth_resized_h5') for st in val_list_fab]
        train_list_depth = [st.replace('.jpg', '.h5') for st in train_list_depth]
        val_list_depth = [st.replace('.jpg', '.h5') for st in val_list_depth]
        print(train_list_depth)
        print(train_list_depth)

    if args.use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device
        torch.cuda.manual_seed(time.time())
        CUDA =True
    else:
        CUDA = False

    model = Model(args.model_file)

    if CUDA:
        model = model.cuda()

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)  # no decay
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay

    if args.adam:
        optimizer = torch.optim.Adam(pg0, lr=args.lr0, betas=(args.momentum, 0.999))  # adjust beta1 to momentum
    else:
        optimizer = torch.optim.SGD(pg0, lr=args.lr0, momentum=args.momentum, nesterov=True)

    optimizer.add_param_group({'params': pg1, 'weight_decay': args.weight_decay})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    del pg0, pg1, pg2

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


    train(args, model, optimizer, train_list, val_list, train_list_depth, val_list_depth, tb_writer, CUDA) 


if __name__ == '__main__':
    main() 