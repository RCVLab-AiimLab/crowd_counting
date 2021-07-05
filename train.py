import sys
import os

import warnings

from model import CSRNet

from utils import save_checkpoint

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import argparse
import json
import cv2
import dataset
import time
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import cm as CM
import math

import pathlib
path = pathlib.Path(__file__).parent.absolute()


parser = argparse.ArgumentParser(description='RCVLab-AmiiLab Crowd counting')

parser.add_argument('--train_json', metavar='TRAIN', default=path/'part_A_train.json', help='path to train json')
parser.add_argument('--test_json', metavar='TEST', default=path/'part_A_val.json', help='path to test json')
parser.add_argument('--pre', '-p', metavar='PRETRAINED', default='../runs/weights/checkpoint.pth.tar', type=str, help='path to the pretrained model')
parser.add_argument('--use_pre', metavar='USEPRETRAINED', default=False, type=bool, help='use the pretrained model?')
parser.add_argument('--use_gpu', default=True, action="store_false", help="Indicates whether or not to use GPU")
parser.add_argument('--gpu_id', metavar='GPU_ID', default='0', type=str, help='GPU id to use.')
parser.add_argument('--checkpoint_path', metavar='CHECKPOINT', default='../runs/weights', type=str, help='checkpoint path')
parser.add_argument('--log_dir', metavar='CHECKPOINT', default='../runs/log', type=str, help='log dir')

parser.add_argument('--depth', metavar='DEPTH', default=False, type=bool, help='using depth?')
parser.add_argument('--input_size', metavar="input_size", default=256, type=int)

parser.add_argument('--lr', metavar="learning_rate", default=0.001, type=float, help="learning rate")
parser.add_argument('--decoder', metavar="decoder", default='FC', help="Decoder structure 'FC' or 'Conv'")
parser.add_argument('--batch_size', metavar="batch_size", default=1, type=int)
parser.add_argument('--epochs', metavar="epochs", default=300, type=int, help="Number of epochs to train for")
parser.add_argument('--file', metavar="filepath", default="", type=str, help="Name of the model to be loaded")
parser.add_argument('--save_images', default=True, action="store_false", help="Set if you want to save reconstruction results each epoch")
parser.add_argument('--alpha', default=0.0005, type=float, help="Alpha constant from paper (Amount of reconstruction loss)")
parser.add_argument('--dataset', default='shanghai', help="Set wanted dataset. Options: [mnist, small_norb,cifar10]")
parser.add_argument('--routing', metavar="routing_iterations", default=3, type=int, help="Number of routing iterations to use")
parser.add_argument('--logfile', metavar="log_filepath", default="", type=str, help="Path to previous logfile if continuing training")
parser.add_argument('--batch_norm', default=False, type=int, help="Turn on/off batch norm in encoder/decoder")
parser.add_argument('--loss', metavar="loss_type", default="L2", help="Define reconstruction loss. Types: [L1, L2]")
parser.add_argument('--anneal_alpha', default="none", help="Set annealing function for alpha. Options: [none, 1, 2]")
parser.add_argument('--leaky', default=False, action="store_true",  help="Turn on/off leaky routing (Add orphan class for reconstruction)")
parser.add_argument('--model', default='model', help="Set model name")


global args

args = parser.parse_args()
args.original_lr = args.lr # 1e-7
#args.lr = 1e-7
args.momentum      = 0.95
args.decay         = 5*1e-4
args.start_epoch   = 0
args.steps         = [-1,1,100,150]
args.scales        = [1,1,1,1]
args.workers = 4
args.seed = time.time()
args.print_freq = 1

tb_writer = SummaryWriter(args.log_dir)
   
def main():
    
    best_pred = 1e6
    
    if not Path(args.checkpoint_path).exists():
        os.mkdir(args.checkpoint_path)
    args.checkpoint_path += '/checkpoint.pth.tar'

    with open(args.train_json, 'r') as outfile:        
        train_list = json.load(outfile)
    with open(args.test_json, 'r') as outfile:       
        val_list = json.load(outfile)

    train_list = [st.replace('/home/leeyh/Downloads/Shanghai', '/media/mohsen/myDrive/datasets/ShanghaiTech_Crowd_Counting_Dataset') for st in train_list]
    val_list = [st.replace('/home/leeyh/Downloads/Shanghai', '/media/mohsen/myDrive/datasets/ShanghaiTech_Crowd_Counting_Dataset') for st in val_list]
    
    if args.use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
        torch.cuda.manual_seed(args.seed)
        CUDA =True
    else:
        CUDA = False

    model = CSRNet(args.depth, reconstruction_type=args.decoder, imsize=args.input_size//8, 
                        routing_iterations=args.routing, primary_caps_gridsize=8,
                        img_channel=3, batchnorm=args.batch_norm, num_primary_capsules=32,
                        loss=args.loss, leaky_routing=args.leaky)
    
    if CUDA:
        model = model.cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.decay)

    if args.pre and args.use_pre:
        if os.path.isfile(args.pre):
            print("=> loading checkpoint '{}'".format(args.pre))
            checkpoint = torch.load(args.pre)
            args.start_epoch = checkpoint['epoch']
            best_pred = checkpoint['best_pred']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.pre, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.pre))
            
    for epoch in range(args.start_epoch, args.epochs):
        
        adjust_learning_rate(optimizer, epoch)

        # Annealing alpha
        alpha = get_alpha(epoch)
        
        train(train_list, val_list, model, optimizer, epoch, alpha, best_pred, CUDA)

def zeropad(img, h, w, target=False):
    if not target:
        color = [0, 0, 0]
        padded = cv2.copyMakeBorder(img, 0, h, 0, w, cv2.BORDER_CONSTANT, value=color)
    else:
        padded = cv2.copyMakeBorder(img, 0, h, 0, w, cv2.BORDER_CONSTANT, value=0)
    return padded

def train(train_list, val_list, model, optimizer, epoch, alpha, best_pred, CUDA):
    
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    
    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(train_list,
                       shuffle=True,
                       transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),]), 
                       train=True, 
                       seen=model.seen,
                       batch_size=args.batch_size,
                       num_workers=args.workers,
                       img_size=args.input_size),
                    batch_size=args.batch_size)
    print('epoch %d, processed %d samples, lr %.10f' % (epoch, epoch * len(train_loader.dataset), args.lr))
    
    model.train()
    end = time.time()

    length = 32
    imgs, targets = [], []
    b_num = 0
    for bi, (img_big, target_big) in enumerate(train_loader):
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
                target_chip = target_big[:, y1:y2, x1:x2]
                target_chip = zeropad(target_chip.squeeze(0).numpy(), length - img_chip.shape[2], length - img_chip.shape[3], target=True)
                target_chip = torch.from_numpy(target_chip).unsqueeze(0)
                assert img_chip.shape[2] == img_chip.shape[3] == length, 'image size error'
                assert target_chip.shape[1] == target_chip.shape[2] == length, 'target size error'

                '''
                imtest = img_chip[0,...].permute(1, 2, 0).cpu()
                imtest = cv2.normalize(np.float32(imtest), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                imtest = imtest.astype(np.uint8)
                plt.subplot(121).imshow(imtest)
                plt.subplot(122).imshow(target_chip.squeeze(0), cmap=CM.jet)
                count = np.sum(target_chip.numpy().squeeze(0))
                plt.title('People count: ' + str(count))
                plt.show()
                '''
                count = np.sum(target_chip.numpy().squeeze(0))
                count = np.round(count)
                if count >= 10:
                    continue
                target = torch.zeros(10)
                target[int(count)] = 1

                img = Variable(img_chip)
                target = Variable(target)
                
                imgs.append(img)
                targets.append(target)
                b_num += 1

                if b_num >= 512:
                    img = torch.stack(imgs, dim=0).squeeze(1)
                    target = torch.stack(targets, dim=0)

                    if CUDA:
                        img = img.cuda()
                        target = target.cuda()

                    '''
                    if epoch <= 1:
                        tb_writer.add_graph(torch.jit.trace(model, img, strict=False), [])
                    '''
                    capsule_output, reconstruction, _ = model(img, target)
                    predictions = torch.norm(capsule_output.squeeze(), dim=2)
                    loss, rec_loss, marg_loss = model.loss(img, target, capsule_output, reconstruction, alpha)
                    
                    losses.update(loss.item(), img.size(0))

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()    
                    
                    batch_time.update(time.time() - end)
                    end = time.time()
                    
                    if bi % args.print_freq == 0:
                        print('Epoch: [{0}][{1}/{2}]\t'
                            'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                            'Epoch Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                            .format(epoch, bi, len(train_loader), batch_time=batch_time, data_time=data_time, loss=losses))
                    
                        #tb_writer.add_scalar('train loss/iteration', losses.avg, epoch * len(train_loader.dataset) + bi)

                    imgs = []
                    targets = []
                    b_num = 0

    tb_writer.add_scalar('train loss/epoch', losses.avg, epoch)

    pred = validate(val_list, model, alpha, CUDA)
        
    is_best = pred < best_pred
    best_pred = min(pred, best_pred)
    print(' * best MAE {mae:.3f} '.format(mae=best_pred))
    save_checkpoint({
        'epoch': epoch + 1,
        'arch': args.pre,
        'state_dict': model.state_dict(),
        'best_pred': best_pred,
        'optimizer' : optimizer.state_dict(),}, is_best, args.checkpoint_path)


def get_alpha(epoch):
    # WARNING: Does not support alpha value saving when continuning training from a saved model
    DEFAULT_ANNEAL_TEMPERATURE = 8
    if args.anneal_alpha == "none":
        alpha = args.alpha
    if args.anneal_alpha == "1":
        alpha = args.alpha * float(np.tanh(epoch/DEFAULT_ANNEAL_TEMPERATURE - np.pi) + 1) / 2
    if args.anneal_alpha == "2":
        alpha = args.alpha * float(np.tanh(epoch/(2 * DEFAULT_ANNEAL_TEMPERATURE)))
    return alpha


def validate(val_list, model, alpha, CUDA):
    print ('begin test')
    test_loader = torch.utils.data.DataLoader(dataset.listDataset(val_list, shuffle=False, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),]), train=False, img_size=args.input_size), batch_size=args.batch_size)    
    
    model.eval()
    
    mae = 0
    length = 32
    imgs, targets = [], []
    b_num = 0
    for bi, (img_big, target_big) in enumerate(test_loader):
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
                target_chip = target_big[:, y1:y2, x1:x2]
                target_chip = zeropad(target_chip.squeeze(0).numpy(), length - img_chip.shape[2], length - img_chip.shape[3], target=True)
                target_chip = torch.from_numpy(target_chip).unsqueeze(0)
                assert img_chip.shape[2] == img_chip.shape[3] == length, 'image size error'
                assert target_chip.shape[1] == target_chip.shape[2] == length, 'target size error'

                count = np.sum(target_chip.numpy().squeeze(0))
                count = np.round(count)
                if count >= 10:
                    continue
                target = torch.zeros(10)
                target[int(count)] = 1

                img = Variable(img_chip)
                target = Variable(target)
                
                imgs.append(img)
                targets.append(target)
                b_num += 1

                if i == (ni-1) and j == (nj-1):
                    img = torch.stack(imgs, dim=0).squeeze(1)
                    target = torch.stack(targets, dim=0)

                    if CUDA:
                        img = img.cuda()
                        target = target.cuda()

                    with torch.no_grad():
                        capsule_output, reconstruction, predictions = model(img, target)
                        #loss, rec_loss, marg_loss = model.loss(img, target, capsule_output, reconstruction, alpha)
                        
                        predictions = np.argmax(predictions.cpu(), axis=1) 
                        target = np.argmax(target.cpu(), axis=1) 
                        mae += abs(predictions.data.sum()-target.sum().type(torch.FloatTensor).cuda())
                        
                        imgs = []
                        targets = []
                        b_num = 0
        
    mae = mae/len(test_loader)    
    print(' * MAE {mae:.3f} '.format(mae=mae))

    return mae    
        
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    
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