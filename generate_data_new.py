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
from model import Model, ComputeLoss, CSRNet
from dataset import listDataset
from tqdm import tqdm
import math
from utils import save_checkpoint, AverageMeter, vis_input, zeropad
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import h5py
import glob

path = pathlib.Path(__file__).parent.absolute()
parser = argparse.ArgumentParser(description='RCVLab-AiimLab Crowd counting')

# GENERAL
# parser.add_argument('--model_desc', default='shanghaiA_cell64_lr10-4_ep50/', help="Set model description")
parser.add_argument('--model_desc', default='test/', help="Set model description")
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
parser.add_argument('--cell_size', default=[128, 128, 64], type=int, help="cell size")
parser.add_argument('--threshold', default=0.01, type=int, help="threshold for the classification output")

# TRAINING
parser.add_argument('--batch_size', default=512, type=int)
parser.add_argument('--epochs', default=50, type=int, help="Number of epochs to train for")
parser.add_argument('--workers', default=4, type=int, help="Number of workers in loading dataset")
parser.add_argument('--start_epoch', default=0, type=int, help="start_epoch")
parser.add_argument('--vis', default=False, type=bool, help='visualize the inputs') 
parser.add_argument('--lr0', default=0.0001, type=float, help="initial learning rate")
parser.add_argument('--weight_decay', default=0.0005, type=float, help="weight_decay")
parser.add_argument('--momentum', default=0.937, type=float, help="momentum")
parser.add_argument('--adam', default=False, type=bool, help='use torch.optim.Adam() optimizer') 


def train(args, model, optimizer, train_list, val_list, train_list_depth, val_list_depth, tb_writer, CUDA):

    

    for epoch in range(args.start_epoch, 1):
        train_loader = torch.utils.data.DataLoader(listDataset(train_list,
                       train_list_depth,
                       shuffle=True,
                       depth=args.depth,
                    #    transform1=transforms.Compose([transforms.ToTensor(),]), 
                    #    transform2=transforms.Compose([transforms.ToTensor(),]), 
                       transform1=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),]), 
                       transform2=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.502], std=[0.291]),]), 
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
        
        for bi, (img_big, target_big, img_big_depth) in pbar:  # batch ----------
            imgs, targets = [], []
            data_time.update(time.time() - end)

            # ni = int(math.ceil(img_big.shape[2] / length)) 
            # nj = int(math.ceil(img_big.shape[3] / length))  
            k = 0 
            ny = 0
            update_len = img_big.shape[2] 
            
            while update_len - length[k] >= 0:
                ny = ny + 1
                update_len = update_len - length[k]
                if k < len(length)-1:
                    k = k+1
            ny = ny + 1
                # update_len = update_len - length[k]

            high_end = img_big.shape[2]
            for j in range(ny):  
                if j > len(length) - 1:
                    l = length[-1]
                else:
                    l = length[j]
                nx = int(math.ceil(img_big.shape[3] / l))  
                # print('row number:',j)
                
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
                    img_chip = zeropad(img_chip.squeeze(0).permute(1,2,0).numpy(), l - img_chip.shape[2], l - img_chip.shape[3])
                    img_chip = torch.from_numpy(img_chip).permute(2,0,1).unsqueeze(0)
                    size = img_chip.shape[-1]
                    orig_path = "crowd_csr_grid/data_chopped/part_A_train"
                    if bi==0:
                        for s in [128,64]: 
                            spcfc_path = os.path.join(orig_path,str(s))
                            if not os.path.exists(spcfc_path):
                                os.makedirs(spcfc_path)
                            files = glob.glob(os.path.normpath(os.path.join(spcfc_path,'*')))
                            for f in files:
                                os.remove(f)

                    pathh = os.path.join("crowd_csr_grid/data_chopped/part_A_train",str(size),str(bi)+"_"+str(j)+"_"+str(i)+".h5")
                    hf = h5py.File(pathh, 'w')
                    hf.create_dataset('image', data=img_chip)
                    
                    if args.depth:
                        img_chip_depth = img_big_depth[:, :, y1:y2, x1:x2]
                        img_chip_depth  = zeropad(img_chip_depth.squeeze(0).permute(1,2,0).numpy(), l - img_chip_depth.shape[2], l - img_chip_depth.shape[3])
                        img_chip_depth = torch.from_numpy(img_chip_depth).unsqueeze(2)
                        img_chip_depth = img_chip_depth.permute(2,0,1).unsqueeze(0)
                        assert img_chip_depth.shape[2] == img_chip_depth.shape[3] == l, 'image size error'
                    
                    target_chip = target_big[:, y1:y2, x1:x2]
                    target_chip = zeropad(target_chip.squeeze(0).numpy(), l - target_chip.shape[1], l - target_chip.shape[2], target=True)
                    target_chip = torch.from_numpy(target_chip).unsqueeze(0)

                    hf.create_dataset('target', data=target_chip)

                    assert img_chip.shape[2] == img_chip.shape[3] == l, 'image size error'
                    assert target_chip.shape[1] == target_chip.shape[2] == l, 'target size error'

                    coord = (target_chip.squeeze(0)).nonzero(as_tuple=False)
                    count = target_chip.squeeze(0).sum()
                    # bxy = [[yb/l, xb/l, i, j] for (yb, xb) in coord]
                    b_num += 1
                    hf.create_dataset('coord', data=coord)
                    hf.create_dataset('count', data=count)
                    # print(coord)
                    # print(count)

                    hf.close()
                    # hf.create_dataset('l', data=l)
                high_end = y1
        validate(args, val_list, val_list_depth)



def validate(args, val_list, val_list_depth):
    print ('begin validation')
    val_loader = torch.utils.data.DataLoader(listDataset(val_list, 
                    val_list_depth,
                    shuffle=False, 
                    depth=args.depth,
                    transform1=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),]), 
                    transform2=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.502], std=[0.291]),]), 
                    train=False), 
                    batch_size=1)    
    

    pbar = enumerate(val_loader)
    pbar = tqdm(pbar, total=len(val_loader))  # progress bar
    

    length = args.cell_size
    imgs, targets = [], []
    b_num = 0
    print('balh')
    with open(os.path.join(args.log_dir,'results.txt'), 'w') as f:
        for bi, (img_big, target_big, img_big_depth) in pbar:  # batch ----------
            imgs, targets = [], []

            # ni = int(math.ceil(img_big.shape[2] / length)) 
            # nj = int(math.ceil(img_big.shape[3] / length))  
            k = 0 
            ny = 0
            update_len = img_big.shape[2] 
            b_num = 0
            while update_len - length[k] >= 0:
                ny = ny + 1
                update_len = update_len - length[k]
                if k < len(length)-1:
                    k = k+1
            ny = ny + 1
                # update_len = update_len - length[k]
            high_end = img_big.shape[2]
            for j in range(ny):  
                if j > len(length) - 1:
                    l = length[-1]
                else:
                    l = length[j]
                nx = int(math.ceil(img_big.shape[3] / l))  
                # print('row number:',j)
                
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
                    img_chip = zeropad(img_chip.squeeze(0).permute(1,2,0).numpy(), l - img_chip.shape[2], l - img_chip.shape[3])
                    img_chip = torch.from_numpy(img_chip).permute(2,0,1).unsqueeze(0)
                    
                    size = img_chip.shape[-1]
                    orig_path = "crowd_csr_grid/data_chopped/part_A_test"
                    if bi==0:
                        for s in [128,64]: 
                            spcfc_path = os.path.join(orig_path,str(s))
                            if not os.path.exists(spcfc_path):
                                os.makedirs(spcfc_path)
                            files = glob.glob(os.path.normpath(os.path.join(spcfc_path,'*')))
                            for f in files:
                                os.remove(f)

                    pathh = os.path.join("crowd_csr_grid/data_chopped/part_A_test",str(size),str(bi)+"_"+str(j)+"_"+str(i)+".h5")
                    hf = h5py.File(pathh, 'w')
                    hf.create_dataset('image', data=img_chip)
            
                    if args.depth:
                        img_chip_depth = img_big_depth[:, :, y1:y2, x1:x2]
                        img_chip_depth = zeropad(img_chip_depth.squeeze(0).permute(1,2,0).numpy(), l - img_chip_depth.shape[2], l - img_chip_depth.shape[3])
                        img_chip_depth = torch.from_numpy(img_chip_depth).unsqueeze(2)
                        img_chip_depth = img_chip_depth.permute(2,0,1).unsqueeze(0)
                        assert img_chip_depth.shape[2] == img_chip_depth.shape[3] == l, 'image size error'
                    
                    target_chip = target_big[:, y1:y2, x1:x2]
                    target_chip = zeropad(target_chip.squeeze(0).numpy(), l - target_chip.shape[1], l - target_chip.shape[2], target=True)
                    target_chip = torch.from_numpy(target_chip).unsqueeze(0)

                    hf.create_dataset('target', data=target_chip)

                    coord = (target_chip.squeeze(0)).nonzero(as_tuple=False)
                    count = target_chip.squeeze(0).sum()
                    # bxy = [[yb/l, xb/l, i, j] for (yb, xb) in coord]
                    b_num += 1
                    hf.create_dataset('coord', data=coord)
                    hf.create_dataset('count', data=count)
                    # print(coord)
                    # print(count)

                    hf.close()
                high_end = y1


def main():
    args = parser.parse_args()

    args.best_pred = 1e6

    args.log_dir = os.path.join(args.log_dir,args.model_desc)
    tb_writer = SummaryWriter(args.log_dir)

    args.checkpoint_path += ('/'+args.model_desc)
    if not pathlib.Path(args.checkpoint_path).exists():
        os.mkdir(args.checkpoint_path)
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