#Import packages
import os
import argparse
import time
import json
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch.autograd import Variable
import pathlib
from model import ComputeLoss, MSPSNet
from dataset import listDataset
from tqdm import tqdm
from utils import save_checkpoint, AverageMeter, vis_input
import cv2

#Arguments parser
path = pathlib.Path(__file__).parent.absolute()
parser = argparse.ArgumentParser(description='RCVLab-AiimLab Crowd counting')
# GENERAL
parser.add_argument('--model_desc', default='ShanghaiA_multi/', help="Set model description")
parser.add_argument('--pre_model_desc', default='shanghaiA_pre/', help="Set model description")
parser.add_argument('--train_json', default=path/'datasets/shanghai/part_A_train.json', help='path to train json')
parser.add_argument('--use_pre', default=False, type=bool, help='use the pretrained model?')
parser.add_argument('--use_gpu', default=True, action="store_false", help="Indicates whether or not to use GPU")
parser.add_argument('--device', default='0', type=str, help='GPU id to use.')
parser.add_argument('--checkpoint_path', default=path/'runs/weights', type=str, help='checkpoint path')
parser.add_argument('--log_dir', default=path/'runs/log', type=str, help='log dir')
parser.add_argument('--exp', default='shanghai', type=str, help='shanghai, sim, or ucf_qnrf, set dataset for training experiment')
parser.add_argument('--density', default=False, type=bool, help='using density map instead of head locations?')
parser.add_argument('--augment', default=True, type=bool, help='augmentation?')

# MODEL
parser.add_argument('--backend', default='vgg', type=str, help='vgg or resnet')

# TRAINING
parser.add_argument('--batch_size', default=512, type=int)
parser.add_argument('--epochs', default=1000, type=int, help="Number of epochs to train for")
parser.add_argument('--workers', default=8, type=int, help="Number of workers in loading dataset")
parser.add_argument('--start_epoch', default=0, type=int, help="start_epoch")
parser.add_argument('--vis', default=False, type=bool, help='visualize the inputs') 
parser.add_argument('--lr0', default=0.000001, type=float, help="initial learning rate")
parser.add_argument('--weight_decay', default=0.0005, type=float, help="weight_decay")
parser.add_argument('--momentum', default=0.937, type=float, help="momentum")
parser.add_argument('--adam', default=True, type=bool, help='use torch.optim.Adam() optimizer') 


#Training function
def train(args, model, optimizer, train_list, tb_writer, CUDA):

    compute_loss = ComputeLoss(model)
    
    for epoch in range(args.start_epoch, args.epochs):

        train_loader = torch.utils.data.DataLoader(listDataset(train_list,
                       density=args.density,
                       augment=args.augment,
                       transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),]), 
                       train=True, 
                       batch_size=1,
                       num_workers=args.workers,
                       exp=args.exp))

        losses = AverageMeter()
        losses_count_0 = AverageMeter()
        losses_count_1 = AverageMeter()
        losses_count_2 = AverageMeter()
        losses_count_3 = AverageMeter()
        data_time = AverageMeter()
        batch_time = AverageMeter()
        end = time.time()

        model.train()

        pbar = enumerate(train_loader)
        pbar = tqdm(pbar, total=len(train_loader))  # progress bar

        optimizer.zero_grad()

        imgs, targets = [], []
        
        for bi, (img_big, target_big) in pbar:  # batch ----------
            data_time.update(time.time() - end)

            length_0 = img_big.size(2)
            length_1 = img_big.size(3)
            
            if args.vis:
                vis_input(img_big.squeeze(0), target_big.squeeze(0))

            if not args.density:
                coord = (target_big.squeeze(0)).nonzero(as_tuple=False)
                bxy = [[yb/length_0, xb/length_1] for (yb, xb) in coord]
                targets.append(torch.tensor(bxy))

            img = torch.clone(img_big)

            img = Variable(img)
            imgs.append(img)

            if args.density:
                target_big = target_big.squeeze(0).numpy()
                target_big = cv2.resize(target_big, (target_big.shape[1]//8,target_big.shape[0]//8),interpolation = cv2.INTER_CUBIC)*64
                targets.append(torch.from_numpy(target_big))

            imgs = torch.stack(imgs, dim=0).squeeze(1)
            if not args.density:
                targets = [ti for ti in targets if len(ti) != 0]
                if not targets:
                    targets.append(torch.tensor([[0, 0]]))
                targets = torch.cat(targets)
            else:
                targets = torch.stack(targets, dim=0)

            if CUDA:
                imgs = imgs.cuda()
                targets = targets.cuda()

            #Loss calculation
            pred0, pred1, pred2, pred3 = model(imgs, training=True)  # forward
            train_loss, lcount_0, lcount_1, lcount_2, lcount_3 = compute_loss(pred0, pred1, pred2, pred3, targets) 
            pred3 = pred3[0]

            loss = sum(train_loss[i] for i in range(len(train_loss)))
            
            losses.update(loss.item(), imgs.size(0))
            losses_count_0.update(lcount_0.item(), imgs.size(0))
            losses_count_1.update(lcount_1.item(), imgs.size(0))
            losses_count_2.update(lcount_2.item(), imgs.size(0))
            losses_count_3.update(lcount_3.item(), imgs.size(0))
            
            # Backward
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

            # end batch ------------

        
        save_checkpoint({
            'epoch': epoch,
            'arch': args.checkpoint_path,
            'state_dict': model.state_dict(),
            'best_pred': args.best_mae,
            'optimizer' : optimizer.state_dict(),}, args.checkpoint_path)
        
        # Log to TensorBoard
        tb_writer.add_scalar('train loss/total', losses.avg, epoch)
        tb_writer.add_scalar('train loss/count_0', losses_count_0.avg, epoch)
        tb_writer.add_scalar('train loss/count_1', losses_count_1.avg, epoch)
        tb_writer.add_scalar('train loss/count_2', losses_count_2.avg, epoch)
        tb_writer.add_scalar('train loss/count_3', losses_count_3.avg, epoch)

        # end epoch ------------



def main():
    torch.manual_seed(0)
    args = parser.parse_args()

    args.best_mae = 1e6

    args.log_dir = args.log_dir / args.model_desc
    tb_writer = SummaryWriter(args.log_dir)

    args.pre_checkpoint_path = args.checkpoint_path / args.pre_model_desc
    args.pre_checkpoint_path = args.pre_checkpoint_path / 'checkpoint.pth.tar'
    
    args.checkpoint_path = args.checkpoint_path / args.model_desc
    if not pathlib.Path(args.checkpoint_path).exists():
        os.mkdir(args.checkpoint_path)
    args.checkpoint_path = args.checkpoint_path / 'checkpoint.pth.tar'

    with open(args.train_json, 'r') as outfile:        
        train_list_main = json.load(outfile)
    #This part of the code is borrowed from https://github.com/leeyeehoo/CSRNet-pytorch
    train_list = [st.replace('/home/leeyh/Downloads/Shanghai', 'C:/Users/mahdi/Desktop/SASNet_ROOT/ShanghaiTech') for st in train_list_main]

    if args.use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device
        CUDA =True
    else:
        CUDA = False

    model = MSPSNet(backend=args.backend)

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
    
    train(args, model, optimizer, train_list, tb_writer, CUDA) 


if __name__ == '__main__':
    main() 