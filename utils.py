#Import packages
import numpy as np 
import torch
import cv2 
import matplotlib.pyplot as plt
from matplotlib import cm as CM
from scipy.ndimage.filters import gaussian_filter

#Useful functions
def save_checkpoint(state, filename='../checkpoint.pth.tar'):
    torch.save(state, filename)

def zeropad(img, h, w, target=False):
    if not target:
        color = [0, 0, 0]
        padded = cv2.copyMakeBorder(img, 0, h, 0, w, cv2.BORDER_CONSTANT, value=color)
    else:
        padded = cv2.copyMakeBorder(img, 0, h, 0, w, cv2.BORDER_CONSTANT, value=0)
    return padded

def vis_input(im, target=None, pred0=None, pred1=None, pred2=None):
    if im.size(0) == 4:
        depth = im[3, :, :].cpu().numpy()
        im = im[:3, :, :]
        plt.subplot(1,1,1).imshow(depth, cmap=CM.jet)
        plt.title('depth')
        plt.show()
    elif (pred1 is None) or (pred2 is None):
        i = 2
    elif pred2 is not None:
        i = 3
    else:
        i = 1

    if im.size(0) == 3:
        im = im.permute(1, 2, 0).cpu()
        im = cv2.normalize(np.float32(im), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        im = im.astype(np.uint8)
    else:
        if im.size(0) == 1:
            im = im.squeeze(0)
        im = im.cpu().numpy()

    plt.subplot(2,i,1).imshow(im)
    target = target.cpu().numpy()
    target = target.squeeze(0) if target.shape[0]==1 else target
    plt.subplot(2,i,2).imshow(target, cmap=CM.jet)
    count = target.sum()
    plt.title('GT: ' + str(count.item()))

    if pred0 is not None:
        pred0 = pred0.cpu().numpy()
        plt.subplot(2,i,4).imshow(pred0, cmap=CM.jet)
        count = pred0.sum()
        plt.title('Pred: ' + str(count.item()))
    
    if pred1 is not None:
        pred1 = pred1.cpu().numpy()
        plt.subplot(2,i,5).imshow(pred1, cmap=CM.jet)
        count = pred1.sum()
        plt.title('Pred: ' + str(count.item()))

    if pred2 is not None:
        pred2 = pred2.cpu().numpy()
        plt.subplot(2,i,6).imshow(pred2, cmap=CM.jet)
        count = pred2.sum()
        plt.title('Pred: ' + str(count.item()))

    plt.show()

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
