
import numpy as np 
import shutil
import h5py
import torch
import cv2 
import matplotlib.pyplot as plt
from matplotlib import cm as CM


def save_checkpoint(state, is_best, filename='../checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        best_model_path = filename.parent / 'model_best.pth.tar'
        shutil.copyfile(filename, best_model_path) 


def zeropad(img, h, w, target=False):
    if not target:
        color = [0, 0, 0]
        padded = cv2.copyMakeBorder(img, 0, h, 0, w, cv2.BORDER_CONSTANT, value=color)
    else:
        padded = cv2.copyMakeBorder(img, 0, h, 0, w, cv2.BORDER_CONSTANT, value=0)
    return padded

def vis_input(im, target, predicted=None, thresholded=None):
    if im.size(0) == 4:
        depth = im[3, :, :].cpu().numpy()
        im = im[:3, :, :]
        plt.subplot(1,1,1).imshow(depth, cmap=CM.jet)
        plt.title('depth')
        plt.show()
    if (predicted is not None) and (thresholded is not None):
        i = 2
    elif (predicted is not None) or (thresholded is not None):
        i = 2
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
    target = target.squeeze(0) if target.size(0)==1 else target
    plt.subplot(2,i,2).imshow(target, cmap=CM.jet)
    count = target.sum()
    plt.title('People count: ' + str(count.item()))

    if predicted is not None:
        predicted = predicted.cpu().numpy()
        plt.subplot(2,i,3).imshow(predicted, cmap=CM.jet)
        #count = 1/predicted.mean()
        count = predicted.sum()
        plt.title('Probability map: ' + str(count.item()))
    
    if thresholded is not None:
        thresholded = thresholded.cpu().numpy()
        plt.subplot(2,i,4).imshow(thresholded, cmap=CM.jet)
        count = thresholded.sum()
        plt.title('Pred count (thresholded): ' + str(count.item()))

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
