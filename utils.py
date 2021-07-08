
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
        best_model_path = filename.replace('checkpoint.pth.tar', 'model_best.pth.tar')
        shutil.copyfile(filename, best_model_path) 


def zeropad(img, h, w, target=False):
    if not target:
        color = [0, 0, 0]
        padded = cv2.copyMakeBorder(img, 0, h, 0, w, cv2.BORDER_CONSTANT, value=color)
    else:
        padded = cv2.copyMakeBorder(img, 0, h, 0, w, cv2.BORDER_CONSTANT, value=0)
    return padded

def vis_input(img, target):
    im = img[0,...].permute(1, 2, 0).cpu()
    im = cv2.normalize(np.float32(im), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    im = im.astype(np.uint8)
    plt.subplot(121).imshow(im)
    plt.subplot(122).imshow(target.squeeze(0), cmap=CM.jet)
    count = np.sum(target.numpy().squeeze(0))
    plt.title('People count: ' + str(count))
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