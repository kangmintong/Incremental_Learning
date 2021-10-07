import sys
sys.path.append('..')
import numpy as np
import torch
from matplotlib import pyplot as plt

def save_pytorch_tensor_to_jpg(tensor,path):
    img=tensor.data
    img=img-img.min()
    img=img/img.max()
    img*=255
    img=img.cpu()
    img=img.squeeze()
    npimg=img.permute(1,2,0).numpy().astype('uint8')
    plt.imsave(path,npimg)

def save_cam(tensor,path):
    img = tensor.data
    img = img - img.min()
    img = img / img.max()
    img *= 255
    img = img.cpu()
    img = img.squeeze()
    npimg=img.numpy().astype('uint8')
    plt.imsave(path, npimg)