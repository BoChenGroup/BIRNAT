
import torch
import scipy.io as scio
import numpy as np


def generate_masks(mask_path):
    mask = scio.loadmat(mask_path + '/mask.mat')
    mask = mask['mask']
    mask = np.transpose(mask, [2, 0, 1])
    mask_s = np.sum(mask, axis=0)
    index = np.where(mask_s == 0)
    mask_s[index] = 1
    mask_s = mask_s.astype(np.uint8)
    mask = torch.from_numpy(mask)
    mask = mask.float()
    mask = mask.cuda()
    mask_s = torch.from_numpy(mask_s)
    mask_s = mask_s.float()
    mask_s = mask_s.cuda()
    return mask, mask_s


def time2file_name(time):
    year = time[0:4]
    month = time[5:7]
    day = time[8:10]
    hour = time[11:13]
    minute = time[14:16]
    second = time[17:19]
    time_filename = year + '_' + month + '_' + day + '_' + hour + '_' + minute + '_' + second
    return time_filename
