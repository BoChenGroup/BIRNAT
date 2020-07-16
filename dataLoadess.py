
from torch.utils.data import Dataset
import os
import torch
import scipy.io as scio


class Imgdataset(Dataset):

    def __init__(self, path):
        super(Imgdataset, self).__init__()
        self.data = []
        if os.path.exists(path):
            groung_truth_path = path + '/gt'
            measurement_path = path + '/measurement'

            if os.path.exists(groung_truth_path) and os.path.exists(measurement_path):
                groung_truth = os.listdir(groung_truth_path)
                measurement = os.listdir(measurement_path)
                self.data = [{'groung_truth': groung_truth_path + '/' + groung_truth[i],
                              'measurement': measurement_path + '/' + measurement[i]} for i in range(len(groung_truth))]
            else:
                raise FileNotFoundError('path doesnt exist!')
        else:
            raise FileNotFoundError('path doesnt exist!')

    def __getitem__(self, index):

        groung_truth, measurement = self.data[index]["groung_truth"], self.data[index]["measurement"]

        gt = scio.loadmat(groung_truth)
        meas = scio.loadmat(measurement)
        if "patch_save" in gt:
            gt = torch.from_numpy(gt['patch_save'] / 255)
        elif "p1" in gt:
            gt = torch.from_numpy(gt['p1'] / 255)
        elif "p2" in gt:
            gt = torch.from_numpy(gt['p2'] / 255)
        elif "p3" in gt:
            gt = torch.from_numpy(gt['p3'] / 255)

        meas = torch.from_numpy(meas['meas'] / 255)

        gt = gt.permute(2, 0, 1)

        return gt, meas

    def __len__(self):

        return len(self.data)
