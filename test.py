from models import forward_rnn, cnn1, backrnn
from utils import generate_masks, time2file_name
import torch.nn as nn
import torch
import scipy.io as scio
import datetime
import os
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if not torch.cuda.is_available():
    raise Exception('NO GPU!')

data_path = "./train"
test_path1 = "./test"  # simulation data for comparison

mask, mask_s = generate_masks(data_path)
last_train = 100
model_save_filename = 'save_model'

block_size = 256
compress_rate = 8

first_frame_net = cnn1(compress_rate + 1).cuda()
rnn1 = forward_rnn().cuda()
rnn2 = backrnn().cuda()

# load pretrained model
if last_train != 0:
    first_frame_net = torch.load(
        './model/' + model_save_filename + "/first_frame_net_model_epoch_{}.pth".format(last_train))
    rnn1 = torch.load('./model/' + model_save_filename + "/rnn1_model_epoch_{}.pth".format(last_train))
    rnn2 = torch.load('./model/' + model_save_filename + "/rnn2_model_epoch_{}.pth".format(last_train))

loss = nn.MSELoss()
loss.cuda()


def test(test_path, epoch, result_path):
    test_list = os.listdir(test_path)
    psnr_forward = torch.zeros(len(test_list))
    psnr_backward = torch.zeros(len(test_list))
    for i in range(len(test_list)):
        pic = scio.loadmat(test_path + '/' + test_list[i])

        pic = pic['orig']
        pic = pic / 255

        pic_gt = np.zeros([pic.shape[2] // compress_rate, compress_rate, block_size, block_size])
        for jj in range(pic.shape[2]):
            if jj % compress_rate == 0:
                meas_t = np.zeros([block_size, block_size])
                n = 0
            pic_t = pic[:, :, jj]
            mask_t = mask[n, :, :]

            mask_t = mask_t.cpu()
            pic_gt[jj // compress_rate, n, :, :] = pic_t
            n += 1
            meas_t = meas_t + np.multiply(mask_t.numpy(), pic_t)

            if jj == compress_rate - 1:
                meas_t = np.expand_dims(meas_t, 0)
                meas = meas_t
            elif (jj + 1) % compress_rate == 0 and jj != compress_rate - 1:
                meas_t = np.expand_dims(meas_t, 0)
                meas = np.concatenate((meas, meas_t), axis=0)
        meas = torch.from_numpy(meas)
        pic_gt = torch.from_numpy(pic_gt)
        meas = meas.cuda()
        pic_gt = pic_gt.cuda()
        meas = meas.float()
        pic_gt = pic_gt.float()

        meas_re = torch.div(meas, mask_s)
        meas_re = torch.unsqueeze(meas_re, 1)

        out_save1 = torch.zeros([meas.shape[0], compress_rate, block_size, block_size]).cuda()
        out_save2 = torch.zeros([meas.shape[0], compress_rate, block_size, block_size]).cuda()
        with torch.no_grad():
            psnr_1 = 0
            psnr_2 = 0
            for ii in range(meas.shape[0]):
                h0 = torch.zeros(1, 20, block_size, block_size).cuda()
                xt1 = first_frame_net(mask, torch.unsqueeze(meas_re[ii, :, :, :], dim=0), block_size, compress_rate)
                out_pic1, h1 = rnn1(xt1, torch.unsqueeze(meas[ii, :, :], dim=0), mask, h0,
                                    torch.unsqueeze(meas_re[ii, :, :, :], dim=0), block_size, compress_rate)
                out_pic2 = rnn2(out_pic1, torch.unsqueeze(meas[ii, :, :], dim=0), mask, h1,
                                torch.unsqueeze(meas_re[ii, :, :, :], dim=0), block_size, compress_rate)

                out_save1[ii, :, :, :] = out_pic1[0, :, :, :]
                out_save2[ii, :, :, :] = out_pic2[0, :, :, :]

                for jj in range(compress_rate):
                    out_pic_forward = out_pic1[0, jj, :, :]
                    out_pic_backward = out_pic2[0, jj, :, :]
                    gt_t = pic_gt[ii, jj, :, :]
                    mse_forward = loss(out_pic_forward * 255, gt_t * 255)
                    mse_forward = mse_forward.data
                    mse_backward = loss(out_pic_backward * 255, gt_t * 255)
                    mse_backward = mse_backward.data
                    psnr_1 += 10 * torch.log10(255 * 255 / mse_forward)
                    psnr_2 += 10 * torch.log10(255 * 255 / mse_backward)

            psnr_1 = psnr_1 / (meas.shape[0] * compress_rate)
            psnr_2 = psnr_2 / (meas.shape[0] * compress_rate)
            psnr_forward[i] = psnr_1
            psnr_backward[i] = psnr_2
            a = test_list[i]
            # name1 = result_path + '/forward_' + a[0:len(a) - 4] + '{}_{:.4f}'.format(epoch, psnr_1) + '.mat'
            name2 = result_path + '/backward_' + a[0:len(a) - 4] + '{}_{:.4f}'.format(epoch, psnr_2) + '.mat'
            # scio.savemat(name1, {'pic': out_save1.cpu().numpy()})
            scio.savemat(name2, {'pic': out_save2.cpu().numpy()})
    print("only forward rnn result: {:.2f}".format(torch.mean(psnr_forward)),
          "     backward rnn result: {:.2f}".format(torch.mean(psnr_backward)))


if __name__ == '__main__':
    date_time = str(datetime.datetime.now())
    date_time = time2file_name(date_time)
    result_path = 'recon' + '/' + date_time
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    test(test_path1, last_train, result_path)
