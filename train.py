from dataLoadess import Imgdataset
from torch.utils.data import DataLoader
from models import forward_rnn, cnn1, backrnn
from utils import generate_masks, time2file_name
from gan_resnet import Discriminator
import torch.optim as optim
import torch.nn as nn
import torch
import scipy.io as scio
import time
import datetime
import os
import numpy as np
from torch.autograd import Variable
from torch import autograd
from torch.nn import functional as F

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if not torch.cuda.is_available():
    raise Exception('NO GPU!')

data_path = "./train"
test_path1 = "./test"  # simulation data for comparison

mask, mask_s = generate_masks(data_path)

last_train = 0
model_save_filename = ''
max_iter = 100
batch_size = 3
learning_rate = 0.0003

block_size = 256
compress_rate = 8

dataset = Imgdataset(data_path)

train_data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

first_frame_net = cnn1(compress_rate + 1).cuda()
rnn1 = forward_rnn().cuda()
rnn2 = backrnn().cuda()
D = Discriminator(1, 256, 64, 1024)
D.cuda()

if last_train != 0:
    first_frame_net = torch.load(
        './model/' + model_save_filename + "/first_frame_net_model_epoch_{}.pth".format(last_train))
    rnn1 = torch.load('./model/' + model_save_filename + "/rnn1_model_epoch_{}.pth".format(last_train))
    rnn2 = torch.load('./model/' + model_save_filename + "/rnn2_model_epoch_{}.pth".format(last_train))

loss = nn.MSELoss()
loss.cuda()
BCE_loss = nn.BCELoss().cuda()


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
            if epoch % 10 == 0:
                a = test_list[i]
                # name1 = result_path + '/forward_' + a[0:len(a) - 4] + '{}_{:.4f}'.format(epoch, psnr_1) + '.mat'
                name2 = result_path + '/backward_' + a[0:len(a) - 4] + '{}_{:.4f}'.format(epoch, psnr_2) + '.mat'
                # scio.savemat(name1, {'pic': out_save1.cpu().numpy()})
                scio.savemat(name2, {'pic': out_save2.cpu().numpy()})
    print("only forward rnn result: {:.2f}".format(torch.mean(psnr_forward)),
          "     backward rnn result: {:.2f}".format(torch.mean(psnr_backward)))


def train(epoch, learning_rate, result_path):
    epoch_loss = 0
    epoch_loss1 = 0
    epoch_loss2 = 0
    Dloss = 0
    regloss = 0
    begin = time.time()

    optimizer_g = optim.Adam([{'params': first_frame_net.parameters()}, {'params': rnn1.parameters()},
                              {'params': rnn2.parameters()}], lr=learning_rate)
    optimizer_d = optim.Adam(D.parameters(), lr=learning_rate)
    if __name__ == '__main__':
        for iteration, batch in enumerate(train_data_loader):
            gt, meas = Variable(batch[0]), Variable(batch[1])
            gt = gt.cuda()  # [batch,8,256,256]
            gt = gt.float()
            meas = meas.cuda()  # [batch,256 256]
            meas = meas.float()

            mini_batch = gt.size()[0]
            y_real_ = torch.ones(mini_batch).cuda()
            y_fake_ = torch.zeros(mini_batch).cuda()

            meas_re = torch.div(meas, mask_s)
            meas_re = torch.unsqueeze(meas_re, 1)

            optimizer_d.zero_grad()

            batch_size1 = gt.shape[0]

            h0 = torch.zeros(batch_size1, 20, 256, 256).cuda()

            xt1 = first_frame_net(mask, meas_re, block_size, compress_rate)
            model_out1, h1 = rnn1(xt1, meas, mask, h0, meas_re, block_size, compress_rate)
            model_out = rnn2(model_out1, meas, mask, h1, meas_re, block_size, compress_rate)

            # discriminator training
            toggle_grad(first_frame_net, False)
            toggle_grad(rnn1, False)
            toggle_grad(rnn2, False)
            toggle_grad(D, True)
            gt.requires_grad_()

            D_result = D(gt, y_real_)
            # assert (D_result > 0.0 & D_result < 1.0).all()
            D_real_loss = compute_loss(D_result, 1)
            Dloss += D_result.data.mean()
            D_real_loss.backward(retain_graph=True)

            # model_out.requires_grad_()
            # d_fake = D(model_out, y_real_)
            # dloss_fake = compute_loss(d_fake, 0)

            batch_size = gt.size(0)
            grad_dout = autograd.grad(
                outputs=D_result.sum(), inputs=gt,
                create_graph=True, retain_graph=True, only_inputs=True
            )[0]
            grad_dout2 = grad_dout.pow(2)
            assert (grad_dout2.size() == gt.size())
            reg1 = grad_dout2.view(batch_size, -1).sum(1)

            reg = 10 * reg1.mean()

            regloss += reg.data.mean()

            reg.backward(retain_graph=True)

            optimizer_d.step()

            # generator training
            toggle_grad(first_frame_net, True)
            toggle_grad(rnn1, True)
            toggle_grad(rnn2, True)
            toggle_grad(D, False)
            optimizer_g.zero_grad()

            D_result = D(model_out, y_real_)
            G_train_loss = compute_loss(D_result, 1)
            Loss1 = loss(model_out1, gt)
            Loss2 = loss(model_out, gt)
            Loss = 0.5 * Loss1 + 0.5 * Loss2 + 0.001 * G_train_loss

            epoch_loss += Loss.data
            epoch_loss1 += Loss1.data
            epoch_loss2 += Loss2.data

            Loss.backward()
            optimizer_g.step()

        test(test_path1, epoch, result_path)

    end = time.time()
    print("===> Epoch {} Complete: Avg. Loss: {:.7f}".format(epoch, epoch_loss / len(train_data_loader)),
          "loss1 {:.7f} loss2: {:.7f}".format(epoch_loss1 / len(train_data_loader),
                                              epoch_loss2 / len(train_data_loader)),
          "d loss: {:.7f},reg loss: {:.7f}".format(Dloss / len(train_data_loader),
                                                   regloss / len(train_data_loader)),
          "  time: {:.2f}".format(end - begin))


def compute_loss(d_out, target):
    targets = d_out.new_full(size=d_out.size(), fill_value=target)
    loss = F.binary_cross_entropy_with_logits(d_out, targets)

    return loss


def checkpoint(epoch, model_path):
    model_out_path = './' + model_path + '/' + "first_frame_net_model_epoch_{}.pth".format(epoch)
    torch.save(first_frame_net, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)


def checkpoint2(epoch, model_path):
    model_out_path = './' + model_path + '/' + "rnn1_model_epoch_{}.pth".format(epoch)
    torch.save(rnn1, model_out_path)
    # print("Checkpoint saved to {}".format(model_out_path))


def checkpoint3(epoch, model_path):
    model_out_path = './' + model_path + '/' + "rnn2_model_epoch_{}.pth".format(epoch)
    torch.save(rnn2, model_out_path)
    # print("Checkpoint saved to {}".format(model_out_path))


def checkpointD(epoch, model_path):
    model_out_path = './' + model_path + '/' + "discriminator_model_epoch_{}.pth".format(epoch)
    torch.save(D, model_out_path)
    # print("Checkpoint saved to {}".format(model_out_path))


def main(learning_rate):
    date_time = str(datetime.datetime.now())
    date_time = time2file_name(date_time)
    result_path = 'recon' + '/' + date_time
    model_path = 'model' + '/' + date_time
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    for epoch in range(last_train + 1, last_train + max_iter + 1):
        train(epoch, learning_rate, result_path)
        if (epoch % 10 == 0 or epoch > 70):
            checkpoint(epoch, model_path)
            checkpoint2(epoch, model_path)
            checkpoint3(epoch, model_path)
            checkpointD(epoch, model_path)
        if (epoch % 5 == 0) and (epoch < 150):
            learning_rate = learning_rate * 0.95
            print(learning_rate)


if __name__ == '__main__':
    main(learning_rate)
