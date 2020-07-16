import torch
import torch.nn as nn
import torch.nn.functional as F


class self_attention(nn.Module):
    def __init__(self, ch):
        super(self_attention, self).__init__()
        self.conv1 = nn.Conv2d(ch, ch // 8, 1)
        self.conv2 = nn.Conv2d(ch, ch // 8, 1)
        self.conv3 = nn.Conv2d(ch, ch, 1)
        self.conv4 = nn.Conv2d(ch, ch, 1)
        self.gamma1 = torch.nn.Parameter(torch.Tensor([0]))
        self.ch = ch

    def forward(self, x):
        batch_size = x.shape[0]

        f = self.conv1(x)
        g = self.conv2(x)
        h = self.conv3(x)
        ht = h.reshape([batch_size, self.ch, -1])

        ft = f.reshape([batch_size, self.ch // 8, -1])
        n = torch.matmul(ft.permute([0, 2, 1]), g.reshape([batch_size, self.ch // 8, -1]))
        beta = F.softmax(n, dim=-1)

        o = torch.matmul(ht, beta)
        o = o.reshape(x.shape)  # [bs, C, h, w]

        o = self.conv4(o)

        x = self.gamma1 * o + x

        return x


class res_part(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(res_part, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x = x1 + x
        x1 = self.conv2(x)
        x = x1 + x
        x1 = self.conv3(x)
        x = x1 + x
        return x


class down_feature(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(down_feature, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 20, 5, stride=1, padding=2),
            nn.Conv2d(20, 20, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(20, 20, 3, stride=1, padding=1),
            nn.Conv2d(20, 40, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(40, out_ch, 3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_feature(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(up_feature, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 40, 3, stride=1, padding=1),
            nn.Conv2d(40, 30, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(30, 20, 3, stride=1, padding=1),
            nn.Conv2d(20, 20, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(20, 20, 3, padding=1),
            nn.Conv2d(20, out_ch, 1),
        )

    def forward(self, x):
        x = self.conv(x)
        return x
