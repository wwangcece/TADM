import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from .module_util import (
    Convkxk,
    Quantization,
    initialize_weights_xavier,
    initialize_weights,
    Quan,
    InvertibleConv1x1,
    Quanti,
)
import math

# var-ae-2.3
class DFRM(nn.Module):
    def __init__(self, in_channels, out_channels, ir_scale=16):
        super(DFRM, self).__init__()
        self.in_nc = in_channels  # 4
        self.out_nc = out_channels # 3
        self.ir_scale = ir_scale

        self.quan = Quan()

        # (1) downsample by cnn
        self.down_module = []
        self.down_module.append(ConvBlock(3, 16, 3, 1, 1))
        curr_pixel_channels = 16
        for _ in range(int(math.log2(ir_scale))):
            self.down_module.append(Downsample(curr_pixel_channels, 2))
            curr_pixel_channels *= 2
            self.down_module.append(Resblock(curr_pixel_channels, 3, 1, 1))
        self.down_module.append(ConvBlock(curr_pixel_channels, 16, 3, 1, 1))
        curr_pixel_channels = 16
        self.down_module = nn.Sequential(*self.down_module)
        self.pixel_channels = curr_pixel_channels

        # (2) cnn network for processing latent code
        self.latent_module = []
        self.latent_module.append(ConvBlock(in_channels, 128, 3, 1, 1))
        for _ in range(8):
            self.latent_module.append(Resblock(128, 3, 1, 1))
        if self.ir_scale == 4:
            self.latent_module.append(nn.PixelShuffle(2))
            self.latent_module.append(ConvBlock(32, 128, 3, 1, 1))
        elif self.ir_scale == 8:
            self.latent_module.append(ConvBlock(128, 128, 3, 1, 1))
        elif self.ir_scale == 16:
            self.latent_module.append(nn.PixelUnshuffle(2))
            self.latent_module.append(ConvBlock(512, 128, 3, 1, 1))
        elif self.ir_scale==32:
            self.latent_module.append(nn.PixelUnshuffle(2))
            self.latent_module.append(ConvBlock(512, 128, 3, 1, 1))
            self.latent_module.append(nn.PixelUnshuffle(2))
            self.latent_module.append(ConvBlock(512, 128, 3, 1, 1))
        else:
            Warning("Not implemented rescaling factor!!!")
        self.latent_module = nn.Sequential(*self.latent_module)
        curr_latent_channels = 128
        self.latent_channels = curr_latent_channels

        # (3) fuse the features from two branches
        curr_channels = curr_latent_channels + curr_pixel_channels # 16+128
        self.fusing_module = []
        for _ in range(8):
            self.fusing_module.append(Resblock(curr_channels, 3, 1, 1))
        self.fusing_module.append(ConvBlock(curr_channels, out_channels, 3, 1, 1))
        self.fusing_module = nn.Sequential(*self.fusing_module)

        # (4) final module to reconstruct latent
        self.final_module = []
        self.final_module.append(ConvBlock(out_channels, 128, 3, 1, 1))
        for _ in range(8):
            self.final_module.append(Resblock(128, 3, 1, 1))
        if self.ir_scale == 4:
            self.final_module.append(nn.PixelUnshuffle(2))
            self.final_module.append(ConvBlock(512, in_channels, 3, 1, 1))
        if self.ir_scale==8:
            self.final_module.append(ConvBlock(128, in_channels, 3, 1, 1))
        elif self.ir_scale==16:
            self.final_module.append(nn.PixelShuffle(2))
            self.final_module.append(ConvBlock(32, in_channels, 3, 1, 1))
        elif self.ir_scale==32:
            self.final_module.append(nn.PixelShuffle(2))
            self.final_module.append(ConvBlock(32, 32, 3, 1, 1))
            self.final_module.append(nn.PixelShuffle(2))
            self.final_module.append(ConvBlock(8, in_channels, 3, 1, 1))
        else:
            Warning("Not implemented rescaling factor!!!")
        self.final_module = nn.Sequential(*self.final_module)

        # (5) invertible module to transfer features to low-res image
        self.inv_module = []
        for _ in range(12):
            self.inv_module.append(InvBlockExp(1, 2))
        self.inv_module = nn.ModuleList(self.inv_module)

    def process(self, x, latent=None, rev=False, train=True):
        out = x
        if not rev:
            assert latent is not None
            pixel_res = self.down_module(out)
            latent_res = self.latent_module(latent)
            out = torch.cat([pixel_res, latent_res], dim=1)
            out = self.fusing_module(out)
            # final branch
            latent_for = self.final_module(out)
            # inv branch
            for op in self.inv_module:
                out = op(out, rev)
            # quantizition
            out = self.quan(out, train=train)
            return out, latent_for
        else:
            for op in reversed(self.inv_module):
                out = op(out, rev)
            out = self.final_module(out)
            latent_rev = out
            return latent_rev

    def forward(self, x, latent, rev=False, train=True):
        out = x
        if not rev:
            assert latent is not None
            pixel_res = self.down_module(out)
            latent_res = self.latent_module(latent)
            out = torch.cat([pixel_res, latent_res], dim=1)
            out = self.fusing_module(out)
            # final branch
            latent_for = self.final_module(out)
            # inv branch
            # out = out.detach()
            for op in self.inv_module:
                out = op(out, rev)
            # quantizition
            out = self.quan(out, train=train)
            return out, latent_for
        else:
            for op in reversed(self.inv_module):
                out = op(out, rev)
            out = self.final_module(out)
            latent_rev = out
            return latent_rev
        
    def inference(self, hr, latent):
        z = latent / 0.18215
        # down
        pixel_res = self.down_module(hr)
        latent_res = self.latent_module(z)
        out = torch.cat([pixel_res, latent_res], dim=1)
        out = self.fusing_module(out)
        for op in self.inv_module:
            out = op(out, False)
        # quantizition
        lr_temp = self.quan(out, train=True)
        # lr_temp = out
        # up
        out = lr_temp
        for op in reversed(self.inv_module):
            out = op(out, True)
        z_back_recon = self.final_module(out)

        result = {
            "hr": hr,
            "z": z,
            "z_back_recon": z_back_recon,
            "lr": lr_temp,
        }
        return result

    def save_model(self, outf):
        sd = {}
        sd["state_dict"] = {k: v for k, v in self.state_dict().items()}
        torch.save(sd, outf)

# CNN modules
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv_in = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.acti = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        out = self.conv_in(x)
        out = self.acti(out)
        return out

class Resblock(nn.Module):
    def __init__(self, n_feat, kernel_size, stride, padding):
        super(Resblock, self).__init__()
        self.res_block = nn.Sequential(
            nn.Conv2d(
                in_channels=n_feat,
                out_channels=n_feat,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(
                in_channels=n_feat,
                out_channels=n_feat,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
        )

    def forward(self, x):
        identity = x
        out = self.res_block(x)
        return out + identity


class Downsample(nn.Module):
    def __init__(self, in_channels, scale):
        super(Downsample, self).__init__()
        self.downsample = nn.Sequential(
            nn.PixelUnshuffle(scale),
            nn.Conv2d(
                in_channels=in_channels * scale * scale,
                out_channels=in_channels * scale,
                kernel_size=1,
            ),
        )

    def forward(self, x):
        return self.downsample(x)

class Upsample(nn.Module):
    def __init__(self, in_channels, scale):
        super(Upsample, self).__init__()
        self.upsample = nn.Sequential(
            nn.PixelShuffle(scale),
            nn.Conv2d(
                in_channels=in_channels // (scale * scale),
                out_channels=in_channels // (scale * scale),
                kernel_size=1,
            ),
        )

    def forward(self, x):
        return self.upsample(x)

# invertible modules

class Conv3x3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_channels, in_channels, 3, 1, 1)

    def forward(self, input, rev=False):
        if not rev:
            z = self.conv1(input)
            # print(z.size())
            return z
        else:
            z = self.conv2(input)
            return z

class InvPixelUnshuffle(nn.Module):
    def __init__(self, downscale_factor):
        super().__init__()
        self.pixel_shuffle = nn.PixelShuffle(downscale_factor)
        self.pixel_unshuffle = nn.PixelUnshuffle(downscale_factor)
        
    def forward(self, input, rev=False):
        if not rev:
            z = self.pixel_unshuffle(input)
            return z
        else:
            z = self.pixel_shuffle(input)
            return z

class InvBlockExp(nn.Module):
    def __init__(self, split_len1, split_len2, clamp=1.):
        super(InvBlockExp, self).__init__()

        self.split_len1 = split_len1 # 
        self.split_len2 = split_len2 #

        self.clamp = clamp

        self.F = DenseBlock(self.split_len2, self.split_len1)
        self.G = DenseBlock(self.split_len1, self.split_len2)
        self.H = DenseBlock(self.split_len1, self.split_len2)

    def forward(self, x, rev=False):
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

        if not rev:
            y1 = x1 + self.F(x2)
            self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
            y2 = x2.mul(torch.exp(self.s)) + self.G(y1)
        else:
            self.s = self.clamp * (torch.sigmoid(self.H(x1)) * 2 - 1)
            y2 = (x2 - self.G(x1)).div(torch.exp(self.s))
            y1 = x1 - self.F(y2)

        return torch.cat((y1, y2), 1)

    def jacobian(self, x, rev=False):
        if not rev:
            jac = torch.sum(self.s)
        else:
            jac = -torch.sum(self.s)

        return jac / x.shape[0]

class DenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=32, bias=True):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(channel_in + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(channel_in + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(channel_in + 4 * gc, channel_out, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if init == 'xavier':
            initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        else:
            initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        initialize_weights(self.conv5, 0)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))

        return x5

# if __name__ == "__main__":
#     x = torch.randn(4, 3, 256, 256).cuda()
#     x_latent = torch.randn(4, 4, 32, 32).cuda()
#     model = ICRM(in_channels=4, out_channels=3, ir_scale=16).cuda()
#     num_param = sum(p.numel() for p in model.parameters())
#     print(num_param / (1000**2))
#     y, _ = model(x, x_latent, rev=False)
#     z = model(y, rev=True)
#     print(y.shape, z.shape)
