import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from basicsr.archs.arch_util import default_init_weights


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU()

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.relu(x)
        out = self.conv2(x)
        return identity + out * self.res_scale


# ConvBlock
class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        zero_conv=False,
    ):
        super(ConvBlock, self).__init__()
        if zero_conv:
            self.conv_in = zero_module(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
            )
        else:
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


# MlpBlock
class MlpBlock(nn.Module):
    def __init__(self, in_channels=4, out_channels=1, hidden_channels=32, num_layers=5):
        super(MlpBlock, self).__init__()
        fuse_mlp = [nn.Linear(in_channels, hidden_channels)]
        for _ in range(num_layers):
            fuse_mlp.append(nn.ReLU(True))
            fuse_mlp.append(nn.Linear(hidden_channels, hidden_channels))
        fuse_mlp.append(nn.Linear(hidden_channels, out_channels))
        fuse_mlp.append(nn.Sigmoid())
        self.fuse_mlp = nn.Sequential(*fuse_mlp)

    def forward(self, x):
        # [B, in_channels] -> [B, 1]
        return self.fuse_mlp(x)


# TimeMapping block
class TimeMapping(nn.Module):

    def __init__(self, in_channels=4, out_channels=1, lower_limit=10, upper_limit=200):
        super(TimeMapping, self).__init__()
        noise_esti = [ConvBlock(in_channels, 32)]
        for _ in range(6):
            noise_esti.append(ResidualBlockNoBN(32))
        self.noise_esti = nn.Sequential(*noise_esti)
        self.time_predict = MlpBlock(64, out_channels, 512, 4)
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit

    def forward(self, x):
        # [B, C, H, W] -> [B, out], whose value follows [lower_limit, upper_limit]
        noise_embed = self.noise_esti(x)
        embed_mean = torch.mean(noise_embed, dim=[-1, -2])
        embed_std = torch.std(noise_embed, dim=[-1, -2])
        noise_embed = torch.cat([embed_mean, embed_std], dim=1)
        noise_embed = self.time_predict(noise_embed)
        if noise_embed.shape[-1] > 1:
            time_embed = (
                noise_embed[:, 0] * (self.upper_limit - self.lower_limit)
                + self.lower_limit
            )
            sqrt_alpha_hat = noise_embed[:, 1] * 0.5 + 0.5
            return time_embed, sqrt_alpha_hat
        else:
            time_embed = (
                noise_embed[:, 0] * (self.upper_limit - self.lower_limit)
                + self.lower_limit
            )
            return time_embed


# Refiner block
class Refiner(nn.Module):

    def __init__(self, in_channels=8, out_channels=4, hidden_channels=64):
        super(Refiner, self).__init__()
        self.time_proj = Timesteps(hidden_channels, False, 1)
        self.time_embedding = TimestepEmbedding(hidden_channels, hidden_channels)

        resnet_blocks = [ConvBlock(in_channels, hidden_channels)]
        for _ in range(6):
            resnet_blocks.append(ResidualBlockNoBN(hidden_channels))
        self.resnet_blocks = nn.ModuleList(resnet_blocks)
        self.zero_conv_out = ConvBlock(hidden_channels, out_channels, zero_conv=True)

    def forward(self, z_hat, noise, ts):
        t_emb = self.time_proj(ts)
        t_emb = t_emb.to(dtype=z_hat.dtype)
        t_emb = self.time_embedding(t_emb).unsqueeze(2).unsqueeze(3)

        feats = self.resnet_blocks[0](torch.cat([z_hat, noise], dim=1))
        for module in self.resnet_blocks[1:]:
            feats = module(feats) + t_emb

        return self.zero_conv_out(feats)
