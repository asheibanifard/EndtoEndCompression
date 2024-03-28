from utils import make_coord
import torch
import math
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from scipy.fft import fft2, ifft2, fftshift


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv3d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)


class ResBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size,
                 bias=True, act=nn.ReLU(True), res_scale=2):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


def LR_image_producer(volume, scale):
    k_space = fftshift(fft2(volume))
    central_fraction = 0.5
    deps, rows, cols = k_space.shape
    center_dep, center_row, center_col = deps // 2, rows // 2, cols // 2
    central_deps = int(central_fraction * deps)
    central_rows = int(central_fraction * rows)
    central_cols = int(central_fraction * cols)

    selected_k_space = k_space[center_dep - central_deps // scale:
                               center_dep + central_deps // scale,
                               center_row - central_rows // scale:
                               center_row + central_rows // scale,
                               center_col - central_cols // scale:
                               center_col + central_cols // scale]
    reconstructed_image = ifft2(fftshift(selected_k_space))
    if np.abs(reconstructed_image).max() != 0:
        if np.abs(reconstructed_image).max() != np.abs(reconstructed_image).\
                min():
            reconstructed = 2 * (np.abs(reconstructed_image) -
                                 np.abs(reconstructed_image).min()) / \
                (np.abs(reconstructed_image).max() -
                 np.abs(reconstructed_image).min()) - 1
    else:
        reconstructed = np.abs(reconstructed_image)
    return reconstructed


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        out = (x - mean) / (std + self.eps)
        out = self.weight * out + self.bias
        return out


class PixelShuffle3D(nn.Module):
    def __init__(self, upscale_factor):
        super(PixelShuffle3D, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        batch_size, num_channels, depth, height, width = x.size()

        # Check if the dimensions are divisible by the upscale factor
        if depth % self.upscale_factor != 0 \
            or height % self.upscale_factor != 0 \
                or width % self.upscale_factor != 0:
            raise ValueError(
                "Input dimensions must be divisible by the upscale factor")

        # Reshape the input tensor
        x = x.view(batch_size, num_channels, depth // self.upscale_factor,
                   self.upscale_factor,
                   height // self.upscale_factor, self.upscale_factor,
                   width // self.upscale_factor, self.upscale_factor)

        # Rearrange the dimensions
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6)

        # Reshape again to get the final output
        x = x.contiguous().view(batch_size,
                                num_channels // self.upscale_factor**3,
                                depth * self.upscale_factor,
                                height * self.upscale_factor,
                                width * self.upscale_factor)

        return x


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, n_feats, 3, bias))
                m.append(PixelShuffle3D(scale))
                m.append(nn.ReLU(True))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(scale))
            m.append(nn.ReLU(True))

        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class EDSR(nn.Module):
    def __init__(self, n_colors, n_feats, scale, n_resblocks,
                 conv=default_conv):
        super(EDSR, self).__init__()
        kernel_size = 3
        act = nn.ReLU(True)
        m_head = [conv(n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            ResBlock(
                conv, n_feats, kernel_size, act=act
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)

        self.out_dim = n_colors
        # define tail module
        m_tail = [
            nn.Conv3d(n_feats, scale * scale * scale,
                      kernel_size=3, stride=1, padding=1),
            PixelShuffle3D(scale)
        ]
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        # x = self.sub_mean(x)

        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        # x = self.add_mean(x)
        return x


class simple_attn(nn.Module):
    def __init__(self, midc, heads):
        super().__init__()

        self.headc = midc // heads
        self.heads = heads
        self.midc = midc

        self.qkv_proj = nn.Conv3d(midc, 3 * midc, 1)
        self.o_proj1 = nn.Conv3d(midc, midc, 1)
        self.o_proj2 = nn.Conv3d(midc, midc, 1)

        self.kln = LayerNorm((self.heads, 1, self.headc))
        self.vln = LayerNorm((self.heads, 1, self.headc))

        self.act = nn.GELU()

    def forward(self, x, name='0'):
        B, C, D, H, W = x.shape
        bias = x

        qkv = self.qkv_proj(x).permute(0, 2, 3, 4, 1).\
            reshape(B, D * H * W, self.heads, 3 * self.headc)

        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)

        k = self.kln(k)
        v = self.vln(v)

        v = torch.matmul(k.transpose(-2, -1), v) / (D * H * W)
        v = torch.matmul(q, v)
        v = v.permute(0, 2, 1, 3).reshape(B, D, H, W, C)
        ret = v.permute(0, 4, 1, 2, 3) + bias
        bias = self.o_proj2(self.act(self.o_proj1(ret))) + bias
        return bias


class SRNO3D(nn.Module):

    def __init__(self, scale, coord, cell, n_feat_edsr, n_resblocks, width,
                 blocks):
        super().__init__()
        self.width = width
        self.coord = coord
        self.cell = cell
        self.conv00 = nn.Conv3d(35, self.width, 1)
        self.conv0 = simple_attn(self.width, blocks)
        self.conv1 = simple_attn(self.width, blocks)
        self.fc1 = nn.Conv3d(self.width, 256, 1)
        self.fc2 = nn.Conv3d(256, 1, 1)
        self.encoder = EDSR(1, n_feat_edsr, scale, n_resblocks)

    def gen_feat(self, inp):
        self.inp = inp
        self.feat = self.encoder(inp)
        return self.feat

    def query_rgb(self, coord, cell):
        feat = (self.feat)
        grid = 0
        pos_lr = make_coord(feat.shape[2:], flatten=False).cuda() \
            .permute(3, 0, 1, 2) \
            .unsqueeze(0).expand(feat.shape[0], 3, *feat.shape[2:])
        rz = 2 / feat.shape[-3] / 2
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2
        vz_lst = [-1, 1]
        vx_lst = [-1, 1]
        vy_lst = [-1, 1]
        eps_shift = 1e-6

        rel_coords = []
        feat_s = []
        areas = []
        for vz in vz_lst:
            for vx in vx_lst:
                for vy in vy_lst:
                    coord_ = coord.clone()
                    coord_[:, :, :, 0] += vz * rz + eps_shift
                    coord_[:, :, :, 1] += vx * rx + eps_shift
                    coord_[:, :, :, 2] += vy * ry + eps_shift

                    coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                    feat_ = F.grid_sample(feat, coord_.unsqueeze(0).flip(-1),
                                          mode='nearest',
                                          align_corners=False)

                    old_coord = F.grid_sample(pos_lr, coord_.unsqueeze(0).
                                              flip(-1), mode='nearest',
                                              align_corners=False)
                    rel_coord = coord.permute(3, 0, 1, 2) - old_coord
                    rel_coord[:, 0, :, :] *= feat.shape[-3]
                    rel_coord[:, 1, :, :] *= feat.shape[-2]
                    rel_coord[:, 2, :, :] *= feat.shape[-1]

                    area = torch.abs(rel_coord[:, 0, :, :, :] *
                                     rel_coord[:, 1, :, :, :] *
                                     rel_coord[:, 2, :, :, :])
                    areas.append(area + 1e-9)

                    rel_coords.append(rel_coord)
                    feat_s.append(feat_)
        rel_cell = cell.clone()

        rel_cell[0, :] *= feat.shape[-3]
        rel_cell[1, :] *= feat.shape[-2]
        rel_cell[2, :] *= feat.shape[-1]

        tot_area = torch.stack(areas).sum(dim=0)

        t = areas[0]
        areas[0] = areas[2]
        areas[2] = t

        t = areas[1]
        areas[1] = areas[3]
        areas[3] = t

        t = areas[2]
        areas[2] = areas[4]
        areas[4] = t

        for index, area in enumerate(areas):
            feat_s[index] = feat_s[index] * (area / tot_area).unsqueeze(1)
        grid = torch.cat([*rel_coords,
                          *feat_s,
                          rel_cell. unsqueeze(-1).unsqueeze(-1).
                          repeat(1,
                                 1,
                                 coord.shape[0],
                                 coord.shape[1],
                                 coord.shape[2])], dim=1)
        x = self.conv00(grid)
        x = self.conv0(x, 0)
        x = self.conv1(x, 1)

        feat = x
        ret = self.fc2(F.gelu(self.fc1(feat)))
        ret = ret + F.grid_sample(self.inp,
                                  coord.unsqueeze(0).flip(-1),
                                  mode='bilinear',
                                  padding_mode='border',
                                  align_corners=False)
        return ret

    def forward(self, inp):
        self.gen_feat(inp)
        return self.query_rgb(self.coord, self.cell)
