import itertools
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lscorer.config import configs
from torch.nn.init import kaiming_normal

FEATURE_DIM = 5

class LineScorer(nn.Module):
    def __init__(self, configs, backbone, dropout_rate=0.5):
        super().__init__()

        # use hourglass_pose network as a backbone
        self.backbone = backbone

        self.batchsize = configs.model.batch_size

        self.configs = configs

        lambda_ = torch.linspace(0, 1, self.configs.model.n_pts0)[:, None]
        self.register_buffer("lambda_", lambda_)
        self.fc1 = nn.Conv2d(256, self.configs.model.dim_loi, 1)
        scale_factor = self.configs.model.n_pts0 // self.configs.model.n_pts1
        if self.configs.model.use_conv:
            self.pooling = nn.Sequential(
                nn.MaxPool1d(scale_factor, scale_factor),
                Bottleneck1D(self.configs.model.dim_loi, self.configs.model.dim_loi),
            )
            self.fc2 = nn.Sequential(
                nn.ReLU(inplace=True), nn.Linear(self.configs.model.dim_loi * self.configs.model.n_pts1 + FEATURE_DIM, 4)
            )
        else:
            self.pooling = nn.MaxPool1d(kernel_size = scale_factor, stride = scale_factor)
            self.fc2 = nn.Sequential(
                nn.Linear(configs.model.dim_loi * configs.model.n_pts1 * 2 + FEATURE_DIM, configs.model.dim_fc),
                nn.ReLU(inplace=True),
                nn.Linear(configs.model.dim_fc, configs.model.dim_fc),
                nn.ReLU(inplace=True),
                nn.Linear(configs.model.dim_fc, 4),
            )

        self.drop_layer = nn.Dropout(p=dropout_rate)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, image, lines, trans_vecs, heatmap):

        # use backbone to produce feature map.
        lines_feature_map = self.backbone(image)[1]

        n_batch, n_channel, row, col = lines_feature_map.size()
        xs, ys = [], []
        fs = []

        for i in range(lines.size(0)):
            p = lines[i] # 64x2x2
            feature = lines_feature_map[i]

            # Attention using heatmap.
            # feature = self.heatmap_attention(feature, heatmap[i])

            feat = self.form_line_geometry_feature(p, configs)

            p = p[:, 0:1, :] * self.lambda_ + p[:, 1:2, :] * (1 - self.lambda_) - 0.5
            p = p.reshape(-1, 2)  # [N_LINE x N_POINT, 2_XY]
            px, py = p[:, 0].contiguous(), p[:, 1].contiguous()
            px0 = px.floor().clamp(min=0, max=127)
            py0 = py.floor().clamp(min=0, max=127)
            px1 = (px0 + 1).clamp(min=0, max=127)
            py1 = (py0 + 1).clamp(min=0, max=127)
            px0l, py0l, px1l, py1l = px0.long(), py0.long(), px1.long(), py1.long()

            # xp = (
            #     (
            #         feature[:, px0l, py0l] * (px1 - px) * (py1 - py)
            #         + feature[:, px1l, py0l] * (px - px0) * (py1 - py)
            #         + feature[:, px0l, py1l] * (px1 - px) * (py - py0)
            #         + feature[:, px1l, py1l] * (px - px0) * (py - py0)
            #     )
            #     .reshape(n_channel, -1, self.configs.model.n_pts0)
            #     .permute(1, 0, 2)
            # )

            xp = (
                (
                    feature[:, py0l, px0l] * (px1 - px) * (py1 - py)
                    + feature[:, py1l, px0l] * (px - px0) * (py1 - py)
                    + feature[:, py0l, px1l] * (px1 - px) * (py - py0)
                    + feature[:, py1l, px1l] * (px - px0) * (py - py0)
                )
                .reshape(n_channel, -1, self.configs.model.n_pts0)
                .permute(1, 0, 2)
            )

            xp = self.pooling(xp)
            xs.append(xp)
            fs.append(feat)

        x = torch.cat(xs)
        f = torch.cat(fs)
        x = x.reshape(x.size(0), x.size(1) * x.size(2))
        x = torch.cat([x, f], 1)

        # # Added dropout layer
        # x = self.drop_layer(x)

        x = self.fc2(x) # x: 128 x 4

        func_sftmx = nn.Softmax(dim = 1)
        prob = func_sftmx(x)

        return x, prob

    def heatmap_attention(self, feat, heatmap):
        # feat: 256 x 128 x 128
        # heatmap: 1 x 128 x 128

        return feat * heatmap

    def form_line_geometry_feature(self, lines, configs):
        # Use line end point location and line length as features
        # lines 64 x 2 x 2
        feat = []

        xyu = lines[:, 0, :]
        xyv = lines[:, 1, :]
        u2v = torch.sqrt(((lines[:, 0, :] - lines[:, 1, :])**2).sum(-1, keepdim=True)).clamp(min=1e-6)

        feat = torch.cat(
            [
                xyu / 128 * configs.model.use_cood, # scale to [0, 1]
                xyv / 128 * configs.model.use_cood, # scale to [0, 1]
                u2v * configs.model.use_slop
            ],
            1,
        )

        return feat

class Bottleneck1D(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(Bottleneck1D, self).__init__()

        planes = outplanes // 2
        self.op = nn.Sequential(
            nn.BatchNorm1d(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv1d(inplanes, planes, kernel_size=1),
            nn.BatchNorm1d(planes),
            nn.ReLU(inplace=True),
            nn.Conv1d(planes, planes, kernel_size=3, padding=1),
            nn.BatchNorm1d(planes),
            nn.ReLU(inplace=True),
            nn.Conv1d(planes, outplanes, kernel_size=1),
        )

    def forward(self, x):
        return x + self.op(x)



# Testing
# lines shape: torch.Size([2, 64, 2, 2])
# trans_vecs p1 shape: torch.Size([2, 64, 2])
# trans_vecs p2 shape: torch.Size([2, 64, 2])
# layout_line_type shape: torch.Size([2, 64, 4])
# line_feature_map shape: torch.Size([2, 256, 128, 128])
# heatmap shape: torch.Size([2, 1, 128, 128])
#
# def TestNetwork():
#     config_file = "../../config/lsun.yaml"
#     configs.update(configs.from_yaml(filename=config_file))
#
#     lines = torch.rand(2, 64, 2, 2)
#     trans_vecs_p1 = torch.rand(2, 64, 2)
#     trans_vecs_p2 = torch.rand(2, 64, 2)
#     trans_vecs = [trans_vecs_p1, trans_vecs_p2]
#     layout_lines_type = torch.rand(2, 64, 4)
#     lines_feature_map = torch.rand(2, 256, 128, 128)
#     heatmap = torch.rand(2, 1, 128, 128)
#
#     lscorer = LineScorer(configs, batchsize=2)
#
#     y = lscorer(lines, trans_vecs, layout_lines_type, lines_feature_map, heatmap)
#
#     zx = 0
#
# TestNetwork()