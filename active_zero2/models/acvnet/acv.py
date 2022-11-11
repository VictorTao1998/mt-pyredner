from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
from active_zero2.models.acvnet.submodule import *
import math
import gc
import time
from active_zero2.utils.reprojection import compute_reproj_loss_patch


class feature_extraction(nn.Module):
    def __init__(self, in_channels):
        super(feature_extraction, self).__init__()

        self.inplanes = 32
        self.firstconv = nn.Sequential(
            convbn(in_channels, 32, 3, 2, 1, 1),
            nn.ReLU(inplace=True),
            convbn(32, 32, 3, 1, 1, 1),
            nn.ReLU(inplace=True),
            convbn(32, 32, 3, 1, 1, 1),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1, 1, 1)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1, 1, 2)

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.firstconv(x)
        x = self.layer1(x)
        l2 = self.layer2(x)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)
        gwc_feature = torch.cat((l2, l3, l4), dim=1)
        return gwc_feature


class hourglass(nn.Module):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(in_channels, in_channels * 2, 3, 2, 1), nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 2, 3, 1, 1), nn.ReLU(inplace=True))

        self.conv3 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 4, 3, 2, 1), nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn_3d(in_channels * 4, in_channels * 4, 3, 1, 1), nn.ReLU(inplace=True))

        self.attention_block = attention_block(channels_3d=in_channels * 4, num_heads=16, block=(4, 4, 4))

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels * 2),
        )

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels),
        )

        self.redir1 = convbn_3d(in_channels, in_channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = convbn_3d(in_channels * 2, in_channels * 2, kernel_size=1, stride=1, pad=0)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv4 = self.attention_block(conv4)
        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(x), inplace=True)
        return conv6


class ACVNet(nn.Module):
    def __init__(
        self,
        num_ir: int,
        min_disp: float,
        max_disp: float,
        num_disp: int,
    ):
        super(ACVNet, self).__init__()
        self.num_ir = num_ir
        self.min_disp = min_disp
        self.max_disp = max_disp
        self.num_disp = num_disp
        assert num_disp % 4 == 0, "Num_disp % 4 should be 0"
        self.num_disp_4 = num_disp // 4
        self.disp_list_4 = torch.linspace(min_disp, max_disp, self.num_disp_4) / 4

        self.num_groups = 40
        self.concat_channels = 32
        self.feature_extraction = feature_extraction(in_channels=num_ir)
        self.concatconv = nn.Sequential(
            convbn(320, 128, 3, 1, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, self.concat_channels, kernel_size=1, padding=0, stride=1, bias=False),
        )

        self.patch = nn.Conv3d(
            40, 40, kernel_size=(1, 3, 3), stride=1, dilation=1, groups=40, padding=(0, 1, 1), bias=False
        )
        self.patch_l1 = nn.Conv3d(
            8, 8, kernel_size=(1, 3, 3), stride=1, dilation=1, groups=8, padding=(0, 1, 1), bias=False
        )
        self.patch_l2 = nn.Conv3d(
            16, 16, kernel_size=(1, 3, 3), stride=1, dilation=2, groups=16, padding=(0, 2, 2), bias=False
        )
        self.patch_l3 = nn.Conv3d(
            16, 16, kernel_size=(1, 3, 3), stride=1, dilation=3, groups=16, padding=(0, 3, 3), bias=False
        )

        self.dres1_att_ = nn.Sequential(convbn_3d(40, 32, 3, 1, 1), nn.ReLU(inplace=True), convbn_3d(32, 32, 3, 1, 1))
        self.dres2_att_ = hourglass(32)
        self.classif_att_ = nn.Sequential(
            convbn_3d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False),
        )

        self.dres0 = nn.Sequential(
            convbn_3d(self.concat_channels * 2, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            convbn_3d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
        )

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1), nn.ReLU(inplace=True), convbn_3d(32, 32, 3, 1, 1))
        self.dres2 = hourglass(32)

        self.dres3 = hourglass(32)

        self.classif0 = nn.Sequential(
            convbn_3d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False),
        )

        self.classif1 = nn.Sequential(
            convbn_3d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False),
        )

        self.classif2 = nn.Sequential(
            convbn_3d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, data_batch, attn_weights_only=False):
        left, right = data_batch["img_l"], data_batch["img_r"]
        disp = torch.linspace(self.min_disp, self.max_disp, self.num_disp).view(1, self.num_disp, 1, 1).to(left.device)

        features_left = self.feature_extraction(left)
        features_right = self.feature_extraction(right)

        # build GWC volume
        disp_list_4 = self.disp_list_4.to(features_left.device)
        [bs, feature_size, H, W] = features_left.size()
        # Original coordinates of pixels
        x_base = (
            torch.linspace(0, 1, W, dtype=features_left.dtype, device=features_left.device)
            .view(1, 1, W, 1)
            .expand(bs, H, W, self.num_disp_4)
        )
        y_base = (
            torch.linspace(0, 1, H, dtype=features_left.dtype, device=features_left.device)
            .view(1, H, 1, 1)
            .expand(bs, H, W, self.num_disp_4)
        )
        disp_grid = (disp_list_4 / (W - 1)).view(1, 1, 1, self.num_disp_4).expand(bs, H, W, self.num_disp_4)
        target_grids = torch.stack((x_base - disp_grid, y_base), dim=-1).view(bs, H, W * self.num_disp_4, 2)
        target_cost_volume = F.grid_sample(
            features_right, 2 * target_grids - 1, mode="bilinear", padding_mode="zeros", align_corners=True
        )
        target_cost_volume = target_cost_volume.view(bs, feature_size, H, W, self.num_disp_4).permute(0, 1, 4, 2, 3)
        ref_cost_volume = features_left.unsqueeze(2).expand(bs, feature_size, self.num_disp_4, H, W)
        gwc_volume = groupwise_correlation_volume(ref_cost_volume, target_cost_volume, self.num_groups)
        gwc_volume = self.patch(gwc_volume)
        patch_l1 = self.patch_l1(gwc_volume[:, :8])
        patch_l2 = self.patch_l2(gwc_volume[:, 8:24])
        patch_l3 = self.patch_l3(gwc_volume[:, 24:40])
        patch_volume = torch.cat((patch_l1, patch_l2, patch_l3), dim=1)
        cost_attention = self.dres1_att_(patch_volume)
        cost_attention = self.dres2_att_(cost_attention)
        att_weights = self.classif_att_(cost_attention)

        if not attn_weights_only:
            concat_feature_left = self.concatconv(features_left)
            concat_feature_right = self.concatconv(features_right)
            [bs, feature_size, H, W] = concat_feature_left.size()
            target_cost_volume = F.grid_sample(
                concat_feature_right, 2 * target_grids - 1, mode="bilinear", padding_mode="zeros", align_corners=True
            )
            target_cost_volume = target_cost_volume.view(bs, feature_size, H, W, self.num_disp_4).permute(0, 1, 4, 2, 3)
            ref_cost_volume = concat_feature_left.unsqueeze(2).expand(bs, feature_size, self.num_disp_4, H, W)
            concat_volume = torch.cat((ref_cost_volume, target_cost_volume), dim=1)
            ac_volume = F.softmax(att_weights, dim=2) * concat_volume  ### ac_volume = att_weights * concat_volume
            cost0 = self.dres0(ac_volume)
            cost0 = self.dres1(cost0) + cost0
            out1 = self.dres2(cost0)
            out2 = self.dres3(out1)

        if self.training:
            cost_attention = F.upsample(
                att_weights,
                [self.num_disp, left.size()[2], left.size()[3]],
                mode="trilinear",
                align_corners=False,
            )
            cost_attention = torch.squeeze(cost_attention, 1)
            pred_attention = F.softmax(cost_attention, dim=1)
            pred_attention = torch.sum(pred_attention * disp, 1, keepdim=True)

            if not attn_weights_only:

                cost0 = self.classif0(cost0)
                cost1 = self.classif1(out1)
                cost2 = self.classif2(out2)

                cost0 = F.upsample(
                    cost0, [self.num_disp, left.size()[2], left.size()[3]], mode="trilinear", align_corners=False
                )
                cost0 = torch.squeeze(cost0, 1)
                pred0 = F.softmax(cost0, dim=1)
                pred0 = torch.sum(pred0 * disp, 1, keepdim=True)

                cost1 = F.upsample(
                    cost1, [self.num_disp, left.size()[2], left.size()[3]], mode="trilinear", align_corners=False
                )
                cost1 = torch.squeeze(cost1, 1)
                pred1 = F.softmax(cost1, dim=1)
                pred1 = torch.sum(pred1 * disp, 1, keepdim=True)

                cost2 = F.upsample(
                    cost2, [self.num_disp, left.size()[2], left.size()[3]], mode="trilinear", align_corners=False
                )
                cost2 = torch.squeeze(cost2, 1)
                pred2 = F.softmax(cost2, dim=1)
                pred2 = torch.sum(pred2 * disp, 1, keepdim=True)

                return {
                    "pred_attention": pred_attention,
                    "pred0": pred0,
                    "pred1": pred1,
                    "pred2": pred2,
                }

            return {"pred_attention": pred_attention}
        else:
            cost2 = self.classif2(out2)
            cost2 = F.upsample(
                cost2, [self.num_disp, left.size()[2], left.size()[3]], mode="trilinear", align_corners=False
            )
            cost2 = torch.squeeze(cost2, 1)
            pred2 = F.softmax(cost2, dim=1)
            pred2 = torch.sum(pred2 * disp, 1, keepdim=True)

            return {
                "pred2": pred2,
            }

    def compute_disp_loss(self, data_batch, pred_dict):
        disp_gt = data_batch["img_disp_l"]
        # Get stereo loss on sim
        # Note in training we do not exclude bg
        mask = (disp_gt < self.max_disp) * (disp_gt > self.min_disp)
        mask.detach()
        loss_disp = F.smooth_l1_loss(pred_dict["pred_attention"][mask], disp_gt[mask], reduction="mean")
        if "pred2" in pred_dict:
            loss_disp /= 2.0
            for pred_name, loss_weight in zip(["pred0", "pred1", "pred2"], [0.5, 0.7, 1.0]):
                if pred_name in pred_dict:
                    loss_disp += loss_weight * F.smooth_l1_loss(
                        pred_dict[pred_name][mask], disp_gt[mask], reduction="mean"
                    )

        return loss_disp

    def compute_reproj_loss(self, data_batch, pred_dict, use_mask: bool, patch_size: int, only_last_pred: bool):
        if use_mask:
            disp_gt = data_batch["img_disp_l"]
            # Get stereo loss on sim
            # Note in training we do not exclude bg
            mask = (disp_gt < self.max_disp) * (disp_gt > self.min_disp)
            mask.detach()
        else:
            mask = None
        if only_last_pred:
            if "pred2" in pred_dict:
                pred_disp = pred_dict["pred2"]
            else:
                pred_disp = pred_dict["pred_attention"]
            loss_reproj = compute_reproj_loss_patch(
                data_batch["img_pattern_l"],
                data_batch["img_pattern_r"],
                pred_disp_l=pred_disp,
                mask=mask,
                ps=patch_size,
            )

            return loss_reproj
        else:
            raise NotImplementedError
