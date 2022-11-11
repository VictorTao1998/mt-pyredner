"""
"""

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from active_zero2.models.psmnet_kpac.psmnet_submodule_3 import *
from active_zero2.utils.reprojection import compute_reproj_loss_patch


class kpac3d(nn.Module):
    def __init__(self, inplanes, outplanes, stride=2, dilation_list=(1, 3, 5, 9)):
        super(kpac3d, self).__init__()
        self.inplanes = inplanes
        self.outplanes = outplanes
        self.stride = stride
        self.dilation_list = dilation_list
        self.conv3d_0_weight = nn.Parameter(torch.randn(outplanes, inplanes, 3, 3, 3))
        self.conv3d_bn = nn.Sequential(nn.BatchNorm3d(outplanes * len(dilation_list)), nn.ReLU(inplace=False))
        self.conv3d_1 = nn.Sequential(
            nn.Conv3d(outplanes * len(dilation_list), outplanes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(outplanes),
            nn.ReLU(inplace=False),
        )
        self._init_weight()

    def forward(self, x):
        y = []
        for d in self.dilation_list:
            y.append(F.conv3d(x, self.conv3d_0_weight, stride=self.stride, padding=d, dilation=d))
        y = torch.cat(y, dim=1)
        y = self.conv3d_bn(y)
        y = self.conv3d_1(y)
        return y

    def _init_weight(self):
        init.kaiming_uniform_(self.conv3d_0_weight, a=math.sqrt(5))


class hourglass(nn.Module):
    def __init__(self, inplanes, dilation=3, dilation_list=(1, 3, 5, 9)):
        super(hourglass, self).__init__()

        self.conv1 = kpac3d(inplanes, inplanes * 2, stride=2, dilation_list=dilation_list)

        self.conv2 = kpac3d(inplanes * 2, inplanes * 2, stride=1, dilation_list=dilation_list)

        self.conv3 = kpac3d(inplanes * 2, inplanes * 2, stride=2, dilation_list=dilation_list)

        self.conv4 = kpac3d(inplanes * 2, inplanes * 2, stride=1, dilation_list=dilation_list)

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(
                inplanes * 2,
                inplanes * 2,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                output_padding=1,
                stride=2,
                bias=False,
            ),
            nn.BatchNorm3d(inplanes * 2),
        )

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(
                inplanes * 2,
                inplanes,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                output_padding=1,
                stride=2,
                bias=False,
            ),
            nn.BatchNorm3d(inplanes),
        )

    def forward(self, x, presqu, postqu):
        out = self.conv1(x)
        pre = self.conv2(out)
        if postqu is not None:
            pre = F.relu(pre + postqu, inplace=False)
        else:
            pre = F.relu(pre, inplace=False)

        out = self.conv3(pre)
        out = self.conv4(out)

        if presqu is not None:
            post = F.relu(self.conv5(out) + presqu, inplace=False)
        else:
            post = F.relu(self.conv5(out) + pre, inplace=False)

        out = self.conv6(post)
        return out, pre, post


class PSMNetKPAC(nn.Module):
    def __init__(
        self, min_disp: float, max_disp: float, num_disp: int, set_zero: bool, dilation: int, dilation_list: List[int]
    ):
        super(PSMNetKPAC, self).__init__()
        self.min_disp = min_disp
        self.max_disp = max_disp
        self.num_disp = num_disp
        self.dilation = dilation
        self.dilation_list = dilation_list
        assert num_disp % 4 == 0, "Num_disp % 4 should be 0"
        self.num_disp_4 = num_disp // 4
        self.set_zero = set_zero  # set zero for invalid reference image cost volume

        self.disp_list = torch.linspace(min_disp, max_disp, num_disp)
        self.disp_list_4 = torch.linspace(min_disp, max_disp, self.num_disp_4) / 4
        self.disp_regression = DisparityRegression(min_disp, max_disp, num_disp)

        self.feature_extraction = FeatureExtraction()

        self.dres0 = nn.Sequential(
            convbn_3d(64, 32, 3, 1, 1),
            nn.ReLU(inplace=False),
            convbn_3d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=False),
        )
        self.dres1 = nn.Sequential(
            convbn_3d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=False),
            convbn_3d(32, 32, 3, 1, 1),
        )

        self.dres2 = hourglass(32, dilation, dilation_list)
        self.dres3 = hourglass(32, dilation, dilation_list)
        self.dres4 = hourglass(32, dilation, dilation_list)

        self.classif1 = nn.Sequential(
            convbn_3d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=False),
            nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False),
        )
        self.classif2 = nn.Sequential(
            convbn_3d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=False),
            nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False),
        )
        self.classif3 = nn.Sequential(
            convbn_3d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=False),
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

    def forward(self, data_batch):
        img_L, img_R = data_batch["img_l"], data_batch["img_r"]
        refimg_feature = self.feature_extraction(img_L)  # [bs, 32, H/4, W/4]
        targetimg_feature = self.feature_extraction(img_R)

        disp_list = self.disp_list.to(refimg_feature.device)
        disp_list_4 = self.disp_list_4.to(refimg_feature.device)

        # Cost Volume
        [bs, feature_size, H, W] = refimg_feature.size()
        # Original coordinates of pixels
        x_base = (
            torch.linspace(0, 1, W, dtype=refimg_feature.dtype, device=refimg_feature.device)
            .view(1, 1, W, 1)
            .expand(bs, H, W, self.num_disp_4)
        )
        y_base = (
            torch.linspace(0, 1, H, dtype=refimg_feature.dtype, device=refimg_feature.device)
            .view(1, H, 1, 1)
            .expand(bs, H, W, self.num_disp_4)
        )
        disp_grid = (disp_list_4 / (W - 1)).view(1, 1, 1, self.num_disp_4).expand(bs, H, W, self.num_disp_4)
        target_grids = torch.stack((x_base - disp_grid, y_base), dim=-1).view(bs, H, W * self.num_disp_4, 2)
        target_cost_volume = F.grid_sample(
            targetimg_feature, 2 * target_grids - 1, mode="bilinear", padding_mode="zeros", align_corners=True
        )
        target_cost_volume = target_cost_volume.view(bs, feature_size, H, W, self.num_disp_4).permute(0, 1, 4, 2, 3)
        ref_cost_volume = refimg_feature.unsqueeze(2).expand(bs, feature_size, self.num_disp_4, H, W)
        if self.set_zero:
            # set invalid area to zero
            valid_mask = (x_base > disp_grid).permute(0, 3, 1, 2).unsqueeze(1)
            ref_cost_volume = ref_cost_volume * valid_mask

        cost = torch.cat((ref_cost_volume, target_cost_volume), dim=1)

        cost0 = self.dres0(cost)
        cost0 = self.dres1(cost0) + cost0

        out1, pre1, post1 = self.dres2(cost0, None, None)
        out1 = out1 + cost0

        out2, pre2, post2 = self.dres3(out1, pre1, post1)
        out2 = out2 + cost0

        out3, pre3, post3 = self.dres4(out2, pre1, post2)
        out3 = out3 + cost0

        cost1 = self.classif1(out1)
        cost2 = self.classif2(out2) + cost1
        cost3 = self.classif3(out3) + cost2

        if self.training:
            cost1 = F.interpolate(
                cost1,
                (self.num_disp, 4 * H, 4 * W),
                mode="trilinear",
                align_corners=False,
            )
            cost2 = F.interpolate(
                cost2,
                (self.num_disp, 4 * H, 4 * W),
                mode="trilinear",
                align_corners=False,
            )

            cost1 = torch.squeeze(cost1, 1)
            pred1 = F.softmax(cost1, dim=1)
            pred1 = self.disp_regression(pred1)

            cost2 = torch.squeeze(cost2, 1)
            pred2 = F.softmax(cost2, dim=1)
            pred2 = self.disp_regression(pred2)

        prob_volume = F.softmax(torch.squeeze(cost3, 1), 1)
        cost3 = F.interpolate(cost3, (self.num_disp, 4 * H, 4 * W), mode="trilinear", align_corners=False)
        cost3 = torch.squeeze(cost3, 1)
        prob_cost3 = F.softmax(cost3, dim=1)

        # For your information: This formulation 'softmax(c)' learned "similarity"
        # while 'softmax(-c)' learned 'matching cost' as mentioned in the paper.
        # However, 'c' or '-c' do not affect the performance because feature-based cost volume provided flexibility.
        pred3 = self.disp_regression(prob_cost3)
        if self.training:
            pred_dict = {
                "pred1": pred1,
                "pred2": pred2,
                "pred3": pred3,
            }
        else:
            pred_dict = {
                "pred3": pred3,
                "prob_volume": prob_volume,
            }

        return pred_dict

    def compute_disp_loss(self, data_batch, pred_dict):
        disp_gt = data_batch["img_disp_l"]
        # Get stereo loss on sim
        # Note in training we do not exclude bg
        mask = (disp_gt < self.max_disp) * (disp_gt > self.min_disp)
        mask.detach()
        loss_disp = 0.0
        for pred_name, loss_weight in zip(["pred1", "pred2", "pred3"], [0.5, 0.7, 1.0]):
            if pred_name in pred_dict:
                loss_disp += loss_weight * F.smooth_l1_loss(pred_dict[pred_name][mask], disp_gt[mask], reduction="mean")

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
            loss_reproj = compute_reproj_loss_patch(
                data_batch["img_pattern_l"],
                data_batch["img_pattern_r"],
                pred_disp_l=pred_dict["pred3"],
                mask=mask,
                ps=patch_size,
            )

            return loss_reproj
        else:
            loss_reproj = 0.0
            for pred_name, loss_weight in zip(["pred1", "pred2", "pred3"], [0.5, 0.7, 1.0]):
                if pred_name in pred_dict:
                    loss_reproj += loss_weight * compute_reproj_loss_patch(
                        data_batch["img_pattern_l"],
                        data_batch["img_pattern_r"],
                        pred_disp_l=pred_dict[pred_name],
                        mask=mask,
                        ps=patch_size,
                    )
            return loss_reproj


if __name__ == "__main__":
    model = PSMNetKPAC(min_disp=12, max_disp=96, num_disp=128, set_zero=False, dilation=3, dilation_list=(1, 5, 9))
    model = model.cuda()
    model.eval()

    data_batch = {
        "img_l": torch.rand(1, 1, 256, 512).cuda(),
        "img_r": torch.rand(1, 1, 256, 512).cuda(),
    }
    pred = model(data_batch)

    for k, v in pred.items():
        print(k, v.shape)
