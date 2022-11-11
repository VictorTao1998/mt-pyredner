import math
from typing import Tuple

import torch

from active_zero2.models.psmnet_dilation_adv4.psmnet_submodule_3 import *
from active_zero2.models.psmnet_dilation_adv4.utils import DispGrad
from active_zero2.utils.reprojection import compute_reproj_loss_patch


class hourglass(nn.Module):
    def __init__(self, inplanes, dilation):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(inplanes, inplanes * 2, kernel_size=3, stride=2, padding=dilation, dilation=dilation),
            nn.BatchNorm3d(inplanes * 2),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, padding=dilation, dilation=dilation),
            nn.BatchNorm3d(inplanes * 2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=2, padding=dilation, dilation=dilation),
            nn.BatchNorm3d(inplanes * 2),
            nn.ReLU(inplace=True),
        )

        self.conv4 = nn.Sequential(
            nn.Conv3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, padding=dilation, dilation=dilation),
            nn.BatchNorm3d(inplanes * 2, inplanes * 2),
            nn.ReLU(inplace=True),
        )

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
            pre = F.relu(pre + postqu, inplace=True)
        else:
            pre = F.relu(pre, inplace=True)

        out = self.conv3(pre)
        out = self.conv4(out)

        if presqu is not None:
            post = F.relu(self.conv5(out) + presqu, inplace=True)
        else:
            post = F.relu(self.conv5(out) + pre, inplace=True)

        out = self.conv6(post)
        return out, pre, post


def conv3d_half(n_ch_in, n_ch_out, bias):
    return nn.Conv3d(n_ch_in, n_ch_out, 4, stride=2, padding=1, dilation=1, groups=1, bias=bias)


def conv3d_minus3(n_ch_in, n_ch_out, bias):
    return nn.Conv3d(n_ch_in, n_ch_out, 4, stride=1, padding=0, dilation=1, groups=1, bias=bias)


# ref: https://github.com/xiumingzhang/GenRe-ShapeHD/blob/ee42add2707de509b5914ab444ae91b832f75981/networks/networks.py#L107
class Discriminator(nn.Module):
    def __init__(self, in_channels, base_channels, bias=False):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            conv3d_half(in_channels, base_channels, bias),
            nn.LeakyReLU(0.2, True),
            conv3d_half(base_channels, base_channels * 2, bias),
            nn.LeakyReLU(0.2, True),
            conv3d_half(base_channels * 2, base_channels * 4, bias),
            nn.LeakyReLU(0.2, True),
            conv3d_minus3(base_channels * 4, 1, bias),
        )

    def forward(self, x):
        """

        :param x: prob cost volume (w/ disp encoding), [B, C, D, H, W]
        :return:
        """
        y = self.net(x)
        return y.view(-1, 1).squeeze(1)


class Discriminator3(nn.Module):
    def __init__(self, in_channels, base_channels, bias=False):
        super(Discriminator3, self).__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, 4, stride=2, padding=0, bias=bias),
            nn.InstanceNorm3d(base_channels),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(base_channels, base_channels * 2, 4, stride=2, padding=0, bias=bias),
            nn.InstanceNorm3d(base_channels * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(base_channels * 2, 1, 4, stride=1, padding=0, bias=bias),
        )

    def forward(self, x):
        """

        :param x: prob cost volume (w/ disp encoding), [B, C, D, H, W]
        :return:
        """
        y = self.net(x)
        return y.view(-1, 1).squeeze(1)


class PSMNetDilation(nn.Module):
    def __init__(self, min_disp: float, max_disp: float, num_disp: int, set_zero: bool, dilation: int):
        super(PSMNetDilation, self).__init__()
        self.min_disp = min_disp
        self.max_disp = max_disp
        self.num_disp = num_disp
        self.dilation = dilation
        assert num_disp % 4 == 0, "Num_disp % 4 should be 0"
        self.num_disp_4 = num_disp // 4
        self.set_zero = set_zero  # set zero for invalid reference image cost volume

        self.disp_list = torch.linspace(min_disp, max_disp, num_disp)
        self.disp_list_4 = torch.linspace(min_disp, max_disp, self.num_disp_4) / 4
        self.disp_regression = DisparityRegression(min_disp, max_disp, num_disp)

        self.feature_extraction = FeatureExtraction()

        self.dres0 = nn.Sequential(
            convbn_3d(64, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            convbn_3d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        self.dres1 = nn.Sequential(
            convbn_3d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            convbn_3d(32, 32, 3, 1, 1),
        )

        self.dres2 = hourglass(32, dilation)
        self.dres3 = hourglass(32, dilation)
        self.dres4 = hourglass(32, dilation)

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
        self.classif3 = nn.Sequential(
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
                "prob_volume": prob_volume,
            }
        else:
            pred_dict = {
                "pred3": pred3,
                "prob_volume": prob_volume,
            }

        return pred_dict


class PSMNetADV4(nn.Module):
    def __init__(
        self,
        min_disp: float,
        max_disp: float,
        num_disp: int,
        set_zero: bool,
        dilation: int,
        epsilon: float,
        grad_threshold: float,
        d_channels: int,
        disp_encoding: Tuple[float],
        wgangp_norm: float,
        wgangp_lambda: float,
        d_type: str,
        disp_grad_norm: str,
    ):
        """

        :param min_disp:
        :param max_disp:
        :param num_disp:
        :param set_zero:
        :param dilation:
        :param epsilon: for grad exp weight
        # Adv
        :param d_channels: Discriminator base channels
        :param disp_encoding:
        :param wgangp_norm: WGANGP gradient penalty norm
        :param wgangp_lambda: WGANGP gradient penalty coefficient
        """
        super(PSMNetADV4, self).__init__()
        self.min_disp = min_disp
        self.max_disp = max_disp
        self.num_disp = num_disp
        self.dilation = dilation
        self.epsilon = epsilon
        self.grad_threshold = grad_threshold
        assert num_disp % 4 == 0, "Num_disp % 4 should be 0"
        self.num_disp_4 = num_disp // 4
        self.set_zero = set_zero  # set zero for invalid reference image cost volume

        self.disp_list = torch.linspace(min_disp, max_disp, num_disp)
        self.T = (max_disp - min_disp) / (self.num_disp_4 - 1) / 4
        self.disp_list_4 = torch.linspace(min_disp, max_disp, self.num_disp_4) / 4
        self.psmnet = PSMNetDilation(min_disp, max_disp, num_disp, set_zero, dilation)

        # disp encoding
        disp_encoded = []
        for t in disp_encoding:
            disp_encoded.append(torch.sin(self.disp_list_4 * t))
            disp_encoded.append(torch.cos(self.disp_list_4 * t))
        if disp_encoded:
            self.disp_encoded = torch.stack(disp_encoded)  # (2*disp_encoding, D)
            self.disp_channels_half = len(disp_encoding)
            in_channels = 3 * len(disp_encoding)
        else:
            self.disp_encoded = None
            in_channels = 1

        assert d_type in ("D", "D3")
        if d_type == "D":
            self.D = Discriminator(in_channels, d_channels, bias=False)
        elif d_type == "D3":
            self.D = Discriminator3(in_channels, d_channels, bias=False)
        self.wgangp_norm = wgangp_norm
        self.wgangp_lambda = wgangp_lambda

        self.disp_grad = DispGrad(grad_threshold)
        self.disp_grad_norm = disp_grad_norm
        assert self.disp_grad_norm in ("L1", "L2")

    def forward(self, data_batch):
        pred_dict = self.psmnet(data_batch)
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

    def calc_grad_penalty(self, real_prob_volume: torch.Tensor, fake_prob_volume: torch.Tensor):
        alpha = torch.rand(real_prob_volume.shape[0], 1)
        alpha = (
            alpha.expand(real_prob_volume.shape[0], real_prob_volume.nelement() // real_prob_volume.shape[0])
            .contiguous()
            .view(*real_prob_volume.shape)
            .to(real_prob_volume.device)
        )
        inter = alpha * real_prob_volume + (1 - alpha) * fake_prob_volume
        inter.requires_grad = True
        err_d_inter = self.D(inter)
        grads = torch.autograd.grad(
            outputs=err_d_inter,
            inputs=inter,
            grad_outputs=torch.ones(err_d_inter.size()).cuda(),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        grad_penalty = (((grads + 1e-16).norm(2, dim=1) - self.wgangp_norm) ** 2).mean() * self.wgangp_lambda
        return grad_penalty

    def D_backward(self, data_batch, pred_dict):
        if "gt_prob_volume" in data_batch:
            gt_prob_volume = data_batch["gt_prob_volume"]
            one = torch.tensor(1, dtype=torch.float, device=gt_prob_volume.device)
            mone = one * -1
        else:
            # generate GT prob volume
            disp_gt = data_batch["img_disp_l"]
            one = torch.tensor(1, dtype=torch.float, device=disp_gt.device)
            mone = one * -1
            disp_gt = F.interpolate(disp_gt, scale_factor=0.25, mode="nearest")
            mask = (disp_gt > self.min_disp) * (disp_gt < self.max_disp)
            mask = mask.squeeze(1)
            disp_gt /= 4
            disp_gt_norm = (disp_gt - self.min_disp / 4) / self.T
            disp_gt_norm = disp_gt_norm.squeeze(1)

            low = torch.floor(disp_gt_norm)
            up = low + 1
            low_value = up - disp_gt_norm
            up_value = disp_gt_norm - low
            low *= mask
            up *= mask
            low_value *= mask
            up_value *= mask

            low_volume = F.one_hot(low.long(), num_classes=self.num_disp_4).permute(0, 3, 1, 2)
            up_volume = F.one_hot(up.long(), num_classes=self.num_disp_4).permute(0, 3, 1, 2)
            low_volume = low_value.unsqueeze(1) * low_volume
            up_volume = up_value.unsqueeze(1) * up_volume

            gt_prob_volume = low_volume + up_volume
            gt_prob_volume = gt_prob_volume.unsqueeze(1)
            if self.disp_encoded is not None:
                batch_size, _, D, H, W = gt_prob_volume.shape
                disp_encoded = (
                    self.disp_encoded.unsqueeze(0)
                    .unsqueeze(-1)
                    .unsqueeze(-1)
                    .expand(batch_size, 2 * self.disp_channels_half, D, H, W)
                    .to(gt_prob_volume.device)
                )
                gt_prob_volume = gt_prob_volume.expand(batch_size, self.disp_channels_half, D, H, W)
                gt_prob_volume = torch.cat([gt_prob_volume, disp_encoded], dim=1)

        err_d_real = self.D(gt_prob_volume).mean()
        err_d_real.backward(mone)
        pred_prob_volume = pred_dict["prob_volume"].unsqueeze(1)
        if self.disp_encoded is not None:
            batch_size, _, D, H, W = pred_prob_volume.shape
            if "gt_prob_volume" in data_batch:
                disp_encoded = (
                    self.disp_encoded.unsqueeze(0)
                    .unsqueeze(-1)
                    .unsqueeze(-1)
                    .expand(batch_size, 2 * self.disp_channels_half, D, H, W)
                    .to(pred_prob_volume.device)
                )
            pred_prob_volume = pred_prob_volume.expand(batch_size, self.disp_channels_half, D, H, W)
            pred_prob_volume = torch.cat([pred_prob_volume, disp_encoded], dim=1)

        err_d_fake = self.D(pred_prob_volume).mean()
        err_d_fake.backward(one)

        return_dict = {
            "gt_prob_volume": gt_prob_volume,
            "pred_prob_volume": pred_prob_volume,
            "err_d_real": -err_d_real.item(),
            "err_d_fake": err_d_fake.item(),
        }
        # gradient penalty
        if self.wgangp_lambda > 0:
            grad_penalty = self.calc_grad_penalty(gt_prob_volume, pred_prob_volume)
            grad_penalty.backward()
            return_dict["err_d_gp"] = grad_penalty.item()
        return return_dict

    def G_backward(self, data_batch, pred_dict, adv_weight):
        pred_prob_volume = pred_dict["prob_volume"].unsqueeze(1)
        mone = torch.tensor(-1, dtype=torch.float, device=pred_prob_volume.device)
        if self.disp_encoded is not None:
            batch_size, _, D, H, W = pred_prob_volume.shape
            disp_encoded = (
                self.disp_encoded.unsqueeze(0)
                .unsqueeze(-1)
                .unsqueeze(-1)
                .expand(batch_size, 2 * self.disp_channels_half, D, H, W)
                .to(pred_prob_volume.device)
            )
            pred_prob_volume = pred_prob_volume.expand(batch_size, self.disp_channels_half, D, H, W)
            pred_prob_volume = torch.cat([pred_prob_volume, disp_encoded], dim=1)
        psmnet_loss = self.D(pred_prob_volume)
        psmnet_loss = psmnet_loss.mean() * adv_weight
        psmnet_loss.backward(mone)
        return -psmnet_loss

    def compute_grad_loss(self, data_batch, pred_dict):
        disp_pred = pred_dict["pred3"]
        disp_grad_pred = self.disp_grad(disp_pred)

        if "img_disp_l" in data_batch:
            disp_gt = data_batch["img_disp_l"]
            disp_grad_gt = self.disp_grad(disp_gt)
            if self.disp_grad_norm == "L1":
                grad_diff = torch.abs(disp_grad_pred - disp_grad_gt)
            elif self.disp_grad_norm == "L2":
                grad_diff = (disp_grad_pred - disp_grad_gt) ** 2
            grad_diff = grad_diff * torch.exp(-torch.abs(disp_grad_gt) * self.epsilon)
            mask = (disp_gt < self.max_disp) * (disp_gt > self.min_disp)
            mask.detach()
            loss = torch.mean(grad_diff * mask)
        else:
            if self.disp_grad_norm == "L1":
                loss = torch.mean(torch.abs(disp_grad_pred))
            elif self.disp_grad_norm == "L2":
                loss = torch.mean(disp_grad_pred**2)
        return loss


if __name__ == "__main__":
    model = PSMNetADV4(
        min_disp=12,
        max_disp=96,
        num_disp=128,
        set_zero=False,
        dilation=3,
        d_channels=16,
        disp_encoding=(),  # (0.5, 2),
        wgangp_norm=1,
        wgangp_lambda=10,
    )
    model = model.cuda()
    model.train()
    for p in model.psmnet.parameters():
        p.requires_grad = False
    for p in model.D.parameters():
        p.requires_grad = True

    data_batch = {
        "img_l": torch.rand(1, 1, 256, 512).cuda(),
        "img_r": torch.rand(1, 1, 256, 512).cuda(),
        "img_disp_l": torch.rand(1, 1, 256, 512).cuda() * 64 + 12,
    }
    pred = model(data_batch)

    for k, v in pred.items():
        print(k, v.shape)

    D_dict = model.D_backward(data_batch, pred)
    for k, v in D_dict.items():
        if isinstance(v, torch.Tensor):
            print(k, v.shape)
        else:
            print(k, v)

    adv_loss = model.compute_adv_loss(data_batch, pred)
    print("adv: ", adv_loss)
