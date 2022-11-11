import math
from typing import Tuple

import torch

from active_zero2.models.psmnet_dilation_df.DFNet import DFNet
from active_zero2.models.psmnet_dilation_df.psmnet_submodule_3 import *
from active_zero2.utils.confidence import compute_confidence
from active_zero2.utils.disp_grad import DispGrad
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

        self.feature_extraction = FeatureExtraction(1)

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
                "prob_cost3": prob_cost3,
            }
        else:
            pred_dict = {
                "pred3": pred3,
                "prob_cost3": prob_cost3,
            }

        return pred_dict


class PSMNetDilationDF(nn.Module):
    def __init__(
        self,
        min_disp: float,
        max_disp: float,
        num_disp: int,
        set_zero: bool,
        dilation: int,
        epsilon: float,
        grad_threshold: float,
        df_channels: int,
        use_image: bool,
        use_off: bool,
        use_edge: bool,
        use_full_volume: bool,
        use_conf_map: bool,
        mask_to_zero: bool,
        df_res: bool,
        mix: bool,
        conf_range: Tuple[float, float],
        sim_disp_weight: float,
    ):
        """

        :param min_disp:
        :param max_disp:
        :param num_disp:
        :param set_zero:
        :param dilation:
        :param epsilon:
        :param grad_threshold:
        # dfnet
        :param df_channels:
        :param use_off:
        :param use_edge:
        :param use_full_volume:
        :param use_conf_map:
        :param mask_to_zero: set value of uncertain area to zero
        :param df_res: predict residual or absolute value
        :param mix: whether to mix stereo pred and dfnet pred
        """
        super(PSMNetDilationDF, self).__init__()
        self.min_disp = min_disp
        self.max_disp = max_disp
        self.num_disp = num_disp
        self.dilation = dilation
        self.epsilon = epsilon
        self.grad_threshold = grad_threshold
        self.use_image = use_image
        self.use_off = use_off
        self.use_edge = use_edge
        self.use_full_volume = use_full_volume
        self.use_conf_map = use_conf_map
        self.mask_to_zero = mask_to_zero
        self.df_res = df_res
        self.mix = mix
        self.conf_range = conf_range
        self.sim_disp_weight = sim_disp_weight

        # stereo network
        self.psmnet = PSMNetDilation(min_disp, max_disp, num_disp, set_zero, dilation)

        df_in_channels = 1 + 1  # disp + mask
        if use_image:
            df_in_channels += 1
        if use_off:
            df_in_channels += 1
        if use_edge:
            df_in_channels += 1
        if use_full_volume:
            df_in_channels += num_disp
        elif use_conf_map:
            df_in_channels += 4
        self.df_in_channels = df_in_channels
        self.DFNet = DFNet(df_in_channels, df_channels, residual=df_res)

        self.disp_grad = DispGrad(grad_threshold)

    def forward(self, data_batch, df=True):
        pred_dict = self.psmnet(data_batch)
        if df:
            pred3 = pred_dict["pred3"]
            prob_cost3 = pred_dict["prob_cost3"]
            pred_norm = (pred3 - self.min_disp) / (self.max_disp - self.min_disp)
            conf_map = compute_confidence(pred_norm, prob_cost3)
            if self.training:
                conf_threshold = torch.rand(1).item() * (self.conf_range[1] - self.conf_range[0]) + self.conf_range[0]
            else:
                conf_threshold = (self.conf_range[0] + self.conf_range[1]) / 2
            conf_mask = (conf_map[:, 1:2] > conf_threshold).float()
            if self.mask_to_zero:
                pred_norm = pred_norm * conf_mask
            df_inputs = [pred_norm, conf_mask]
            if self.use_image:
                df_inputs.append(data_batch["img_l"])
            if self.use_off:
                df_inputs.append(data_batch["img_off_l"])
            if self.use_edge:
                df_inputs.append(data_batch["img_edge_l"])
            if self.use_full_volume:
                df_inputs.append(prob_cost3)
            elif self.use_conf_map:
                df_inputs.append(conf_map)

            df_inputs = torch.cat(df_inputs, dim=1)
            df_inputs = df_inputs.detach()
            df_pred = self.DFNet(df_inputs)
            if self.df_res:
                df_pred = df_pred + pred_norm
                df_pred = torch.clip(df_pred, 0, 1)
            if self.mix:
                df_pred = df_pred * (1 - conf_mask) + pred_norm * conf_mask

            df_pred = df_pred * (self.max_disp - self.min_disp) + self.min_disp
            pred_dict["df_pred"] = df_pred
            pred_dict["conf_map"] = conf_map
            pred_dict["conf_mask"] = conf_mask
        else:
            df_inputs = torch.zeros_like(data_batch["img_l"]).expand(-1, self.df_in_channels, -1, -1)
            df_pred = self.DFNet(df_inputs)
            pred_dict["df_pred"] = df_pred

        return pred_dict

    def compute_disp_loss(self, data_batch, pred_dict, df=False):
        disp_gt = data_batch["img_disp_l"]
        # Get stereo loss on sim
        # Note in training we do not exclude bg
        mask = (disp_gt < self.max_disp) * (disp_gt > self.min_disp)
        mask.detach()
        loss_disp = {}
        for pred_name, loss_weight in zip(["pred1", "pred2", "pred3"], [0.5, 0.7, 1.0]):
            if pred_name in pred_dict:
                loss_disp[pred_name] = loss_weight * F.smooth_l1_loss(
                    pred_dict[pred_name][mask], disp_gt[mask], reduction="mean"
                )
        loss_df = self.sim_disp_weight * F.smooth_l1_loss(pred_dict["df_pred"][mask], disp_gt[mask], reduction="mean")
        if not df:
            loss_df *= 0.0
        loss_disp["df"] = loss_df

        return loss_disp

    def compute_reproj_loss(
        self, data_batch, pred_dict, use_mask: bool, patch_size: int, only_last_pred: bool, df=True
    ):
        if use_mask:
            disp_gt = data_batch["img_disp_l"]
            # Get stereo loss on sim
            # Note in training we do not exclude bg
            mask = (disp_gt < self.max_disp) * (disp_gt > self.min_disp)
            mask.detach()
        else:
            mask = None
        loss_reproj = {}
        loss_reproj["pred3"] = compute_reproj_loss_patch(
            data_batch["img_pattern_l"],
            data_batch["img_pattern_r"],
            pred_disp_l=pred_dict["pred3"],
            mask=mask,
            ps=patch_size,
        )
        loss_reproj["df_pred"] = compute_reproj_loss_patch(
            data_batch["img_pattern_l"],
            data_batch["img_pattern_r"],
            pred_disp_l=pred_dict["df_pred"],
            mask=mask,
            ps=patch_size,
        )
        if not df:
            loss_reproj["df_pred"] *= 0

        return loss_reproj

    def compute_grad_loss(self, data_batch, pred_dict, df=True):
        disp_pred = pred_dict["pred3"]
        disp_grad_pred = self.disp_grad(disp_pred)
        edge = data_batch["img_edge_l"]
        loss_grad = {}
        loss_grad["pred3"] = torch.mean(torch.abs(disp_grad_pred) * torch.exp(-edge * self.epsilon))

        disp_pred = pred_dict["df_pred"]
        disp_grad_pred = self.disp_grad(disp_pred)
        loss_grad["df_pred"] = torch.mean(torch.abs(disp_grad_pred) * torch.exp(-edge * self.epsilon))
        if not df:
            loss_grad["df_pred"] *= 0
        return loss_grad


if __name__ == "__main__":
    model = PSMNetDilationDF(
        min_disp=12,
        max_disp=96,
        num_disp=128,
        set_zero=False,
        dilation=3,
        use_off=False,
        use_full_volume=True,
        use_conf_map=False,
        df_channels=16,
        df_disp_weight=1.0,
    )
    model = model.cuda()
    model.eval()

    data_batch = {
        "img_l": torch.rand(1, 1, 256, 512).cuda(),
        "img_r": torch.rand(1, 1, 256, 512).cuda(),
    }
    pred = model(data_batch)

    for k, v in pred.items():
        print(k, v.shape)
