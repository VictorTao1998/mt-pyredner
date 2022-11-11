import math
import tabnanny

import torch

from active_zero2.models.psmnet_inpainting.loss import AdversarialLoss
from active_zero2.models.psmnet_inpainting.networks import Discriminator, InpaintGenerator
from active_zero2.models.psmnet_inpainting.psmnet_submodule_3 import *
from active_zero2.models.psmnet_inpainting.utils import DispGrad
from active_zero2.utils.confidence import compute_confidence
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
                "prob_cost3": prob_cost3,
            }
        else:
            pred_dict = {
                "pred3": pred3,
                "prob_cost3": prob_cost3,
            }

        return pred_dict


class PSMNetInpainting(nn.Module):
    def __init__(
        self,
        min_disp: float,
        max_disp: float,
        num_disp: int,
        set_zero: bool,
        dilation: int,
        epsilon: float,
        grad_threshold: float,
        sub_avg_size: int,
        disp_grad_norm: str,
        use_off: bool,
        use_edge: bool,
        conf_range,
    ):
        """
        :param min_disp:
        :param max_disp:
        :param num_disp:
        :param set_zero:
        :param dilation:
        :param epsilon: for grad exp weight
        :param sub_avg_size: subtract the local average in discriminator
        :param disp_grad_norm: L1 or L2
        """
        super(PSMNetInpainting, self).__init__()
        self.min_disp = min_disp
        self.max_disp = max_disp
        self.num_disp = num_disp
        self.dilation = dilation
        self.epsilon = epsilon
        self.grad_threshold = grad_threshold
        # stereo network
        self.psmnet = PSMNetDilation(min_disp, max_disp, num_disp, set_zero, dilation)

        # inpainting network
        self.use_off = use_off
        self.use_edge = use_edge
        in_channels = 3  # disp + mask + IR
        if use_off:
            in_channels += 1
        if use_edge:
            in_channels += 1
        self.inpaint_generator = InpaintGenerator(in_channels)
        self.D = Discriminator(in_channels=1, sub_avg_size=sub_avg_size)
        self.adv_loss = AdversarialLoss()

        self.conf_range = conf_range

        self.disp_grad_norm = disp_grad_norm
        assert self.disp_grad_norm in ("L1", "L2")
        self.disp_grad = DispGrad(grad_threshold)

    def forward(self, data_batch, real=True):
        pred_dict = self.psmnet(data_batch)
        if real:
            pred3 = pred_dict["pred3"]
            prob_cost3 = pred_dict["prob_cost3"]
            pred_norm = (pred3 - self.min_disp) / (self.max_disp - self.min_disp)
            conf_map = compute_confidence(pred_norm, prob_cost3)
            if self.training:
                conf_threshold = torch.rand(1).item() * (self.conf_range[1] - self.conf_range[0]) + self.conf_range[0]
            else:
                conf_threshold = (self.conf_range[0] + self.conf_range[1]) / 2
            valid_mask = (conf_map[:, 1:2] > conf_threshold).float()
            pred_norm = pred_norm * valid_mask
            inpaint_input = [pred_norm, valid_mask, data_batch["img_l"]]
            if self.use_off:
                inpaint_input.append(data_batch["img_off_l"])
            if self.use_edge:
                inpaint_input.append(data_batch["img_edge_l"])
            inpaint_input = torch.cat(inpaint_input, dim=1)
            inpaint_input = inpaint_input.detach()

            inpaint_output = self.inpaint_generator(inpaint_input)
            inpaint_output = inpaint_output * (self.max_disp - self.min_disp) + self.min_disp
            pred_dict["inpaint"] = inpaint_output

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
            for pred_name, loss_weight in zip(["pred3", "inpaint"], [1.0, 1.0]):
                if pred_name in pred_dict:
                    loss_reproj += loss_weight * compute_reproj_loss_patch(
                        data_batch["img_pattern_l"],
                        data_batch["img_pattern_r"],
                        pred_disp_l=pred_dict[pred_name],
                        mask=mask,
                        ps=patch_size,
                    )
            return loss_reproj

    def compute_grad_loss(self, data_batch, pred_dict):
        disp_pred = pred_dict["pred3"]
        disp_grad_pred = self.disp_grad(disp_pred)
        edge = data_batch["img_edge_l"]
        if self.disp_grad_norm == "L1":
            loss = torch.mean(torch.abs(disp_grad_pred) * torch.exp(-edge * self.epsilon))
        elif self.disp_grad_norm == "L2":
            loss = torch.mean(disp_grad_pred**2 * torch.exp(-edge * self.epsilon))

        if "inpaint" in pred_dict:
            disp_pred = pred_dict["inpaint"]
            disp_grad_pred = self.disp_grad(disp_pred)
            if self.disp_grad_norm == "L1":
                loss += torch.mean(torch.abs(disp_grad_pred) * torch.exp(-edge * self.epsilon))
            elif self.disp_grad_norm == "L2":
                loss += torch.mean(disp_grad_pred**2 * torch.exp(-edge * self.epsilon))
        return loss

    def compute_D_loss(self, data_batch, pred_dict):
        real_disp = data_batch["real_disp"]
        real_disp = torch.clip((real_disp - self.min_disp) / (self.max_disp - self.min_disp), 0, 1)
        dis_real = self.D(real_disp)

        fake_disp = pred_dict["inpaint"].detach()
        fake_disp = (fake_disp - self.min_disp) / (self.max_disp - self.min_disp)
        dis_fake = self.D(fake_disp)

        dis_real_loss = self.adv_loss(dis_real, True)
        dis_fake_loss = self.adv_loss(dis_fake, False)

        return dis_real_loss, dis_fake_loss

    def compute_G_loss(self, data_batch, pred_dict):
        fake_disp = pred_dict["inpaint"]
        fake_disp = (fake_disp - self.min_disp) / (self.max_disp - self.min_disp)
        gen_fake = self.D(fake_disp)
        gen_loss = self.adv_loss(gen_fake, True)
        return gen_loss

    def test_D(self, data_batch, pred_dict):
        disp_gt = data_batch["img_disp_l"]
        disp_gt = torch.clip((disp_gt - self.min_disp) / (self.max_disp - self.min_disp), 0, 1)
        pred_dict["d_gt"] = self.D(disp_gt)

        disp_pred = pred_dict["pred3"]
        disp_pred = torch.clip((disp_pred - self.min_disp) / (self.max_disp - self.min_disp), 0, 1)
        pred_dict["d_pred"] = self.D(disp_pred)


if __name__ == "__main__":
    model = PSMNetInpainting(
        min_disp=12,
        max_disp=96,
        num_disp=128,
        set_zero=False,
        dilation=3,
        epsilon=1.0,
        d_channels=16,
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
        "real_disp": torch.rand(1, 1, 256, 512).cuda() * 64 + 12,
    }
    pred = model(data_batch)

    for k, v in pred.items():
        print(k, v.shape)

    grad_loss = model.compute_grad_loss(data_batch, pred)
    print("grad loss: ", grad_loss)

    D_dict = model.D_backward(data_batch, pred)
    for k, v in D_dict.items():
        if isinstance(v, torch.Tensor):
            print(k, v.shape)
        else:
            print(k, v)

    adv_loss = model.compute_adv_loss(data_batch, pred)
    print("adv: ", adv_loss)
