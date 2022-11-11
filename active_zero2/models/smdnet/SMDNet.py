import math

import torch
import torch.nn as nn

from active_zero2.models.smdnet.losses import *
from active_zero2.models.smdnet.psmnet_3 import PSMNet
from active_zero2.models.smdnet.Regressor import Regressor
from active_zero2.models.smdnet.utils import *
from active_zero2.utils.reprojection import compute_reproj_loss_patch_points


class SMDNet(nn.Module):
    def __init__(
        self,
        output_representation: str,
        maxdisp: int,
        no_sine: bool,
        no_residual: bool,
    ):
        super(SMDNet, self).__init__()
        self.output_representation = output_representation
        self.maxdisp = maxdisp
        self.last_dim = {"standard": 1, "unimodal": 2, "bimodal": 5}

        self.stereo_network = PSMNet(maxdisp)
        self.mlp = Regressor(
            filter_channels=[
                self.stereo_network.init_dim,
                1024,
                512,
                256,
                128,
                self.last_dim[self.output_representation],
            ],
            no_sine=no_sine,
            no_residual=no_residual,
        )

    def get_error(self):
        mask = torch.mul(self.labels > 0, self.labels <= 1.0)

        if self.output_representation == "bimodal":
            loss = bimodal_loss(
                self.mu0[mask],
                self.mu1[mask],
                self.sigma0[mask],
                self.sigma1[mask],
                self.pi0[mask],
                self.pi1[mask],
                self.labels[mask],
                dist="laplacian",
            ).mean()
            errors = {"log_likelihood_loss": loss}

        elif self.output_representation == "unimodal":
            loss = unimodal_loss(self.disp[mask], self.var[mask], self.labels[mask]).mean()
            errors = {"log_likelihood_loss": loss}
        else:
            loss = torch.abs(self.disp[mask] - self.labels[mask]).mean()
            errors = {"l1_loss": loss}

        return errors

    def forward(self, data_batch):
        left = data_batch["img_l"]
        # Get stereo features
        psmnet_pred_dict = self.stereo_network(data_batch)
        feat_list = [psmnet_pred_dict["cost3"], psmnet_pred_dict["refimg_fea"]]
        batch_size = left.shape[0]
        height = left.shape[2]
        width = left.shape[3]
        pred_dict = {}

        # Coordinated between [-1, 1]
        if "img_points" in data_batch:  # train and validation
            points = data_batch["img_points"]
            u = scale_coords(points[:, 0:1, :], width)
            v = scale_coords(points[:, 1:2, :], height)
            uv = torch.cat([u, v], 1)
            # Interpolate features
            for i, im_feat in enumerate(feat_list):
                interp_feat = interpolate(im_feat, uv)
                features = interp_feat if not i else torch.cat([features, interp_feat], 1)
        else:  # test
            features = torch.cat([feat_list[0], feat_list[1]], 1)
            features = F.interpolate(features, (height, width), mode="bilinear", align_corners=True)
            feature_channel = features.shape[1]
            features = features.view(batch_size, feature_channel, -1)

        pred = self.mlp(features)
        activation = nn.Sigmoid()

        # Bimodal output representation
        if self.output_representation == "bimodal":
            eps = 1e-2  # 1e-3 in case of gaussian distribution
            mu0 = activation(torch.unsqueeze(pred[:, 0, :], 1))
            mu1 = activation(torch.unsqueeze(pred[:, 1, :], 1))

            sigma0 = torch.clamp(activation(torch.unsqueeze(pred[:, 2, :], 1)), eps, 1.0)
            sigma1 = torch.clamp(activation(torch.unsqueeze(pred[:, 3, :], 1)), eps, 1.0)

            pi0 = activation(torch.unsqueeze(pred[:, 4, :], 1))
            pi1 = 1.0 - pi0

            # Mode with the highest density value as final prediction
            mask = (pi0 / sigma0 > pi1 / sigma1).float()
            disp = mu0 * mask + mu1 * (1.0 - mask)  # winner takes all

            # Rescale outputs
            pred_dict.update(
                {
                    "point_disp": disp * self.maxdisp,
                    "mu0": mu0 * self.maxdisp,
                    "mu1": mu1 * self.maxdisp,
                    "sigma0": sigma0,
                    "sigma1": sigma1,
                    "pi0": pi0,
                    "pi1": pi1,
                }
            )

        # Unimodal output representation
        elif self.output_representation == "unimodal":
            disp = activation(torch.unsqueeze(pred[:, 0, :], 1))
            var = activation(torch.unsqueeze(pred[:, 1, :], 1))
            pred_dict.update({"point_disp": disp * self.maxdisp, "var": var})

        # Standard regression
        else:
            disp = activation(pred)
            pred_dict.update({"point_disp": disp * self.maxdisp})

        if "img_points" not in data_batch:  # test
            pred_dict.update({"pred_disp": pred_dict["point_disp"].view(batch_size, 1, height, width)})

        return pred_dict

    def compute_disp_loss(self, data_batch, pred_dict):
        gt_disp = data_batch["img_labels"]
        gt_disp /= self.maxdisp
        mask = torch.mul(gt_disp > 0, gt_disp <= 1.0)

        if self.output_representation == "bimodal":
            loss = bimodal_loss(
                pred_dict["mu0"][mask] / self.maxdisp,
                pred_dict["mu1"][mask] / self.maxdisp,
                pred_dict["sigma0"][mask],
                pred_dict["sigma1"][mask],
                pred_dict["pi0"][mask],
                pred_dict["pi1"][mask],
                gt_disp[mask],
                dist="laplacian",
            ).mean()

        elif self.output_representation == "unimodal":
            loss = unimodal_loss(
                pred_dict["point_disp"][mask] / self.maxdisp, pred_dict["var"][mask], gt_disp[mask]
            ).mean()
        else:
            loss = torch.abs(pred_dict["point_disp"][mask] / self.maxdisp - gt_disp[mask]).mean()

        return loss

    def compute_reproj_loss(self, data_batch, pred_dict, use_mask: bool, patch_size: int, only_last_pred: bool):
        point_disp = pred_dict["point_disp"]
        height, width = data_batch["img_pattern_l"].shape[2:]

        loss = compute_reproj_loss_patch_points(
            data_batch["img_pattern_l"],
            data_batch["img_pattern_r"],
            points=data_batch["img_points"],
            pred_disp_points=point_disp,
            height=height,
            width=width,
            ps=patch_size,
        )

        return loss


if __name__ == "__main__":
    model = SMDNet(output_representation="bimodal", maxdisp=192, no_sine=False, no_residual=False)
    model = model.cuda()
    model.eval()

    data_batch = {
        "img_l": torch.rand(1, 1, 256, 512).cuda(),
        "img_r": torch.rand(1, 1, 256, 512).cuda(),
    }
    pred = model(data_batch)

    for k, v in pred.items():
        print(k, v.shape)
