import math

import torch

from active_zero2.models.psmnet_clear_grasp.psmnet_submodule_3 import *
from active_zero2.models.psmnet_clear_grasp.deeplab import DeepLab
from active_zero2.utils.confidence import compute_confidence
from active_zero2.utils.reprojection import compute_reproj_loss_patch
from active_zero2.utils.disp_grad import DispGrad


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


class PSMNetClearGrasp(nn.Module):
    def __init__(
        self,
        min_disp: float,
        max_disp: float,
        num_disp: int,
        set_zero: bool,
        dilation: int,
        epsilon: float,
        grad_threshold: float,
        use_off: bool,
        edge_weight: float,
    ):
        super(PSMNetClearGrasp, self).__init__()
        self.min_disp = min_disp
        self.max_disp = max_disp
        self.num_disp = num_disp
        self.dilation = dilation
        self.epsilon = epsilon
        self.use_off = use_off
        self.edge_weight = edge_weight
        assert num_disp % 4 == 0, "Num_disp % 4 should be 0"
        self.num_disp_4 = num_disp // 4
        self.set_zero = set_zero  # set zero for invalid reference image cost volume

        self.deeplab_normal = DeepLab(num_classes=3)
        self.deeplab_edge = DeepLab(num_classes=2)

        self.edge_loss = nn.CrossEntropyLoss(weight=torch.Tensor([1, edge_weight]), reduction="none")

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

        self.disp_grad = DispGrad(grad_threshold)

    def forward(self, data_batch):
        # normal prediction
        if self.use_off:
            normal3 = self.deeplab_normal(data_batch["img_off_l"])
        else:
            normal3 = self.deeplab_normal(data_batch["img_l"])
        normal3 = normal3 / torch.linalg.norm(normal3, dim=1, keepdim=True)

        # edge prediction
        if self.use_off:
            edge = self.deeplab_edge(data_batch["img_off_l"])
        else:
            edge = self.deeplab_edge(data_batch["img_l"])

        if self.training:
            pred_dict = {
                "edge": edge,
                "normal3": normal3,
            }
        else:
            pred_dict = {
                "edge": edge,
                "normal3": normal3,
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

    def compute_edge_loss(
        self,
        data_batch,
        pred_dict,
    ):
        disp_gt = data_batch["img_disp_l"]
        mask = (disp_gt < self.max_disp) * (disp_gt > self.min_disp)
        mask.detach()
        edge_pred = pred_dict["edge"]
        edge_gt = data_batch["img_disp_edge_l"]
        edge_loss = self.edge_loss(edge_pred, edge_gt)
        edge_loss = (edge_loss * mask).sum() / (mask.sum() + 1e-7)

        with torch.no_grad():
            mask = mask.squeeze(1)
            edge_pred = torch.argmax(edge_pred, dim=1)
            tp = (edge_pred == 1) * (edge_gt == 1) * mask
            prec = tp.sum() / (((edge_pred == 1) * mask).sum() + 1e-7)
            recall = tp.sum() / ((edge_gt[mask] == 1).sum() + 1e-7)

        return edge_loss, prec, recall

    def compute_grad_loss(self, data_batch, pred_dict):
        disp_pred = pred_dict["pred3"]
        disp_grad_pred = self.disp_grad(disp_pred)

        if "img_disp_l" in data_batch:
            disp_gt = data_batch["img_disp_l"]
            disp_grad_gt = self.disp_grad(disp_gt)
            grad_diff = torch.abs(disp_grad_pred - disp_grad_gt)
            grad_diff = grad_diff * torch.exp(-torch.abs(disp_grad_gt) * self.epsilon)
            mask = (disp_gt < self.max_disp) * (disp_gt > self.min_disp)
            mask.detach()
            loss = torch.mean(grad_diff * mask)
        else:
            loss = torch.mean(torch.abs(disp_grad_pred))
        return loss

    def compute_normal_loss(self, data_batch, pred_dict):
        normal_gt = data_batch["img_normal_l"]
        normal_pred = pred_dict["normal3"]
        cos = F.cosine_similarity(normal_gt, normal_pred, dim=1, eps=1e-6)
        loss_cos = 1.0 - cos
        if "img_disp_l" in data_batch:
            disp_gt = data_batch["img_disp_l"]
            mask = (disp_gt < self.max_disp) * (disp_gt > self.min_disp)
            mask = mask.squeeze(1)
            mask.detach()
        else:
            mask = torch.ones_like(loss_cos)

        if "img_normal_weight" in data_batch:
            img_normal_weight = data_batch["img_normal_weight"]  # (B, H, W)
            loss_cos = (loss_cos * img_normal_weight * mask).sum() / (img_normal_weight * mask).sum()
        else:
            loss_cos = (loss_cos * mask).sum() / mask.sum()
        return loss_cos


if __name__ == "__main__":
    model = PSMNetClearGrasp(
        min_disp=12,
        max_disp=96,
        num_disp=128,
        set_zero=False,
        dilation=3,
        epsilon=1.0,
        grad_threshold=100.0,
        use_off=False,
        edge_weight=5.0,
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
