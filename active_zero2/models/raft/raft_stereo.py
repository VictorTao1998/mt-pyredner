import torch
import torch.nn as nn
import torch.nn.functional as F

from active_zero2.models.raft.corr import AlternateCorrBlock, CorrBlock1D, CorrBlockFast1D, PytorchAlternateCorrBlock1D
from active_zero2.models.raft.extractor import BasicEncoder, MultiBasicEncoder, ResidualBlock
from active_zero2.models.raft.update import BasicMultiUpdateBlock
from active_zero2.models.raft.raft_utils import coords_grid, upflow8
from active_zero2.utils.reprojection import compute_reproj_loss_patch

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass


class RAFTStereo(nn.Module):
    def __init__(self, max_disp):
        super().__init__()
        self.max_disp = max_disp
        self.CORR_IMPLEMENTATION = "reg"  # ["reg", "alt", "reg_cuda", "alt_cuda"]
        self.SHARE_BACKBONE = True
        self.CORR_LEVELS = 4
        self.CORR_RADIUS = 4
        self.N_DOWNSAMPLE = 2
        self.SLOW_FAST_GRU = True
        self.N_GRU_LAYERS = 3
        self.HIDDEN_DIMS = [128] * 3
        self.MIXED_PRECISION = True
        self.TRAIN_ITERS = 22
        self.loss_gamma = 0.9
        context_dims = self.HIDDEN_DIMS

        self.cnet = MultiBasicEncoder(
            output_dim=[self.HIDDEN_DIMS, context_dims],
            norm_fn="batch",
            downsample=self.N_DOWNSAMPLE,
        )
        self.update_block = BasicMultiUpdateBlock(hidden_dims=self.HIDDEN_DIMS)

        self.context_zqr_convs = nn.ModuleList(
            [nn.Conv2d(context_dims[i], self.HIDDEN_DIMS[i] * 3, 3, padding=3 // 2) for i in range(self.N_GRU_LAYERS)]
        )

        if self.SHARE_BACKBONE:
            self.conv2 = nn.Sequential(
                ResidualBlock(128, 128, "instance", stride=1),
                nn.Conv2d(128, 256, 3, padding=1),
            )
        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn="instance", downsample=self.N_DOWNSAMPLE)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, _, H, W = img.shape

        coords0 = coords_grid(N, H, W).to(img.device)
        coords1 = coords_grid(N, H, W).to(img.device)

        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination"""
        N, D, H, W = flow.shape
        factor = 2**self.N_DOWNSAMPLE
        mask = mask.view(N, 1, 9, factor, factor, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(factor * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, D, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, D, factor * H, factor * W)

    def forward(self, data_batch):
        """Estimate optical flow between pair of frames"""
        image1, image2 = data_batch["img_l"], data_batch["img_r"]
        iters = 12
        # run the context network
        with autocast(enabled=self.MIXED_PRECISION):
            if self.SHARE_BACKBONE:
                *cnet_list, x = self.cnet(
                    torch.cat((image1, image2), dim=0),
                    dual_inp=True,
                    num_layers=self.N_GRU_LAYERS,
                )
                fmap1, fmap2 = self.conv2(x).split(dim=0, split_size=x.shape[0] // 2)
            else:
                cnet_list = self.cnet(image1, num_layers=self.N_GRU_LAYERS)
                fmap1, fmap2 = self.fnet([image1, image2])
            net_list = [torch.tanh(x[0]) for x in cnet_list]
            inp_list = [torch.relu(x[1]) for x in cnet_list]

            # Rather than running the GRU's conv layers on the context features multiple times, we do it once at the beginning
            inp_list = [
                list(conv(i).split(split_size=conv.out_channels // 3, dim=1))
                for i, conv in zip(inp_list, self.context_zqr_convs)
            ]

        if self.CORR_IMPLEMENTATION == "reg":  # Default
            corr_block = CorrBlock1D
            fmap1, fmap2 = fmap1.float(), fmap2.float()
        elif self.CORR_IMPLEMENTATION == "alt":  # More memory efficient than reg
            corr_block = PytorchAlternateCorrBlock1D
            fmap1, fmap2 = fmap1.float(), fmap2.float()
        elif self.CORR_IMPLEMENTATION == "reg_cuda":  # Faster version of reg
            corr_block = CorrBlockFast1D
        elif self.CORR_IMPLEMENTATION == "alt_cuda":  # Faster version of alt
            corr_block = AlternateCorrBlock
        corr_fn = corr_block(fmap1, fmap2, radius=self.CORR_RADIUS, num_levels=self.CORR_LEVELS)

        coords0, coords1 = self.initialize_flow(net_list[0])

        disp_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)  # index correlation volume
            flow = coords1 - coords0
            with autocast(enabled=self.MIXED_PRECISION):
                if self.N_GRU_LAYERS == 3 and self.SLOW_FAST_GRU:  # Update low-res GRU
                    net_list = self.update_block(
                        net_list,
                        inp_list,
                        iter32=True,
                        iter16=False,
                        iter08=False,
                        update=False,
                    )
                if self.N_GRU_LAYERS >= 2 and self.SLOW_FAST_GRU:  # Update low-res GRU and mid-res GRU
                    net_list = self.update_block(
                        net_list,
                        inp_list,
                        iter32=self.N_GRU_LAYERS == 3,
                        iter16=True,
                        iter08=False,
                        update=False,
                    )
                net_list, up_mask, delta_flow = self.update_block(
                    net_list,
                    inp_list,
                    corr,
                    flow,
                    iter32=self.N_GRU_LAYERS == 3,
                    iter16=self.N_GRU_LAYERS >= 2,
                )

            # in stereo mode, project flow onto epipolar
            delta_flow[:, 1] = 0.0

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # We do not need to upsample or output intermediate results in test_mode
            if not self.training and itr < iters - 1:
                continue

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            flow_up = flow_up[:, :1]

            disp_predictions.append(-flow_up)  # Convert pred disp to flow

        return {"disp_predictions": disp_predictions}

    def compute_disp_loss(self, data_batch, pred_dict):
        disp_gt = data_batch["img_disp_l"]
        # Get stereo loss on sim
        # Note in training we do not exclude bg
        mask = (disp_gt < self.max_disp) * (disp_gt > 0)

        n_predictions = len(pred_dict["disp_predictions"])
        assert n_predictions >= 1
        flow_loss = 0.0

        for i in range(n_predictions):
            # We adjust the loss_gamma so it is consistent for any number of RAFT-Stereo iterations
            adjusted_loss_gamma = self.loss_gamma ** (15 / (n_predictions - 1))
            i_weight = adjusted_loss_gamma ** (n_predictions - i - 1)
            i_loss = (pred_dict["disp_predictions"][i] - disp_gt).abs()
            flow_loss += i_weight * i_loss[mask.bool()].mean()

        return flow_loss

    def compute_reproj_loss(self, data_batch, pred_dict, use_mask: bool, patch_size: int, only_last_pred: bool):
        if use_mask:
            disp_gt = data_batch["img_disp_l"]
            mask = (disp_gt < self.max_disp) * (disp_gt > 0)
            mask.detach()
        else:
            mask = None

        if only_last_pred:
            loss_reproj = compute_reproj_loss_patch(
                data_batch["img_pattern_l"],
                data_batch["img_pattern_r"],
                pred_disp_l=pred_dict["disp_predictions"][-1],
                mask=mask,
                ps=patch_size,
            )
            return loss_reproj
        else:
            raise NotImplementedError
