"""
Depth Filler Network.

Author: Hongjie Fang.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from active_zero2.models.psmnet_dilation_df.dense import DenseBlock
from active_zero2.models.psmnet_dilation_df.duc import DenseUpsamplingConvolution


class DFNet(nn.Module):
    """
    Depth Filler Network (DFNet).
    """

    def __init__(self, in_channels=4, hidden_channels=64, residual: bool = True, L=5, k=12, use_DUC=True):
        super(DFNet, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.residual = residual
        self.L = L
        self.k = k
        self.use_DUC = use_DUC
        # First
        self.first = nn.Sequential(
            nn.Conv2d(self.in_channels, self.hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.hidden_channels, self.hidden_channels),
            nn.ReLU(True),
        )
        # Dense1: skip
        self.dense1s_conv1 = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.hidden_channels, self.hidden_channels),
            nn.ReLU(True),
        )
        self.dense1s = DenseBlock(self.hidden_channels, self.L, self.k, with_bn=True)
        self.dense1s_conv2 = nn.Sequential(
            nn.Conv2d(self.k, self.hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.hidden_channels, self.hidden_channels),
            nn.ReLU(True),
        )
        # Dense1: normal
        self.dense1_conv1 = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.hidden_channels, self.hidden_channels),
            nn.ReLU(True),
        )
        self.dense1 = DenseBlock(self.hidden_channels, self.L, self.k, with_bn=True)
        self.dense1_conv2 = nn.Sequential(
            nn.Conv2d(self.k, self.hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.hidden_channels, self.hidden_channels),
            nn.ReLU(True),
        )
        # Dense2: skip
        self.dense2s_conv1 = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.hidden_channels, self.hidden_channels),
            nn.ReLU(True),
        )
        self.dense2s = DenseBlock(self.hidden_channels, self.L, self.k, with_bn=True)
        self.dense2s_conv2 = nn.Sequential(
            nn.Conv2d(self.k, self.hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.hidden_channels, self.hidden_channels),
            nn.ReLU(True),
        )
        # Dense2: normal
        self.dense2_conv1 = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.hidden_channels, self.hidden_channels),
            nn.ReLU(True),
        )
        self.dense2 = DenseBlock(self.hidden_channels, self.L, self.k, with_bn=True)
        self.dense2_conv2 = nn.Sequential(
            nn.Conv2d(self.k, self.hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.hidden_channels, self.hidden_channels),
            nn.ReLU(True),
        )
        # Dense3: skip
        self.dense3s_conv1 = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.hidden_channels, self.hidden_channels),
            nn.ReLU(True),
        )
        self.dense3s = DenseBlock(self.hidden_channels, self.L, self.k, with_bn=True)
        self.dense3s_conv2 = nn.Sequential(
            nn.Conv2d(self.k, self.hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.hidden_channels, self.hidden_channels),
            nn.ReLU(True),
        )
        # Dense3: normal
        self.dense3_conv1 = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.hidden_channels, self.hidden_channels),
            nn.ReLU(True),
        )
        self.dense3 = DenseBlock(self.hidden_channels, self.L, self.k, with_bn=True)
        self.dense3_conv2 = nn.Sequential(
            nn.Conv2d(self.k, self.hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.hidden_channels, self.hidden_channels),
            nn.ReLU(True),
        )
        # Dense4
        self.dense4_conv1 = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.hidden_channels, self.hidden_channels),
            nn.ReLU(True),
        )
        self.dense4 = DenseBlock(self.hidden_channels, self.L, self.k, with_bn=True)
        self.dense4_conv2 = nn.Sequential(
            nn.Conv2d(self.k, self.hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.hidden_channels, self.hidden_channels),
            nn.ReLU(True),
        )
        # DUC upsample 1
        self.updense1_conv = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.hidden_channels, self.hidden_channels),
            nn.ReLU(True),
        )
        self.updense1 = DenseBlock(self.hidden_channels, self.L, self.k, with_bn=True)
        self.updense1_duc = self._make_upconv(self.k, self.hidden_channels, upscale_factor=2)
        # DUC upsample 2
        self.updense2_conv = nn.Sequential(
            nn.Conv2d(self.hidden_channels * 2, self.hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.hidden_channels, self.hidden_channels),
            nn.ReLU(True),
        )
        self.updense2 = DenseBlock(self.hidden_channels, self.L, self.k, with_bn=True)
        self.updense2_duc = self._make_upconv(self.k, self.hidden_channels, upscale_factor=2)
        # DUC upsample 3
        self.updense3_conv = nn.Sequential(
            nn.Conv2d(self.hidden_channels * 2, self.hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.hidden_channels, self.hidden_channels),
            nn.ReLU(True),
        )
        self.updense3 = DenseBlock(self.hidden_channels, self.L, self.k, with_bn=True)
        self.updense3_duc = self._make_upconv(self.k, self.hidden_channels, upscale_factor=2)
        # DUC upsample 4
        self.updense4_conv = nn.Sequential(
            nn.Conv2d(self.hidden_channels * 2, self.hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.hidden_channels, self.hidden_channels),
            nn.ReLU(True),
        )
        self.updense4 = DenseBlock(self.hidden_channels, self.L, self.k, with_bn=True)
        self.updense4_duc = self._make_upconv(self.k, self.hidden_channels, upscale_factor=2)
        # Final
        self.final = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.hidden_channels, self.hidden_channels),
            nn.ReLU(True),
        )
        if self.residual:
            self.output = nn.Conv2d(self.hidden_channels, 1, kernel_size=3, stride=1, padding=1, bias=False)
            # init weight
            nn.init.zeros_(self.output.weight)
        else:
            self.output = nn.Sequential(
                nn.Conv2d(self.hidden_channels, 1, kernel_size=3, stride=1, padding=1), nn.Sigmoid()
            )

    def _make_upconv(self, in_channels, out_channels, upscale_factor=2):
        if self.use_DUC:
            return DenseUpsamplingConvolution(in_channels, out_channels, upscale_factor=upscale_factor)
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=upscale_factor,
                    stride=upscale_factor,
                    padding=0,
                    output_padding=0,
                ),
                nn.BatchNorm2d(out_channels, out_channels),
                nn.ReLU(True),
            )

    def forward(self, input):
        # BxCxHxW
        depth = input[:, 0:1]
        h = self.first(input)

        # dense1
        depth1 = F.interpolate(depth, scale_factor=0.5, mode="bilinear", align_corners=True)
        # dense1: skip
        h_d1s = self.dense1s_conv1(h)
        h_d1s = self.dense1s(torch.cat((h_d1s, depth1), dim=1))
        h_d1s = self.dense1s_conv2(h_d1s)
        # dense1: normal
        h = self.dense1_conv1(h)
        h = self.dense1(torch.cat((h, depth1), dim=1))
        h = self.dense1_conv2(h)

        # dense2
        depth2 = F.interpolate(depth1, scale_factor=0.5, mode="bilinear", align_corners=True)
        # dense2: skip
        h_d2s = self.dense2s_conv1(h)
        h_d2s = self.dense2s(torch.cat((h_d2s, depth2), dim=1))
        h_d2s = self.dense2s_conv2(h_d2s)
        # dense2: normal
        h = self.dense2_conv1(h)
        h = self.dense2(torch.cat((h, depth2), dim=1))
        h = self.dense2_conv2(h)

        # dense3
        depth3 = F.interpolate(depth2, scale_factor=0.5, mode="bilinear", align_corners=True)
        # dense3: skip
        h_d3s = self.dense3s_conv1(h)
        h_d3s = self.dense3s(torch.cat((h_d3s, depth3), dim=1))
        h_d3s = self.dense3s_conv2(h_d3s)
        # dense3: normal
        h = self.dense3_conv1(h)
        h = self.dense3(torch.cat((h, depth3), dim=1))
        h = self.dense3_conv2(h)

        # dense4
        depth4 = F.interpolate(depth3, scale_factor=0.5, mode="bilinear", align_corners=True)
        h = self.dense4_conv1(h)
        h = self.dense4(torch.cat((h, depth4), dim=1))
        h = self.dense4_conv2(h)

        # updense1
        h = self.updense1_conv(h)
        h = self.updense1(torch.cat((h, depth4), dim=1))
        h = self.updense1_duc(h)

        # updense2
        h = torch.cat((h, h_d3s), dim=1)
        h = self.updense2_conv(h)
        h = self.updense2(torch.cat((h, depth3), dim=1))
        h = self.updense2_duc(h)

        # updense3
        h = torch.cat((h, h_d2s), dim=1)
        h = self.updense3_conv(h)
        h = self.updense3(torch.cat((h, depth2), dim=1))
        h = self.updense3_duc(h)

        # updense4: 360 x 640 -> 720 x 1280
        h = torch.cat((h, h_d1s), dim=1)
        h = self.updense4_conv(h)
        h = self.updense4(torch.cat((h, depth1), dim=1))
        h = self.updense4_duc(h)

        # final
        h = self.final(h)
        h = self.output(h)
        return h


if __name__ == "__main__":
    df_net = DFNet().cuda()

    d = torch.rand(1, 4, 544, 960).cuda()

    d_out = df_net(d)
    print(d_out.shape)
