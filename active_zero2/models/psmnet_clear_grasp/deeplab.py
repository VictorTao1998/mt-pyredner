import torch
import torch.nn as nn
import torch.nn.functional as F
from active_zero2.models.psmnet_clear_grasp.aspp import build_aspp
from active_zero2.models.psmnet_clear_grasp.decoder_masks import build_decoder
from active_zero2.models.psmnet_clear_grasp.drn import drn_d_54


class DeepLab(nn.Module):
    def __init__(self, backbone="drn", num_classes=21, freeze_bn=False):
        super(DeepLab, self).__init__()
        assert backbone == "drn"
        output_stride = 8

        BatchNorm = nn.BatchNorm2d

        self.backbone = drn_d_54(BatchNorm=BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)

        if freeze_bn:
            self.freeze_bn()

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode="bilinear", align_corners=True)

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p


if __name__ == "__main__":
    model = DeepLab(backbone="drn")
    model.eval()
    input = torch.rand(1, 1, 256, 256)
    output = model(input)
    print(output.size())
