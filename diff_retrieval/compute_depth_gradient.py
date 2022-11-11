import os

import cv2
import numpy as np
from path import Path
from tqdm import tqdm
from scipy import ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F


class Sobel(nn.Module):
    def __init__(self):
        super().__init__()
        self.filter = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, bias=False)

        Gx = -torch.tensor([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
        Gy = -torch.tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
        G = torch.cat([Gx.unsqueeze(0), Gy.unsqueeze(0)], 0)
        G = G.unsqueeze(1)
        self.filter.weight = nn.Parameter(G, requires_grad=False)

    def forward(self, img):
        img = F.pad(img, (1, 1, 1, 1), mode="reflect")
        x = self.filter(img)
        return x


root_dir = Path("/media/DATA/LINUX_DATA/ICCV2021_Diagnosis/baseline_images")
view_dir_list = sorted(root_dir.listdir("baseline_*_02-*"))
for view_dir in tqdm(view_dir_list):
    view_idx = view_dir.name.split("-")[-1]
    if view_idx == 0:
        continue
    depth = cv2.imread(view_dir / "1024_depthL_real.png", cv2.IMREAD_UNCHANGED)
    depth = (depth.astype(float)) / 1000.0

    mask = (depth == 0).astype(np.uint8)
    ker = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.dilate(mask, ker)
    cv2.imwrite(view_dir / "maskL.png", (mask.astype(np.uint8) * 255))
    os.system(
        f"kubectl cp {view_dir}/maskL.png rayc-http-8487b7c466-xcqrd:/messytable-slow/messy-table-dataset/baseline_images/{view_dir.name}")

    grad_x = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=3).astype(np.float32)
    grad_y = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=3).astype(np.float32)

    np.save(view_dir / "1024_depthL_grad_real.npy", np.stack([grad_x, grad_y], axis=-1))
    os.system(
        f"kubectl cp {view_dir}/1024_depthL_grad_real.npy rayc-http-8487b7c466-xcqrd:/messytable-slow/messy-table-dataset/baseline_images/{view_dir.name}")

    # depth_t = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0).float()
    # sobel = Sobel()
    # grad = sobel(depth_t).numpy()[0]
    #
    # print(np.allclose(grad_x, grad[0]))
    # print(np.allclose(grad_y, grad[1]))
