#!/usr/bin/env python
import os
import os.path as osp
import sys
import numpy as np
import cv2
import torch

_ROOT_DIR = os.path.abspath(osp.join(osp.dirname(__file__), ".."))
sys.path.insert(0, _ROOT_DIR)

import matplotlib.pyplot as plt

from active_zero2.config import cfg
from active_zero2.datasets.build_dataset import build_dataset
from active_zero2.utils.reprojection import apply_disparity, apply_disparity_v2
from active_zero2.models.psmnet_grad.utils import DispGrad


def main():
    config_file = osp.join(osp.dirname(__file__), "../configs/example_psmnetdilation.yml")
    cfg.merge_from_file(config_file)
    cfg.freeze()

    train_sim_dataset = build_dataset(cfg, mode="train", domain="sim")
    train_real_dataset = build_dataset(cfg, mode="train", domain="real")
    val_sim_dataset = build_dataset(cfg, mode="val", domain="sim")
    val_real_dataset = build_dataset(cfg, mode="val", domain="real")
    test_sim_dataset = build_dataset(cfg, mode="test", domain="sim")
    test_real_dataset = build_dataset(cfg, mode="test", domain="real")

    data = test_real_dataset[0]
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            print(k, v.shape)
        else:
            print(k, v)
    print("edge: ", data["img_disp_edge_l"].numpy().mean())
    cv2.imshow("edge", (data["img_disp_edge_l"].numpy().astype(np.uint8) * 255))
    cv2.waitKey(0)
    exit()

    disp_grad = DispGrad()
    disp_gt = disp_grad(data["img_disp_l"].unsqueeze(0))
    print(disp_gt.shape)
    print(
        "x grad: ",
        "min: ",
        disp_gt[0, 0].min(),
        "max: ",
        disp_gt[0, 0].max(),
        "avg: ",
        disp_gt[0, 0].mean(),
        "abs avg: ",
        torch.abs(disp_gt[0, 0]).mean(),
    )
    print(
        "y grad: ",
        "min: ",
        disp_gt[0, 1].min(),
        "max: ",
        disp_gt[0, 1].max(),
        "avg: ",
        disp_gt[0, 1].mean(),
        "abs avg: ",
        torch.abs(disp_gt[0, 0]).mean(),
    )

    # check data aug
    data_1 = train_sim_dataset[1]
    mean = 0.45
    std = 0.224

    img_l_0 = data["img_l"].numpy()[0]
    img_l_1 = data_1["img_l"].numpy()[0]
    img_l_0 = img_l_0 * std + mean
    img_l_1 = img_l_1 * std + mean

    img_l_0 = (img_l_0 * 255).astype(np.uint8)
    img_l_1 = (img_l_1 * 255).astype(np.uint8)
    cv2.imwrite("img_l_0.png", img_l_0)
    cv2.imwrite("img_l_1.png", img_l_1)

    if "img_points" in data:
        img_points = data["img_points"].numpy()
        img_points = np.rint(img_points).astype(np.uint16)
        img_l_0 = cv2.cvtColor(img_l_0, cv2.COLOR_GRAY2BGR)
        for p in img_points.T:
            img_l_0[p[1], p[0]] = np.array([255, 0.0, 0.0])
        cv2.imwrite("img_l_points.png", img_l_0)
        exit()

    # check reprojection
    img_l = data["img_l"].unsqueeze(0).cuda()
    img_r = data["img_r"].unsqueeze(0).cuda()
    img_disp_l = data["img_disp_l"].unsqueeze(0).cuda()

    img_pattern_l = data["img_pattern_l"].unsqueeze(0).cuda()
    img_pattern_r = data["img_pattern_r"].unsqueeze(0).cuda()

    img_l_reprojed = apply_disparity_v2(img_r, -img_disp_l)
    img_pattern_l_reprojed = apply_disparity_v2(img_pattern_r, -img_disp_l)

    img_l = img_l[0, 0].cpu().numpy()
    img_r = img_r[0, 0].cpu().numpy()
    img_l_reprojed = img_l_reprojed[0, 0].cpu().numpy()
    img_pattern_l = img_pattern_l[0, 0].cpu().numpy()
    img_pattern_r = img_pattern_r[0, 0].cpu().numpy()
    img_pattern_l_reprojed = img_pattern_l_reprojed[0, 0].cpu().numpy()

    plt.figure(
        f"Check reprojection",
        figsize=(24, 16),
    )
    plt.subplot(2, 4, 1)
    plt.gca().set_title("img_l")
    plt.imshow(img_l)
    plt.subplot(2, 4, 2)
    plt.gca().set_title("img_r")
    plt.imshow(img_r)
    plt.subplot(2, 4, 3)
    plt.gca().set_title("img_l_reproj")
    plt.imshow(img_l_reprojed)
    plt.subplot(2, 4, 4)
    plt.gca().set_title("img_l_reproj_err")
    plt.imshow(img_l - img_l_reprojed)
    # plt.colorbar()
    plt.subplot(2, 4, 5)
    plt.gca().set_title("img_pattern_l")
    plt.imshow(img_pattern_l)
    plt.subplot(2, 4, 6)
    plt.gca().set_title("img_pattern_r")
    plt.imshow(img_pattern_r)
    plt.subplot(2, 4, 7)
    plt.gca().set_title("img_pattern_l_reproj")
    plt.imshow(img_pattern_l_reprojed)
    plt.subplot(2, 4, 8)
    plt.gca().set_title("img_pattern_l_reproj_err")
    plt.imshow(img_pattern_l - img_pattern_l_reprojed)
    # plt.colorbar()

    plt.show(block=False)

    plt.figure(
        f"Check disp reprojection",
        figsize=(36, 8),
    )
    img_disp_l = data["img_disp_l"].numpy()[0]
    img_disp_r = data["img_disp_r"].numpy()[0]
    img_disp_l_reprojed = data["img_disp_l_reprojed"].numpy()[0]
    mask = img_disp_l_reprojed > 0
    disp_reproj_err = img_disp_l - img_disp_l_reprojed
    img_disp_l_reprojed_v2 = data["img_disp_l_reprojed_v2"].numpy()[0]
    mask_v2 = img_disp_l_reprojed_v2 > 0
    disp_reproj_err_v2 = img_disp_l - img_disp_l_reprojed_v2

    plt.subplot(1, 6, 1)
    plt.gca().set_title("img_l")
    plt.imshow(img_disp_l, vmin=0, vmax=96, cmap="jet")
    plt.subplot(1, 6, 2)
    plt.gca().set_title("img_r")
    plt.imshow(img_disp_r, vmin=0, vmax=96, cmap="jet")
    plt.subplot(1, 6, 3)
    plt.gca().set_title("img_l_reproj")
    plt.imshow(img_disp_l_reprojed, vmin=0, vmax=96, cmap="jet")
    plt.subplot(1, 6, 4)
    plt.gca().set_title("img_l_reproj_err")
    plt.imshow(disp_reproj_err, vmin=-10, vmax=10, cmap="jet")
    plt.subplot(1, 6, 5)
    plt.gca().set_title("img_l_reproj_v2")
    plt.imshow(img_disp_l_reprojed_v2, vmin=0, vmax=96, cmap="jet")
    plt.subplot(1, 6, 6)
    plt.gca().set_title("img_l_reproj_err_v2")
    plt.imshow(disp_reproj_err_v2, vmin=-10, vmax=10, cmap="jet")

    print("v1: ", np.mean(np.abs(disp_reproj_err[mask])))
    print("v2: ", np.mean(np.abs(disp_reproj_err_v2[mask_v2])))

    plt.show(block=False)

    plt.figure(f"Occlusion mask")
    plt.imshow(data["occ_mask_l"][0].numpy())
    plt.show()


if __name__ == "__main__":
    main()
