#!/usr/bin/env python
import os
import os.path as osp
import sys
import numpy as np
from tqdm import tqdm

import torch

_ROOT_DIR = os.path.abspath(osp.join(osp.dirname(__file__), ".."))
sys.path.insert(0, _ROOT_DIR)

import matplotlib.pyplot as plt

from active_zero2.config import cfg
from active_zero2.datasets.build_dataset import build_dataset
from active_zero2.utils.reprojection import apply_disparity


def main():
    config_file = osp.join(osp.dirname(__file__), "../configs/example.yml")
    cfg.merge_from_file(config_file)
    cfg.freeze()

    train_sim_dataset = build_dataset(cfg, mode="train", domain="sim")
    val_sim_dataset = build_dataset(cfg, mode="val", domain="sim")
    test_real_dataset = build_dataset(cfg, mode="test", domain="real")

    train_disps = []
    val_disps = []
    test_disps = []

    train_depths = []
    val_depths = []
    test_depths = []

    for d in tqdm(train_sim_dataset):
        train_disps.append(d["img_disp_l"][0].numpy())
        train_depths.append(d["img_depth_l"][0].numpy())

    for d in tqdm(val_sim_dataset):
        val_disps.append(d["img_disp_l"][0].numpy())
        val_depths.append(d["img_depth_l"][0].numpy())

    for d in tqdm(test_real_dataset):
        test_disps.append(d["img_disp_l"][0][2:-2].numpy())
        test_depths.append(d["img_depth_l"][0][2:-2].numpy())

    train_disps = np.stack(train_disps)
    val_disps = np.stack(val_disps)
    test_disps = np.stack(test_disps)
    train_depths = np.stack(train_depths)
    val_depths = np.stack(val_depths)
    test_depths = np.stack(test_depths)

    print("Train Stats: ", train_disps.shape)
    print("Disp min: ", train_disps.min(), "max: ", train_disps.max(), "mean: ", train_disps.mean())
    print("Depth min: ", train_depths.min(), "max: ", train_depths.max(), "mean: ", train_depths.mean())

    print("Val Stats: ", val_disps.shape)
    print("Disp min: ", val_disps.min(), "max: ", val_disps.max(), "mean: ", val_disps.mean())
    print("Depth min: ", val_depths.min(), "max: ", val_depths.max(), "mean: ", val_depths.mean())

    print("Test Stats: ", test_disps.shape)
    print("Disp min: ", test_disps.min(), "max: ", test_disps.max(), "mean: ", test_disps.mean())
    print("Depth min: ", test_depths.min(), "max: ", test_depths.max(), "mean: ", test_depths.mean())

    plt.figure(
        f"Dataset Stats",
        figsize=(36, 24),
    )
    plt.subplot(2, 3, 1)
    plt.gca().set_title("Train Disp")
    plt.hist(train_disps.flatten(), 32, (0, 96))
    plt.xticks(list(np.arange(0, 100, 5)))
    plt.subplot(2, 3, 2)
    plt.gca().set_title("Val Disp")
    plt.hist(val_disps.flatten(), 32, (0, 96))
    plt.xticks(list(np.arange(0, 100, 5)))
    plt.subplot(2, 3, 3)
    plt.gca().set_title("Test Disp ")
    plt.hist(test_disps.flatten(), 32, (0, 96))
    plt.xticks(list(np.arange(0, 100, 5)))
    plt.subplot(2, 3, 4)
    plt.gca().set_title("Train Depth")
    plt.hist(train_depths.flatten(), 32, (0, 4))
    plt.xticks(list(np.arange(0, 4, 0.2)))
    plt.subplot(2, 3, 5)
    plt.gca().set_title("Val Depth")
    plt.hist(val_depths.flatten(), 32, (0, 4))
    plt.xticks(list(np.arange(0, 4, 0.2)))
    plt.subplot(2, 3, 6)
    plt.gca().set_title("Test Depth ")
    plt.hist(test_depths.flatten(), 32, (0, 4))
    plt.xticks(list(np.arange(0, 4, 0.2)))

    plt.show()


def main2():
    config_file = osp.join(osp.dirname(__file__), "../configs/psmnetdilation_thinv2B.yml")
    cfg.merge_from_file(config_file)
    cfg.freeze()

    train_sim_dataset = build_dataset(cfg, mode="train", domain="sim")

    train_disps = []

    train_depths = []

    for d in tqdm(train_sim_dataset):
        train_disps.append(d["img_disp_l"][0].numpy())
        train_depths.append(d["img_depth_l"][0].numpy())

    train_disps = np.stack(train_disps)
    train_depths = np.stack(train_depths)

    print("Train Stats: ", train_disps.shape)
    print("Disp min: ", train_disps.min(), "max: ", train_disps.max(), "mean: ", train_disps.mean())
    print("Depth min: ", train_depths.min(), "max: ", train_depths.max(), "mean: ", train_depths.mean())


if __name__ == "__main__":
    main2()
