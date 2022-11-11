import os
import os.path as osp
import sys

_ROOT_DIR = os.path.abspath(osp.join(osp.dirname(__file__), ".."))
sys.path.insert(0, _ROOT_DIR)

import argparse
import hashlib
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from path import Path
from tqdm import tqdm

from active_zero2.utils.io import load_pickle
from simsense import DepthSensor
from data_rendering.utils.sim_depth import sim_ir_noise

MAX_DEPTH = 2.0


def visualize_depth(depth):
    cmap = plt.get_cmap("rainbow")
    if depth.dtype == np.uint16:
        depth = depth.astype(np.float32) / 1000.0
    if len(depth.shape) == 3:
        depth = depth[..., 0]
    depth = np.clip(depth / MAX_DEPTH, 0.0, 1.0)
    vis_depth = cmap(depth)
    vis_depth = (vis_depth[:, :, :3] * 255.0).astype(np.uint8)
    vis_depth = cv2.cvtColor(vis_depth, cv2.COLOR_RGB2BGR)
    return vis_depth


def get_md5(filename):
    hash_obj = hashlib.md5()
    with open(filename, "rb") as f:
        hash_obj.update(f.read())
    return hash_obj.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-s", "--split-file", type=str, required=True)
    parser.add_argument("-d", "--dir", type=str, required=True)
    parser.add_argument("-l", "--left-name", type=str, default="0128_irL_kuafu_half.png")
    parser.add_argument("-r", "--right-name", type=str, default="0128_irR_kuafu_half.png")
    parser.add_argument("-n", "--ndisp", type=int, default=96)
    parser.add_argument("-w", "--wsize", type=int, default=7)
    parser.add_argument("--unique", type=int, required=True)
    args = parser.parse_args()
    return args


def main():
    LR_SIZE = (960, 540)
    RGB_SIZE = (1920, 1080)
    args = parse_args()
    data_dir = Path(args.dir)
    img_dirs = []
    with open(args.split_file, "r") as f_split_file:
        for l in f_split_file.readlines():
            img_dirs.append(data_dir / l.strip())

    timestamp = time.strftime("%y-%m-%d_%H-%M-%S")
    run_name = "{:s}".format(timestamp)
    log_file = open(data_dir / f"stereo_depth_{run_name}.txt", "w")
    log_file.write(str(args))
    log_file.write("\n")

    start = time.time()
    noise_scale = 0
    depth_name = f"1024_depth_uq{args.unique}_ss.png"
    index = 0
    median_blur_time = 0.0

    depthSensor_base = None
    depthSensor_hand = None

    for img_dir in tqdm(img_dirs):
        view_idx = int(img_dir.split("-")[-1])
        index += 1
        meta = load_pickle(img_dir / "meta.pkl")
        irl = cv2.imread(img_dir / args.left_name, 0)
        irr = cv2.imread(img_dir / args.right_name, 0)
        irl = cv2.resize(irl, (960, 540), interpolation=cv2.INTER_LINEAR)
        irr = cv2.resize(irr, (960, 540), interpolation=cv2.INTER_LINEAR)

        cam_ir_intrinsic = meta["intrinsic_l"]
        assert (
                900 < cam_ir_intrinsic[0, 2] < 1000 and 500 < cam_ir_intrinsic[1, 2] < 600
        ), f"illegal cam_ir_intrinsic: {cam_ir_intrinsic} in {img_dir}"

        cam_ir_intrinsic[:2] /= 2
        cam_ir_intrinsic = cam_ir_intrinsic.astype('float64')

        rt_lr = np.linalg.inv(meta["extrinsic_l"] @ np.linalg.inv(meta["extrinsic_r"]))
        rt_mainl = meta["extrinsic"] @ np.linalg.inv(meta["extrinsic_l"])
        rt_lr = rt_lr.astype('float64')
        rt_mainl = rt_mainl.astype('float64')

        if view_idx == 0 and depthSensor_base is None:
            depthSensor_base = DepthSensor(LR_SIZE, cam_ir_intrinsic, cam_ir_intrinsic, rt_lr,
                                           RGB_SIZE, meta["intrinsic"], rt_mainl,
                                           min_depth=0.0, max_depth=1e4,
                                           rectified=True,
                                           census_width=7, census_height=7, block_width=7, block_height=7,
                                           max_disp=96, p1_penalty=8, p2_penalty=32, median_filter_size=3,
                                           uniqueness_ratio=args.unique, depth_dilation=True)
        if view_idx > 0 and depthSensor_hand is None:
            depthSensor_hand = DepthSensor(LR_SIZE, cam_ir_intrinsic, cam_ir_intrinsic, rt_lr,
                                           RGB_SIZE, meta["intrinsic"], rt_mainl,
                                           min_depth=0.0, max_depth=1e4,
                                           rectified=True,
                                           census_width=7, census_height=7, block_width=7, block_height=7,
                                           max_disp=96, p1_penalty=8, p2_penalty=32, median_filter_size=3,
                                           uniqueness_ratio=args.unique, depth_dilation=True)

        if view_idx == 0:
            depthSensor = depthSensor_base
        else:
            depthSensor = depthSensor_hand
        depth = depthSensor.compute(irl.copy(), irr.copy())
        depth = (depth * 1000).astype(np.uint16)
        tic = time.time()
        depth = cv2.medianBlur(depth, 5)
        median_blur_time += (time.time() - tic)
        cv2.imwrite(os.path.join(img_dir, depth_name), depth)
        vis_depth = visualize_depth(depth)
        cv2.imwrite(os.path.join(img_dir, f"{depth_name[:-4]}_colored.png"), vis_depth)

        log_file.write(os.path.join(img_dir, depth_name))
        log_file.write("\n")
        log_file.flush()
        print(f"{index}/{len(img_dirs)} {time.time() - start:.2f}s")

    log_file.close()
    print("median blur: ", median_blur_time / index / 2)


if __name__ == "__main__":
    main()
