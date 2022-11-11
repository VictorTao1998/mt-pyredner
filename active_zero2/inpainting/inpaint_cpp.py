import numpy as np
import argparse
from tqdm import tqdm
from path import Path
import cv2
import matplotlib.pyplot as plt
import time
import torch
import os
import sys

_ROOT_DIR = os.path.abspath(os.path.dirname(__file__) + "../../..")
sys.path.insert(0, _ROOT_DIR)
from skimage import feature

from active_zero2.utils.metrics import ErrorMetric
from active_zero2.utils.loguru_logger import setup_logger
from active_zero2.inpainting.func import compute_disparity_plane


def parse_args():
    parser = argparse.ArgumentParser(
        description="disp map inpainting using confidence patch match stereo and evaluation"
    )
    parser.add_argument("-d", "--data-folder", type=str, required=True)
    parser.add_argument("-p", "--pred-folder", type=str, required=True)
    parser.add_argument(
        "-s",
        "--split-file",
        type=str,
        metavar="FILE",
        required=True,
    )
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--plane-size", type=int, default=5)
    parser.add_argument("--patch", type=int, default=31)
    parser.add_argument("--iter", type=int, default=3)
    parser.add_argument("--gt-edge", action="store_true")
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    data_folder = Path(args.data_folder)
    pred_folder = Path(args.pred_folder)
    output_folder = Path(args.pred_folder + "_cpp")
    output_folder.makedirs_p()

    # run name
    timestamp = time.strftime("%y-%m-%d_%H-%M-%S")
    run_name = "{:s}".format(timestamp)
    logger = setup_logger(f"ActiveZero2.test CPP", output_folder, rank=0, filename=f"log.test.{run_name}.txt")
    logger.info(args)

    # evaluate
    from active_zero2.config import cfg

    cfg.TEST.MAX_DISP = 128

    metric = ErrorMetric(
        model_type="Inpainting",
        use_mask=cfg.TEST.USE_MASK,
        max_disp=cfg.TEST.MAX_DISP,
        depth_range=cfg.TEST.DEPTH_RANGE,
        num_classes=cfg.DATA.NUM_CLASSES,
        is_depth=cfg.TEST.IS_DEPTH,
    )
    metric.reset()

    from active_zero2.datasets.messytable import MessyTableDataset

    dataset = MessyTableDataset(
        mode="test",
        domain="real",
        root_dir=args.data_folder,
        split_file=args.split_file,
        height=544,
        width=960,
        meta_name="meta.pkl",
        depth_name="depthL.png",
        normal_name="normalL.png",
        left_name="1024_irL_real.png",
        right_name="1024_irR_real.png",
        left_pattern_name="",
        right_pattern_name="",
        label_name="irL_label_image.png",
    )

    eval_tic = time.time()
    for data in tqdm(dataset):
        sc = data["dir"]
        print(sc)
        view_folder = output_folder / sc
        view_folder.makedirs_p()
        curr_pred_dir = pred_folder / f"{sc}_real"

        # left image
        conf_map = np.load(curr_pred_dir / "confidence.npy")
        conf_edges = []
        for i in range(4):
            conf_edge = feature.canny(conf_map[i], sigma=2)
            conf_edges.append(conf_edge)
            plt.imsave(view_folder / f"conf_edge{i}.png", conf_edge.astype(float), vmin=0.0, vmax=1.0, cmap="jet")

        conf_mask = conf_map[1] > args.threshold
        conf_mask_bak = conf_mask.copy()
        plt.imsave(view_folder / "conf_mask1.png", conf_mask.astype(float), vmin=0.0, vmax=1.0, cmap="jet")
        conf_map = np.clip((conf_map[1] - args.threshold) / (1 - args.threshold), 0, 1)
        cv2.imwrite(view_folder / "conf_map_u16.png", (conf_map * 65535).astype(np.uint16))
        conf_mask = conf_mask.astype(np.uint8)
        # Marker labelling
        ret, comps, stats, _ = cv2.connectedComponentsWithStats(conf_mask)
        inv_idx = np.where(stats[:, -1] < 300)[0]
        inv_mask = np.logical_or.reduce([comps == x for x in inv_idx])
        conf_mask *= np.logical_not(inv_mask)
        plt.imsave(view_folder / "conf_mask2.png", conf_mask.astype(float), vmin=0.0, vmax=1.0, cmap="jet")
        # kernel = np.ones((5, 5), np.uint8)
        # conf_mask = cv2.dilate(conf_mask, kernel=kernel, iterations=1)
        # conf_edge = np.logical_and(conf_edges[1], conf_mask > 0)
        # cv2.imwrite(view_folder / "edge_left.png", conf_edges[1].astype(np.uint8) * 255)
        # conf_mask = cv2.dilate(conf_mask, kernel=kernel, iterations=3)
        # plt.imsave(view_folder / "conf_mask.png", conf_mask.astype(float), vmin=0.0, vmax=1.0, cmap="jet")

        disp_pred = np.load(curr_pred_dir / "disp_pred.npy")
        disp_conf = disp_pred * conf_mask.astype(float)
        cv2.imwrite(view_folder / "disp_left_u16.png", (disp_conf * 500).astype(np.uint16))
        disp_plane, res = compute_disparity_plane(disp_conf, args.plane_size)
        # plt.imshow(res)
        # plt.colorbar()
        # plt.show()
        plane_conf = disp_plane[..., 2] != 0
        plane_conf = plane_conf.astype(np.uint8)
        ret, comps, stats, _ = cv2.connectedComponentsWithStats(plane_conf)
        inv_idx = np.where(stats[:, -1] < 300)[0]
        inv_mask = np.logical_or.reduce([comps == x for x in inv_idx])
        plane_conf *= np.logical_not(inv_mask)
        disp_plane *= plane_conf.astype(np.uint16)[..., None]
        plane_conf_left_num = np.sum(plane_conf)
        print("plane_conf_left_num: ", plane_conf_left_num)
        cv2.imwrite(view_folder / "disp_plane_u16.png", disp_plane)

        if not (data_folder / sc / "1024_irL_real_540.png").exists():
            ir_image = cv2.imread(data_folder / sc / "1024_irL_real.png")
            ir_image = cv2.resize(ir_image, (960, 540), interpolation=cv2.INTER_LANCZOS4)
            cv2.imwrite(data_folder / sc / "1024_irL_real_540.png", ir_image)
        if not (data_folder / sc / "1024_irR_real_540.png").exists():
            ir_image = cv2.imread(data_folder / sc / "1024_irR_real.png")
            ir_image = cv2.resize(ir_image, (960, 540), interpolation=cv2.INTER_LANCZOS4)
            cv2.imwrite(data_folder / sc / "1024_irR_real_540.png", ir_image)
        off_image = cv2.imread(data_folder / sc / "1024_irL_real_off.png", 0)
        off_image = cv2.resize(off_image, (960, 540), interpolation=cv2.INTER_LANCZOS4)
        off_edge = cv2.Canny(off_image, 100, 300)
        cv2.imwrite(view_folder / "off_edge.png", off_edge)
        off_edge = off_edge > 0
        pred_edge = cv2.imread(curr_pred_dir / "edge_pred.png", 0)
        pred_edge = pred_edge[2:-2] > 0
        edge = np.logical_or.reduce([pred_edge, off_edge]).astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        edge = cv2.dilate(edge, kernel, iterations=1)
        cv2.imwrite(view_folder / "edge_left.png", edge * 255)

        cpp_tic = time.time()
        print("Execute CPP...")
        o = (
            f"{_ROOT_DIR}/active_zero2/inpainting/ConfPatchProp/build/cpp"
            f" {data_folder}/{sc}/1024_irL_real_540.png {data_folder}/{sc}/1024_irR_real_540.png"
            f" {view_folder}/disp_left_u16.png  {view_folder}/conf_map_u16.png {view_folder}/disp_plane_u16.png"
        )
        if args.gt_edge:
            o += f" {curr_pred_dir}/edge_gt.png {args.patch} {args.iter} {view_folder}/disp_left_cpp.txt"
        else:
            o += f" {view_folder}/edge_left.png {args.patch} {args.iter} {view_folder}/disp_left_cpp.txt"
        os.system(o)
        cpp_time = time.time() - cpp_tic
        logger.info(f"{sc} CPP time: {cpp_time:.2f}s")

        disp_cpp = np.loadtxt(view_folder / "disp_left_cpp.txt")
        plt.imsave(
            view_folder / "disp_cpp.png",
            np.clip(disp_cpp, 0, metric.max_disp),
            vmin=0.0,
            vmax=metric.max_disp,
            cmap="jet",
        )
        disp_cpp_mask = np.logical_and(disp_cpp > 0.1, disp_cpp < metric.max_disp)
        plt.imsave(view_folder / "disp_cpp_mask.png", disp_cpp_mask.astype(float), vmin=0.0, vmax=1.0, cmap="jet")
        disp_cpp_mask = np.logical_and(disp_cpp_mask, np.logical_not(conf_mask_bak))
        disp_cpp = disp_cpp * disp_cpp_mask + disp_pred * (1 - disp_cpp_mask)
        np.save(view_folder / "disp_cpp.npy", disp_cpp)

        pred_dict = {"disp": disp_cpp}
        data = {k: v.unsqueeze(0) for k, v in data.items() if isinstance(v, torch.Tensor)}
        data["dir"] = sc
        curr_metric = metric.compute(data, pred_dict, save_folder=view_folder, real_data=True)
        logger.info("\t".join([f"{k}: {v:.3f}" for k, v in curr_metric.items()]))

    # END
    epoch_time_eval = time.time() - eval_tic
    logger.info("Real Test total_time: {:.2f}s".format(epoch_time_eval))
    logger.info("Real Eval Metric: \n" + metric.summary())


if __name__ == "__main__":
    main()
