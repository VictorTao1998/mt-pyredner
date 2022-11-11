import numpy as np
import argparse
from tqdm import tqdm
from path import Path
import cv2
import matplotlib.pyplot as plt
import time
import torch
import os

_ROOT_DIR = os.path.abspath(os.path.dirname(__file__) + "../../..")
from skimage import feature

from active_zero2.utils.metrics import ErrorMetric
from active_zero2.utils.loguru_logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="disp map inpainting and evaluation")
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
    parser.add_argument("--lambda-d", type=float, default=1.0)
    parser.add_argument("--lambda-s", type=float, default=3.0)
    parser.add_argument("--iter", type=int, default=500)
    parser.add_argument("--refined", action="store_true")
    parser.add_argument("--gt-edge", action="store_true")

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    data_folder = Path(args.data_folder)
    pred_folder = Path(args.pred_folder)
    output_folder = Path(args.pred_folder + "_inpainted")
    output_folder.makedirs_p()

    # run name
    timestamp = time.strftime("%y-%m-%d_%H-%M-%S")
    run_name = "{:s}".format(timestamp)
    logger = setup_logger(f"ActiveZero2.test inpainting", output_folder, rank=0, filename=f"log.test.{run_name}.txt")
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
        conf_map = np.load(curr_pred_dir / "confidence.npy")
        conf_mask = conf_map[1] > args.threshold
        conf_map = np.clip((conf_map[1] - args.threshold) / (1 - args.threshold), 0, 1)
        cv2.imwrite(view_folder / "conf_map_u8.png", (conf_map * 255).astype(np.uint8))
        conf_mask = conf_mask.astype(np.uint8)
        # remove small outlier region
        kernel = np.ones((5, 5), np.uint8)
        conf_mask = cv2.erode(conf_mask, kernel=kernel, iterations=3)
        conf_mask = cv2.dilate(conf_mask, kernel=kernel, iterations=3)
        # set edge to conf
        kernel = np.ones((3, 3), np.uint8)
        conf_mask2 = cv2.dilate(conf_mask, kernel=kernel, iterations=2)
        conf_mask2 = cv2.erode(conf_mask2, kernel=kernel, iterations=2)
        conf_edge = (conf_mask2.astype(int) - conf_mask.astype(int)).astype(np.uint8)
        cv2.imwrite(view_folder / "conf_edge.png", conf_edge * 255)
        conf_mask = conf_mask2.astype(float)
        plt.imsave(view_folder / "conf_mask.png", conf_mask.astype(float), vmin=0.0, vmax=1.0, cmap="jet")

        disp_pred = np.load(curr_pred_dir / "disp_pred.npy")
        disp_conf = disp_pred * conf_mask

        cv2.imwrite(view_folder / "disp_conf_u16.png", (disp_conf * 500).astype(np.uint16))

        # label_image = cv2.imread(data_folder / sc / "irL_label_image.png", cv2.IMREAD_UNCHANGED)
        # label_image = cv2.resize(label_image, (960, 540), interpolation=cv2.INTER_NEAREST)
        # label_edge = cv2.Canny(label_image.astype(np.uint8), 1, 2)
        # kernel = np.ones((3, 3), np.uint8)
        # label_edge = cv2.dilate(label_edge, kernel, iterations=1)
        # cv2.imwrite(view_folder / "label_edge.png", label_edge)

        off_image = cv2.imread(data_folder / sc / "1024_irL_real_off.png", 0)
        off_image = cv2.resize(off_image, (960, 540), interpolation=cv2.INTER_LANCZOS4)
        off_edge = cv2.Canny(off_image, 100, 300)
        cv2.imwrite(view_folder / "off_edge.png", off_edge)

        if args.gt_edge:
            gt_edge = cv2.imread(curr_pred_dir / "edge_gt.png", 0)
            gt_edge = gt_edge[2:-2]
            edge = (gt_edge + off_edge.astype(int) + conf_edge.astype(int)).astype(np.uint8)
        else:
            pred_edge = cv2.imread(curr_pred_dir / "edge_pred.png", 0)
            pred_edge = pred_edge[2:-2]
            edge = (pred_edge + off_edge.astype(int) + conf_edge.astype(int)).astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        edge = cv2.dilate(edge, kernel, iterations=2)
        cv2.imwrite(view_folder / "edge.png", edge)

        o = (
            f"{_ROOT_DIR}/active_zero2/inpainting/cpp/build/densify {view_folder}/disp_conf_u16.png"
            f" {view_folder}/conf_map_u8.png {view_folder}/edge.png {args.lambda_d} {args.lambda_s} {args.iter}"
            f" {view_folder}/disp_AR.txt"
        )
        os.system(o)

        disp_init = np.loadtxt(view_folder / "disp_AR_init.txt")
        plt.imsave(
            view_folder / "disp_init.png",
            np.clip(disp_init, 0, metric.max_disp),
            vmin=0.0,
            vmax=metric.max_disp,
            cmap="jet",
        )

        disp_AR = np.loadtxt(view_folder / "disp_AR.txt")
        disp_AR = disp_AR * (1 - conf_mask) + disp_conf
        np.save(view_folder / "disp_AR.npy", disp_AR)

        noise_mask = np.logical_and(disp_conf > 0, np.abs(disp_AR - disp_conf) > 2)
        disp_conf[noise_mask] = 0
        disp_AR_refined = np.loadtxt(view_folder / "disp_AR_refined.txt")
        disp_conf[disp_conf == 0] = disp_AR_refined[disp_conf == 0]
        np.save(view_folder / "disp_AR_refined.npy", disp_conf)
        if args.refined:
            disp_AR = np.load(view_folder / "disp_AR_refined.npy")
        else:
            disp_AR = np.load(view_folder / "disp_AR.npy")

        pred_dict = {"disp_AR": disp_AR}

        data = {k: v.unsqueeze(0) for k, v in data.items() if isinstance(v, torch.Tensor)}
        data["dir"] = sc
        metric.compute(data, pred_dict, save_folder=view_folder, real_data=True)

    # END
    epoch_time_eval = time.time() - eval_tic
    logger.info("Real Test total_time: {:.2f}s".format(epoch_time_eval))
    logger.info("Real Eval Metric: \n" + metric.summary())


if __name__ == "__main__":
    main()
