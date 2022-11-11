import pyredner
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy
import scipy.ndimage
import numpy as np
import os
from path import Path
import json
import cv2
import matplotlib.pyplot as plt
import math
from sapien.core import Pose
from tqdm import tqdm
import csv
from loguru import logger
import time
import sys

from diffbm_utils import block_matching
from retieve_2 import intrinsic_from_opencv, ProjectionLight

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


SCENE_DIR = Path("/home/rayu/Projects/ICCV2021_Diagnosis/ocrtoc_materials/scenes")
OBJ_MODEL_DIR = Path("/home/rayu/Projects/ICCV2021_Diagnosis/ocrtoc_materials/models")
REPO_DIR = Path("/home/rayu/Projects/active_zero2")


class DiffSceneInference:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.makedirs_p()
        self.img_height, self.img_width = 360, 640
        self.epsilon = torch.tensor([1e-3], dtype=torch.float32, device=pyredner.get_device(), requires_grad=True)
        self.mu = torch.tensor([0.0], dtype=torch.float32, device=pyredner.get_device(), requires_grad=True)
        self.sigma = torch.tensor([1e-3], dtype=torch.float32, device=pyredner.get_device(), requires_grad=True)
        self.beta = torch.tensor([20], dtype=torch.float32, device=pyredner.get_device(), requires_grad=True)

        self.cnn_1 = nn.Sequential(nn.Conv2d(3, 32, 5, stride=1, padding=2, bias=False),
                                   nn.ReLU(),
                                   ).cuda()
        self.cnn_2 = nn.Conv2d(32, 1, 1, bias=False).cuda()
        torch.nn.init.zeros_(self.cnn_2.weight)
        self.cam_poses = np.load(REPO_DIR / "data_rendering/materials/cam_db_neoneo.npy")
        self.cam_irl_rel_extrinsic_base = np.loadtxt(
            REPO_DIR / "data_rendering/materials/cam_irL_rel_extrinsic_base.txt")
        self.cam_irr_rel_extrinsic_base = np.loadtxt(
            REPO_DIR / "data_rendering/materials/cam_irR_rel_extrinsic_base.txt")
        self.cam_irl_rel_extrinsic_hand = np.loadtxt(
            REPO_DIR / "data_rendering/materials/cam_irL_rel_extrinsic_hand.txt")
        self.cam_irr_rel_extrinsic_hand = np.loadtxt(
            REPO_DIR / "data_rendering/materials/cam_irR_rel_extrinsic_hand.txt")
        self.cam_ir_intrinsic_base = np.loadtxt(REPO_DIR / "data_rendering/materials/cam_ir_intrinsic_base.txt")
        self.cam_ir_intrinsic_redner_base = intrinsic_from_opencv(self.cam_ir_intrinsic_base, (1920, 1080),
                                                                  (self.img_width, self.img_height))
        self.cam_ir_intrinsic_hand = np.loadtxt(REPO_DIR / "data_rendering/materials/cam_ir_intrinsic_hand.txt")
        self.cam_ir_intrinsic_redner_hand = intrinsic_from_opencv(self.cam_ir_intrinsic_hand, (1920, 1080),
                                                                  (self.img_width, self.img_height))
        light_image = pyredner.imread(REPO_DIR / 'data_rendering/materials/d415-pattern-sq.png')
        # Convert light_image to current device
        self.light_image = light_image.to(pyredner.get_device())

        self.min_disp = 8
        self.max_disp = 64
        self.batch_size = 1024  # for parallel block matching
        table_pose_np = np.loadtxt(REPO_DIR / "data_rendering/materials/optical_table/pose.txt")
        self.table_pose = torch.tensor(Pose(table_pose_np[:3], table_pose_np[3:]).to_transformation_matrix()).to(
            pyredner.device).float()

    def load_weight(self, weight_path):
        state_dict = torch.load(weight_path, map_location="cpu")
        self.epsilon = torch.tensor([state_dict["eps"]], dtype=torch.float32, device=pyredner.get_device(),
                                    requires_grad=True)
        self.mu = torch.tensor([state_dict["mu"]], dtype=torch.float32, device=pyredner.get_device(),
                               requires_grad=True)
        self.sigma = torch.tensor([state_dict["sigma"]], dtype=torch.float32, device=pyredner.get_device(),
                                  requires_grad=True)
        self.beta = torch.tensor([state_dict["beta"]], dtype=torch.float32, device=pyredner.get_device(),
                                 requires_grad=True)

        self.cnn_1.load_state_dict(state_dict["cnn1"])
        self.cnn_2.load_state_dict(state_dict["cnn2"])

    def render(self, scene_idx, view_idx, padding=False):
        # load table
        table = pyredner.load_obj(REPO_DIR / "data_rendering/materials/optical_table/optical_table.obj",
                                  return_objects=True)[0]
        table.vertices = table.vertices.clone() @ torch.t(self.table_pose[:3, :3]) + self.table_pose[:3, 3]

        # load objects
        obj_poses = json.load(open(SCENE_DIR / f"{scene_idx}/input.json"))
        objects = [table, ]
        for obj_name, obj_pose in obj_poses.items():
            obj_pose = torch.tensor(obj_pose).to(pyredner.device).float()
            obj_path = OBJ_MODEL_DIR / obj_name / "visual_mesh.obj"
            single_object = pyredner.load_obj(obj_path, return_objects=True)
            # Obtain the object vertices we want to apply the transformation on.
            vertices = []
            for obj in single_object:
                vertices.append(obj.vertices.clone())
            # set object pose
            for obj, v in zip(single_object, vertices):
                obj.vertices = v @ torch.t(obj_pose[:3, :3]) + obj_pose[:3, 3]

            objects.extend(single_object)

        cam_pose = self.cam_poses[view_idx]
        cam2world = np.linalg.inv(cam_pose)
        if view_idx == 0:
            cam_irL_extrinsic = np.linalg.inv(np.linalg.inv(cam_pose) @ self.cam_irl_rel_extrinsic_base)
            cam_irR_extrinsic = np.linalg.inv(np.linalg.inv(cam_pose) @ self.cam_irr_rel_extrinsic_base)
            camera_ir = pyredner.Camera(
                cam_to_world=torch.tensor(np.linalg.inv(cam_irL_extrinsic)).float(),
                intrinsic_mat=torch.tensor(self.cam_ir_intrinsic_redner_base).float(),
                resolution=(self.img_height, self.img_width),
            )
            focal_length = self.cam_ir_intrinsic_base[0, 0] / 1920 * self.img_width
            baseline_length = 5.476e-2
        else:
            cam_irL_extrinsic = np.linalg.inv(np.linalg.inv(cam_pose) @ self.cam_irl_rel_extrinsic_hand)
            cam_irR_extrinsic = np.linalg.inv(np.linalg.inv(cam_pose) @ self.cam_irr_rel_extrinsic_hand)
            camera_ir = pyredner.Camera(
                cam_to_world=torch.tensor(np.linalg.inv(cam_irL_extrinsic)).float(),
                intrinsic_mat=torch.tensor(self.cam_ir_intrinsic_redner_hand).float(),
                resolution=(self.img_height, self.img_width),
            )
            focal_length = self.cam_ir_intrinsic_hand[0, 0] / 1920 * self.img_width
            baseline_length = 5.49e-2
        scene = pyredner.Scene(camera=camera_ir, objects=objects)
        active_light = ProjectionLight(
            position=torch.tensor(cam2world[:3, 3], dtype=torch.float32, device=pyredner.get_device()),
            look_at=torch.tensor(cam2world[:3, 3] + cam2world[:3, 2], dtype=torch.float32,
                                 device=pyredner.get_device()),
            up=torch.tensor(-cam2world[:3, 0], dtype=torch.float32, device=pyredner.get_device()),
            fov=torch.tensor([100.0], device=pyredner.get_device()),
            intensity=10 * self.light_image,
            scene=scene,
            epsilon=self.epsilon
        )
        ambient_light = pyredner.AmbientLight(torch.tensor([0.1, 0.02, 0.02]))
        clean_depth = pyredner.render_g_buffer(scene=scene, channels=[pyredner.channels.depth])[..., 0]
        img_irl = pyredner.render_deferred(scene=scene, lights=[active_light, ambient_light], alpha=False)[..., 0]
        shadow_map = active_light.shadow_map.permute(2, 0, 1).unsqueeze(0)
        camera_ir.cam_to_world = torch.tensor(np.linalg.inv(cam_irR_extrinsic)).float()
        img_irr = pyredner.render_deferred(scene=scene, lights=[active_light, ambient_light], alpha=False)[..., 0]

        # add noise
        img_irl = img_irl + torch.randn_like(img_irl) * self.sigma + self.mu
        img_irr = img_irr + torch.randn_like(img_irr) * self.sigma + self.mu
        if padding:
            img_irl = torch.cat(
                [torch.zeros(img_irl.shape[0], self.max_disp, device=pyredner.get_device(), dtype=img_irl.dtype),
                 img_irl,
                 # torch.zeros(img_irl.shape[0], self.max_disp, device=pyredner.get_device(), dtype=img_irl.dtype)
                 ],
                dim=1)
            img_irr = torch.cat(
                [torch.zeros(img_irr.shape[0], self.max_disp, device=pyredner.get_device(), dtype=img_irr.dtype),
                 img_irr,
                 # torch.zeros(img_irr.shape[0], self.max_disp, device=pyredner.get_device(), dtype=img_irr.dtype)
                 ],
                dim=1)

        disp = block_matching(
            img_irl,
            img_irr,
            min_disp=self.min_disp,
            max_disp=self.max_disp,
            block_size=9,
            temperature=self.beta,
            batch_size=self.batch_size
        )

        # Disparity to depth conversion
        stereo_depth = focal_length * baseline_length / disp.unsqueeze(
            0)  # Of shape [h, w-max_disp], viewed in left camera's frame
        shadow_map = F.interpolate(shadow_map, (self.img_height, self.img_width), mode="bilinear")[0]
        if padding:
            clean_depth = clean_depth.unsqueeze(0)
            output = self.cnn_2(self.cnn_1(torch.stack([stereo_depth, clean_depth, shadow_map], dim=1)))[
                         0] + stereo_depth
            output = output.detach().cpu().numpy()[0]
        else:
            clean_depth = clean_depth[:, self.max_disp:].unsqueeze(0)
            shadow_map = shadow_map[:, :, self.max_disp:]
            output = self.cnn_2(self.cnn_1(torch.stack([stereo_depth, clean_depth, shadow_map], dim=1)))[
                         0] + stereo_depth
            output = output.detach().cpu().numpy()[0]
            output = np.concatenate([np.zeros((output.shape[0], self.max_disp)), output], axis=1)

        return {
            "irl": img_irl.detach().cpu().numpy(),
            "irr": img_irr.detach().cpu().numpy(),
            "output": output
        }


if __name__ == '__main__':
    # Use GPU if available
    pyredner.set_use_gpu(torch.cuda.is_available())
    pyredner.set_print_timing(False)

    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--sub", type=int, required=True)
    parser.add_argument("--total", type=int, required=True)
    parser.add_argument("--target-root", type=str, required=True)
    parser.add_argument("--padding", type=bool, default=True)
    args = parser.parse_args()

    output_dir = args.target_root
    Path(output_dir).makedirs_p()

    timestamp = time.strftime("%y-%m-%d_%H-%M-%S")
    filename = Path(output_dir) / f"log.diff_retrieval.inferrence.sub{args.sub:02d}.tot{args.total:02d}.{timestamp}.txt"
    logger.remove()
    fmt = (
        f"<green>{{time:YYYY-MM-DD HH:mm:ss.SSS}}</green> | "
        f"<lvl>{{level}}</lvl> | "
        f"<lvl>{{message}}</lvl>"
    )

    # logger to file
    logger.add(filename, format=fmt)

    # logger to std stream
    logger.add(sys.stdout, format=fmt)

    diff_scene = DiffSceneInference(output_dir=Path(output_dir) / "data")
    diff_scene.load_weight(weight_path=(REPO_DIR / "diff_retrieval/EP002.pth"))

    if args.padding:
        depth_name = "0000_depth_diffpad.png"
    else:
        depth_name = "0000_depth_diff.png"

    sub_total_scene = 1000 // args.total
    if args.sub < args.total:
        scene_idx_list = np.arange(1000)[(args.sub - 1) * sub_total_scene: args.sub * sub_total_scene]
    else:
        scene_idx_list = np.arange(1000)[(args.sub - 1) * sub_total_scene:]
    for level in (0, 1):
        for scene_idx in scene_idx_list:
            for view_idx in range(21):
                curr_output_dir = Path(output_dir) / f"data/{level}-{scene_idx}-{view_idx}"
                if (curr_output_dir / depth_name[:-4] + "_colored.png").exists():
                    logger.info(f"skip {curr_output_dir}")
                    continue
                curr_output_dir.makedirs_p()
                render_result = diff_scene.render(f"{level}-{scene_idx}", view_idx, padding=args.padding)
                irl = np.clip(render_result["irl"] * 255, 0, 255).astype(np.uint8)
                irr = np.clip(render_result["irr"] * 255, 0, 255).astype(np.uint8)
                cv2.imwrite(curr_output_dir / "0000_irL_diff.png", irl)
                cv2.imwrite(curr_output_dir / "0000_irR_diff.png", irr)

                depth = render_result["output"] * 1000.0
                depth = cv2.resize(depth, (1920, 1080), interpolation=cv2.INTER_CUBIC)
                depth = depth.astype(np.uint16)
                cv2.imwrite(curr_output_dir / depth_name, depth)
                vis_depth = visualize_depth(depth)
                cv2.imwrite(curr_output_dir / depth_name[:-4] + "_colored.png", vis_depth)
                logger.info(f"{curr_output_dir} finished.")
