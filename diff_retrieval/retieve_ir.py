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


BASELINE_DATA_DIR = Path("/messytable-slow-vol/messy-table-dataset/baseline_images")
BASELINE_POSE_DIR = Path("/rayc-fast/ICCV2021_Diagnosis/ocrtoc_materials/baseline_poses")
OBJ_MODEL_DIR = Path("/rayc-fast/ICCV2021_Diagnosis/ocrtoc_materials/models")
REPO_DIR = Path("/jianyu-fast-vol/mt-pyredner")

OBJECTS = []
objects_info_csv = csv.DictReader(open("/rayc-fast/ICCV2021_Diagnosis/ocrtoc_materials/objects.csv"))
for idx, row in enumerate(objects_info_csv):
    info = dict(row)
    object_name = info['object']
    OBJECTS.append(object_name)


def intrinsic_from_opencv(intrinsic_cv, cv_resolution, redner_resolution):
    w, h = redner_resolution
    w_cv, h_cv = cv_resolution
    f_x, f_y = intrinsic_cv[0, 0], intrinsic_cv[1, 1]
    c_x, c_y = intrinsic_cv[0, 2], intrinsic_cv[1, 2]

    redner_intrinsic = np.eye(3)
    redner_intrinsic[0, 0] = 2 * f_x / w_cv
    redner_intrinsic[0, 2] = 2 * c_x / w_cv - 1
    redner_intrinsic[1, 1] = -(2 * f_y * h) / (w * h_cv)
    redner_intrinsic[1, 2] = h / w - (2 * h * c_y) / (w * h_cv)
    return redner_intrinsic


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


class ProjectionLight(pyredner.DeferredLight):
    def __init__(self,
                 position: torch.Tensor,
                 look_at: torch.Tensor,
                 up: torch.Tensor,
                 fov: torch.Tensor,
                 intensity: torch.Tensor,
                 scene: pyredner.Scene,
                 epsilon: torch.Tensor,
                 aa_samples: int = 2):
        self.position = position
        self.look_at = look_at
        self.up = up
        self.look_at_matrix = torch.inverse(pyredner.gen_look_at_matrix(position, look_at, up)).contiguous()
        self.look_at_matrix = self.look_at_matrix.to(pyredner.get_device())
        self.fov = fov
        self.fov_factor = 1.0 / torch.tan(pyredner.radians(0.5 * fov))
        self.fov_factor = self.fov_factor.to(pyredner.get_device())
        self.intensity = intensity.to(pyredner.get_device()).permute(2, 0, 1).unsqueeze(0)
        self.scene = scene
        self.aa_samples = aa_samples
        self.epsilon = epsilon
        self.shadow_map = None

    def render(self,
               position: torch.Tensor,
               normal: torch.Tensor,
               albedo: torch.Tensor):
        # Transform position to the light source's local space
        flattened_position = position.view(-1, 3)
        flattened_position = torch.cat(
            (flattened_position, torch.ones(flattened_position.shape[0], 1, device=pyredner.get_device())), dim=1)
        local = flattened_position @ torch.transpose(self.look_at_matrix, 0, 1)
        local = local / local[:, 3:4]
        local = local[:, 0:3]
        # Project from local space to screen space
        local_fov_scaled = torch.stack((local[:, 0] * self.fov_factor,
                                        local[:, 1] * self.fov_factor,
                                        local[:, 2]), axis=1)
        local_fov_scaled = local_fov_scaled / local_fov_scaled[:, 2:3]
        image_samples = local_fov_scaled[:, 0:2]
        image_samples = image_samples.view(1, position.shape[0], position.shape[1], 2)
        # Now we have a list of 2D points, we want to find the corresponding
        # positions on the input image
        intensity = torch.nn.functional.grid_sample(self.intensity, image_samples)
        # NCHW -> HWC
        intensity = intensity.permute(0, 2, 3, 1).squeeze()

        # Compute the distance squared term and the material response
        light_dir = self.position - position
        light_dist_sq = torch.sum(light_dir * light_dir, dim=-1, keepdim=True)
        light_dist = torch.sqrt(light_dist_sq)
        # Normalize light direction
        light_dir = light_dir / light_dist
        dot_l_n = torch.sum(light_dir * normal, dim=-1, keepdim=True)
        dot_l_n = torch.max(dot_l_n, torch.zeros_like(dot_l_n))
        img = intensity * dot_l_n * (albedo / math.pi) / light_dist_sq

        # ============= CODE FOR SHADOW MAPPING =============

        # Computing depth map as seen from projector:
        old_cam = self.scene.camera
        self.scene.camera = pyredner.Camera(self.position, self.look_at, self.up, self.fov, resolution=(1024, 1024))
        projector_img = pyredner.render_g_buffer(scene=self.scene, channels=[pyredner.channels.depth])
        self.scene.camera = old_cam
        depth_map_from_projector = projector_img[..., 0]
        # Set pixels with depth = 0 to a large number (they are infinitely far away)
        depth_map_from_projector = torch.where(depth_map_from_projector == 0.0,
                                               torch.ones_like(depth_map_from_projector) * 1000.0,
                                               depth_map_from_projector)

        # Sampling the projector depth map to map to the visible elements:
        # Flip the y axis coordinate for depth map sampling
        shadow_image_samples = torch.stack([image_samples[:, :, :, 0], -image_samples[:, :, :, 1]], axis=-1)
        depth_map_from_projector_ = torch.nn.functional.grid_sample( \
            depth_map_from_projector.unsqueeze(0).unsqueeze(0), shadow_image_samples)
        depth_map_from_projector_ = depth_map_from_projector_.squeeze()

        image_samples_3channels = local_fov_scaled[..., :3].view(position.shape[0], position.shape[1], 3)
        image_samples_3channels[..., 2] = -1.0  # grid_samples values are from -1 to +1
        sample_vis = image_samples_3channels * .5 + .5
        h, w = image_samples.shape[1:3]
        xv, yv = torch.meshgrid([torch.arange(0., h), torch.arange(0., w)])
        meshgrid = torch.stack((xv / h, yv / w, torch.zeros_like(xv)), dim=-1)

        # Getting distance from positions in image to projector (depth to projector for objects visible to the camera):
        distance_to_projector = local[..., 2].view(depth_map_from_projector_.shape)

        # Computing shadow map:
        # shadow = (depth_map_from_projector_ + bias < distance_to_projector).unsqueeze(-1).float()
        shadow = torch.sigmoid(distance_to_projector - depth_map_from_projector_ - self.epsilon).unsqueeze(-1)
        self.shadow_map = shadow

        # Applying shadow map to image:
        img = img * (1.0 - shadow)

        return img


class DiffScene:
    def __init__(self):
        self.img_height, self.img_width = 360, 640
        self.epsilon = torch.tensor([1e-3], dtype=torch.float32, device=pyredner.get_device(), requires_grad=True)
        #self.mu = torch.tensor([0.0], dtype=torch.float32, device=pyredner.get_device(), requires_grad=True)
        #self.sigma = torch.tensor([1e-3], dtype=torch.float32, device=pyredner.get_device(), requires_grad=True)
        #self.beta = torch.tensor([20], dtype=torch.float32, device=pyredner.get_device(), requires_grad=True)

        #self.cnn_1 = nn.Sequential(nn.Conv2d(3, 32, 5, stride=1, padding=2, bias=False),
        #                           nn.ReLU(),
        #                           ).cuda()
        #self.cnn_2 = nn.Conv2d(32, 1, 1, bias=False).cuda()
        #torch.nn.init.zeros_(self.cnn_2.weight)

        self.path_list = self._gen_path_list()
        self.cam_poses = np.load(REPO_DIR / "data_rendering/materials/cam_db_neoneo.npy")
        self.cam_irl_rel_extrinsic_hand = np.loadtxt(
            REPO_DIR / "data_rendering/materials/cam_irL_rel_extrinsic_hand.txt")
        self.cam_irr_rel_extrinsic_hand = np.loadtxt(
            REPO_DIR / "data_rendering/materials/cam_irR_rel_extrinsic_hand.txt")
        self.cam_ir_intrinsic = np.loadtxt(REPO_DIR / "data_rendering/materials/cam_ir_intrinsic_hand.txt")
        self.cam_ir_intrinsic_redner = intrinsic_from_opencv(self.cam_ir_intrinsic, (1920, 1080),
                                                             (self.img_width, self.img_height))
        light_image = pyredner.imread(REPO_DIR / 'data_rendering/materials/d415-pattern-sq.png')
        # Convert light_image to current device
        self.light_image = light_image.to(pyredner.get_device())
        self.light_image = torch.zeros_like(self.light_image, requires_grad=True)

        self.min_disp = 8
        self.max_disp = 64
        self.focal_length = self.cam_ir_intrinsic[0, 0] / 1920 * self.img_width
        self.baseline_length = 5.49e-2
        self.min_depth = self.focal_length * self.baseline_length / self.max_disp
        self.max_depth = self.focal_length * self.baseline_length / self.min_disp

        self.batch_size = 1024  # for parallel block matching
        table_pose_np = np.loadtxt(REPO_DIR / "data_rendering/materials/optical_table/pose.txt")
        self.table_pose = torch.tensor(Pose(table_pose_np[:3], table_pose_np[3:]).to_transformation_matrix()).to(
            pyredner.device).float()

    def render(self, idx):
        paths = self.path_list[idx]
        # load table
        table = pyredner.load_obj(REPO_DIR / "data_rendering/materials/optical_table/optical_table.obj",
                                  return_objects=True)[0]
        table.vertices = table.vertices.clone() @ torch.t(self.table_pose[:3, :3]) + self.table_pose[:3, 3]

        # load object
        objects = pyredner.load_obj(paths["obj"], return_objects=True)
        # Obtain the object vertices we want to apply the transformation on.
        vertices = []
        for obj in objects:
            vertices.append(obj.vertices.clone())

        # set object pose
        pose = torch.tensor(np.array(paths["pose"])).to(pyredner.get_device()).float()
        for obj, v in zip(objects, vertices):
            obj.vertices = v + pose[:3, 3]

        objects.append(table)

        cam_pose = self.cam_poses[paths["view_idx"]]
        cam2world = np.linalg.inv(cam_pose)
        cam_irL_extrinsic = np.linalg.inv(np.linalg.inv(cam_pose) @ self.cam_irl_rel_extrinsic_hand)
        cam_irR_extrinsic = np.linalg.inv(np.linalg.inv(cam_pose) @ self.cam_irr_rel_extrinsic_hand)

        camera_ir = pyredner.Camera(
            cam_to_world=torch.tensor(np.linalg.inv(cam_irL_extrinsic)).float(),
            intrinsic_mat=torch.tensor(self.cam_ir_intrinsic_redner).float(),
            resolution=(self.img_height, self.img_width),
        )

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
        #img_irl = img_irl + torch.randn_like(img_irl) * self.sigma + self.mu
        #img_irr = img_irr + torch.randn_like(img_irr) * self.sigma + self.mu
        """
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
        stereo_depth = self.focal_length * self.baseline_length / disp.unsqueeze(
            0)  # Of shape [h, w-max_disp], viewed in left camera's frame
        clean_depth = clean_depth[:, self.max_disp:].unsqueeze(0)
        shadow_map = F.interpolate(shadow_map, (self.img_height, self.img_width), mode="bilinear")[0]
        shadow_map = shadow_map[:, :, self.max_disp:]

        output = self.cnn_2(self.cnn_1(torch.stack([stereo_depth, clean_depth, shadow_map], dim=1)))[0] + stereo_depth
        """
        # load real input
        real_depth = cv2.imread(
            paths["real_depth"],
            cv2.IMREAD_UNCHANGED)

        
        real_depth = (real_depth.astype(float)) / 1000.0
        real_depth = cv2.resize(real_depth, (self.img_width, self.img_height),
                                interpolation=cv2.INTER_NEAREST)
        real_depth_grad = np.load(
            paths["real_depth_grad"])
        real_depth_grad = cv2.resize(real_depth_grad, (self.img_width, self.img_height),
                                     interpolation=cv2.INTER_NEAREST)
        gt_irl = cv2.imread(paths["gt_irl"], cv2.IMREAD_UNCHANGED)
        gt_irl = cv2.resize(gt_irl, (self.img_width, self.img_height),
                                interpolation=cv2.INTER_NEAREST)
        gt_irl = torch.tensor(gt_irl).to(pyredner.device).float()

        mask = cv2.imread(paths["mask"], 0)
        mask = cv2.resize(mask, (self.img_width, self.img_height), interpolation=cv2.INTER_NEAREST)
        mask = mask < 10
        #print(type(gt_irl), type(img_irl), gt_irl.shape, img_irl.shape)
        #assert 1==0
        return {
            #"clean_depth": clean_depth,
            #"stereo_depth": stereo_depth,
            #"shadow_map": shadow_map,
            "gt_irl": gt_irl,
            #"post_depth": output,
            "irl": img_irl,
            "irr": img_irr,
            "real_depth": torch.tensor(real_depth).to(pyredner.device).float(),
            "real_depth_grad": torch.tensor(real_depth_grad).to(pyredner.device).float(),
            "mask": torch.tensor(mask).to(pyredner.device).float(),
        }

    def _gen_path_list(self):
        path_list = []
        for obj in OBJECTS:
            for pose_idx in range(1):
                for view_idx in range(1, 2):
                    pose = json.load(open(BASELINE_POSE_DIR / f"baseline_{obj}_{pose_idx:02d}/input.json"))[obj]

                    paths = {
                        "obj_name": obj,
                        "obj": OBJ_MODEL_DIR / obj / "visual_mesh.obj",
                        "pose": pose,
                        "view_idx": view_idx,
                        "real_depth": BASELINE_DATA_DIR / f"baseline_{obj}_{pose_idx:02d}-{view_idx}/1024_depthL_real.png",
                        "real_depth_grad": BASELINE_DATA_DIR / f"baseline_{obj}_{pose_idx:02d}-{view_idx}/1024_depthL_grad_real.npy",
                        "mask": BASELINE_DATA_DIR / f"baseline_{obj}_{pose_idx:02d}-{view_idx}/maskL.png",
                        "gt_irl": BASELINE_DATA_DIR / f"baseline_{obj}_{pose_idx:02d}-{view_idx}/0128_irL_kuafuv2_half_6.png"
                    }
                    path_list.append(paths)

        return path_list

    def __len__(self):
        return len(self.path_list)


if __name__ == "__main__":
    # Use GPU if available
    torch.autograd.set_detect_anomaly(True)
    pyredner.set_use_gpu(torch.cuda.is_available())
    pyredner.set_print_timing(False)

    timestamp = time.strftime("%y-%m-%d_%H-%M-%S")
    filename = f"log.diff_retrieval.{timestamp}.txt"
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

    # hyper parameters
    NUM_EPOCHS = 10
    huber_delta = 0.03
    GRAD_WEIGHT = 1e-3

    huber_loss_fn = nn.HuberLoss(delta=huber_delta)
    sobel = Sobel().to(pyredner.device)
    diff_scene = DiffScene()
    num_sample = len(diff_scene)


    def compute_loss(render_dict):
        mask = render_dict["mask"].float() + (render_dict["real_depth"] < diff_scene.min_depth).float() + (render_dict[
                                                                                                               "real_depth"] > diff_scene.max_depth).float()
        mask = (mask > 0)[:, diff_scene.max_disp:]
        depth_grad = sobel(render_dict["post_depth"]).permute(1, 2, 0)
        grad_loss = F.l1_loss(depth_grad[mask], render_dict["real_depth_grad"][:, diff_scene.max_disp:][mask])
        depth_loss = huber_loss_fn(render_dict["post_depth"][0][mask],
                                   render_dict["real_depth"][:, diff_scene.max_disp:][mask])
        return depth_loss, grad_loss

    def compute_loss_img(render_dict):
        mse_loss = F.mse_loss(render_dict["gt_irl"], render_dict["irl"])

        return mse_loss


    optimizer = torch.optim.Adam(
        [diff_scene.light_image, diff_scene.epsilon],
        lr=0.001)

    logger.info(
        f"eps: {diff_scene.epsilon.item():.2f};")

    for epoch_idx in range(1, NUM_EPOCHS + 1):
        loss_depth = 0.0
        loss_grad = 0.0
        loss_total = 0.0
        idx_order = np.arange(num_sample)
        np.random.shuffle(idx_order)
        for i in tqdm(range(num_sample)):
            optimizer.zero_grad()
            render_dict = diff_scene.render(idx_order[i])
            render_loss = compute_loss_img(render_dict)

            render_loss.backward()
            optimizer.step()
            loss_total += render_loss.item()
            if i % 20 == 0:
                logger.info(
                    f"iter: {i:4d} loss_total: {loss_total / (i + 1):.3f}, loss_depth: {loss_depth / (i + 1):.3f},"
                    f" loss_grad: {loss_grad / (i + 1):.3f}")
        loss_total /= num_sample
        loss_depth /= num_sample
        loss_grad /= num_sample
        logger.info(
            f"EPOCH: {epoch_idx:3d} loss_total: {loss_total:.3f}, loss_depth: {loss_depth:.3f}, loss_grad: {loss_grad:.3f}")
        logger.info(
            f"eps: {diff_scene.epsilon.item():.2f}; mu: {diff_scene.mu.item():.2f}; sigma: {diff_scene.sigma.item():.2f}; beta: {diff_scene.beta.item():.2f}")
        logger.info("===================================================")

        torch.save(
            {
                "eps": diff_scene.epsilon,
                "mu": diff_scene.mu,
                "sigma": diff_scene.sigma,
                "beta": diff_scene.beta,
                "cnn1": diff_scene.cnn_1.state_dict(),
                "cnn2": diff_scene.cnn_2.state_dict(),
            },
            f"EP{epoch_idx:03d}.pth"
        )
        # cv2.imwrite("clean_depth.png", visualize_depth(render_dict["clean_depth"][0].cpu().numpy()))
        # cv2.imwrite("stereo_depth.png", visualize_depth(render_dict["stereo_depth"][0].detach().cpu().numpy()))
        # cv2.imwrite("post_depth.png", visualize_depth(render_dict["post_depth"][0].detach().cpu().numpy()))
        # plt.imsave("shadow.png", render_dict["shadow_map"][0].detach().cpu().numpy())

    logger.info(
        f"eps: {diff_scene.epsilon.item():.2f}; mu: {diff_scene.mu.item():.2f}; sigma: {diff_scene.sigma.item():.2f}; beta: {diff_scene.beta.item():.2f}")
