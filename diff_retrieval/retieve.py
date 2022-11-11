import pyredner
import torch
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

BASELINE_DATA_DIR = "/media/DATA/LINUX_DATA/ICCV2021_Diagnosis/baseline_images"
BASELINE_POSE_DIR = "/home/rayu/Projects/ICCV2021_Diagnosis/ocrtoc_materials/baseline_poses"
# SEARCH_TARGET = "cellphone"
SEARCH_TARGET = "camera"
OBJ_MODEL_DIR = Path("/home/rayu/Projects/ICCV2021_Diagnosis/ocrtoc_materials/models")

VIEW_IDX = 14
REPO_DIR = Path("/home/rayu/Projects/active_zero2")


class ProjectionLight(pyredner.DeferredLight):
    def __init__(self,
                 position: torch.Tensor,
                 look_at: torch.Tensor,
                 up: torch.Tensor,
                 fov: torch.Tensor,
                 intensity: torch.Tensor,
                 scene: pyredner.Scene,
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
        self.epsilon = torch.tensor([10], dtype=torch.float32, device=pyredner.get_device(), requires_grad=True)

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
        old_cam = scene.camera
        scene.camera = pyredner.Camera(self.position, self.look_at, self.up, self.fov, resolution=(1024, 1024))
        projector_img = pyredner.render_g_buffer(scene=scene, channels=[pyredner.channels.depth])
        scene.camera = old_cam
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

        # Applying shadow map to image:
        img = img * (1.0 - shadow)

        return img


if __name__ == "__main__":
    # Use GPU if available
    pyredner.set_use_gpu(torch.cuda.is_available())
    # load table
    table = pyredner.load_obj(REPO_DIR / "data_rendering/materials/optical_table/optical_table.obj",
                              return_objects=True)[0]
    table_pose_np = np.loadtxt(REPO_DIR / "data_rendering/materials/optical_table/pose.txt")
    table_pose = torch.tensor(Pose(table_pose_np[:3], table_pose_np[3:]).to_transformation_matrix()).to(
        pyredner.device).float()
    table.vertices = table.vertices.clone() @ torch.t(table_pose[:3, :3]) + table_pose[:3, 3]

    objects = pyredner.load_obj(OBJ_MODEL_DIR / SEARCH_TARGET / "visual_mesh.obj", return_objects=True)
    # Obtain the object vertices we want to apply the transformation on.
    vertices = []
    for obj in objects:
        vertices.append(obj.vertices.clone())

    # set object pose
    pose = json.load(open(os.path.join(BASELINE_POSE_DIR, f"baseline_{SEARCH_TARGET}_00/input.json")))[SEARCH_TARGET]
    pose = torch.tensor(np.array(pose)).to(pyredner.get_device()).float()
    # Shift the vertices to the center, apply rotation matrix,
    # shift back to the original space, then apply the translation.
    for obj, v in zip(objects, vertices):
        obj.vertices = v + pose[:3, 3]

    objects.append(table)
    cam_poses = np.load(REPO_DIR / "data_rendering/materials/cam_db_neoneo.npy")
    cam_pose = cam_poses[VIEW_IDX]
    cam2world = np.linalg.inv(cam_pose)

    cam_irl_rel_extrinsic_hand = np.loadtxt(REPO_DIR / "data_rendering/materials/cam_irL_rel_extrinsic_hand.txt")
    cam_irr_rel_extrinsic_hand = np.loadtxt(REPO_DIR / "data_rendering/materials/cam_irR_rel_extrinsic_hand.txt")
    cam_irL_extrinsic = np.linalg.inv(np.linalg.inv(cam_pose) @ cam_irl_rel_extrinsic_hand)
    cam_irR_extrinsic = np.linalg.inv(np.linalg.inv(cam_pose) @ cam_irr_rel_extrinsic_hand)
    cam_ir_intrinsic = np.loadtxt(REPO_DIR / "data_rendering/materials/cam_ir_intrinsic_hand.txt")


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


    cam_ir_intrinsic_redner = intrinsic_from_opencv(cam_ir_intrinsic, (1920, 1080), (1280, 720))

    camera_ir = pyredner.Camera(
        cam_to_world=torch.tensor(np.linalg.inv(cam_irL_extrinsic)).float(),
        intrinsic_mat=torch.tensor(cam_ir_intrinsic_redner).float(),
        resolution=(720, 1280),
    )

    scene = pyredner.Scene(camera=camera_ir, objects=objects)
    light_image = pyredner.imread(REPO_DIR / 'data_rendering/materials/d415-pattern-sq.png')
    # Convert light_image to current device
    light_image = light_image.to(pyredner.get_device())
    active_light = ProjectionLight(
        position=torch.tensor(cam2world[:3, 3], dtype=torch.float32, device=pyredner.get_device()),
        look_at=torch.tensor(cam2world[:3, 3] + cam2world[:3, 2], dtype=torch.float32, device=pyredner.get_device()),
        up=torch.tensor(-cam2world[:3, 0], dtype=torch.float32, device=pyredner.get_device()),
        fov=torch.tensor([100.0], device=pyredner.get_device()),
        intensity=10 * light_image,
        scene=scene
    )

    ambient_light = pyredner.AmbientLight(torch.tensor([0.1, 0.02, 0.02]))
    img_irl = pyredner.render_deferred(scene=scene, lights=[active_light, ambient_light], alpha=True)
    img_irl = torch.pow(img_irl, 1.0 / 2.2).detach().cpu().numpy()
    img_irl = np.clip(img_irl * 255, 0, 255).astype(np.uint8)
    img_irl = cv2.cvtColor(img_irl, cv2.COLOR_RGBA2GRAY)
    # img_irl = cv2.resize(img_irl, (1280, 720))
    cv2.imwrite("irl.png", img_irl)

    camera_ir.cam_to_world = torch.tensor(np.linalg.inv(cam_irR_extrinsic)).float()
    img_irr = pyredner.render_deferred(scene=scene, lights=[active_light, ambient_light], alpha=True)
    img_irr = torch.pow(img_irr, 1.0 / 2.2).detach().cpu().numpy()
    img_irr = np.clip(img_irr * 255, 0, 255).astype(np.uint8)
    img_irr = cv2.cvtColor(img_irr, cv2.COLOR_RGBA2GRAY)
    # img_irr = cv2.resize(img_irr, (1280, 720))
    cv2.imwrite("irr.png", img_irr)
