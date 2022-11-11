import sapien.core as sapien
from sapien.utils import Viewer
import numpy as np
from sapien.core import Pose
import os
from path import Path
import json
import transforms3d as t3d
from PIL import Image, ImageColor
import cv2
from simsense import DepthSensor
import matplotlib.pyplot as plt
from data_rendering.utils.sim_depth import sim_ir_noise

OBJECT_DIR = "/home/rayu/Projects/ICCV2021_Diagnosis/ocrtoc_materials/models/"
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


def load_obj_vk(scene, obj_name, pose=Pose(), is_kinematic=False):
    builder = scene.create_actor_builder()
    builder.add_visual_from_file(os.path.join(OBJECT_DIR, obj_name, "visual_mesh.obj"))
    builder.add_multiple_collisions_from_file(os.path.join(OBJECT_DIR, obj_name, "collision_mesh.obj"))
    if is_kinematic:
        obj = builder.build_kinematic(name=obj_name)
    else:
        obj = builder.build(name=obj_name)
    obj.set_pose(pose)
    return obj


def load_mesh_list(builder, mesh_list_file, renderer, material_name):
    mesh_dir = Path(mesh_list_file).dirname()
    fmesh = open(mesh_list_file, "r")
    for l in fmesh.readlines():
        mesh_name = l.strip()
        mesh_path = mesh_dir / mesh_name
        kuafu_material_path = mesh_dir / f"{mesh_name[:-4]}_{material_name}.json"
        if kuafu_material_path.exists():
            obj_material = load_kuafu_material(kuafu_material_path, renderer)
            builder.add_visual_from_file(mesh_path, material=obj_material)
        else:
            builder.add_visual_from_file(mesh_path)


def load_kuafu_material(json_path, renderer: sapien.KuafuRenderer):
    with open(json_path, "r") as js:
        material_dict = json.load(js)

    js_dir = Path(json_path).dirname()
    object_material = renderer.create_material()
    object_material.set_base_color(material_dict["base_color"])
    if material_dict["diffuse_tex"]:
        object_material.set_diffuse_texture_from_file(str(js_dir / material_dict["diffuse_tex"]))
    object_material.set_emission(material_dict["emission"])
    object_material.set_ior(material_dict["ior"])
    object_material.set_metallic(material_dict["metallic"])
    if material_dict["metallic_tex"]:
        object_material.set_metallic_texture_from_file(str(js_dir / material_dict["metallic_tex"]))
    object_material.set_roughness(material_dict["roughness"])
    if material_dict["roughness_tex"]:
        object_material.set_roughness_texture_from_file(str(js_dir / material_dict["roughness_tex"]))
    object_material.set_specular(material_dict["specular"])
    object_material.set_transmission(material_dict["transmission"])
    if material_dict["transmission_tex"]:
        object_material.set_transmission_texture_from_file(str(js_dir / material_dict["transmission_tex"]))

    return object_material


def load_obj(scene, obj_name, renderer, pose=Pose(), is_kinematic=False, material_name="kuafu_material"):
    builder = scene.create_actor_builder()
    kuafu_material_path = os.path.join(OBJECT_DIR, obj_name, f"{material_name}.json")
    mesh_list_file = os.path.join(OBJECT_DIR, obj_name, "visual_mesh_list.txt")
    if os.path.exists(mesh_list_file):
        load_mesh_list(builder, mesh_list_file, renderer, material_name)
    elif os.path.exists(kuafu_material_path):
        obj_material = load_kuafu_material(kuafu_material_path, renderer)
        builder.add_visual_from_file(os.path.join(OBJECT_DIR, obj_name, "visual_mesh.obj"), material=obj_material)
    else:
        builder.add_visual_from_file(os.path.join(OBJECT_DIR, obj_name, "visual_mesh.obj"))
    builder.add_multiple_collisions_from_file(os.path.join(OBJECT_DIR, obj_name, "collision_mesh.obj"))
    if is_kinematic:
        obj = builder.build_kinematic(name=obj_name)
    else:
        obj = builder.build(name=obj_name)
    obj.set_pose(pose)

    return obj


def load_objects(scene, renderer, vulkan):
    object_poses = {
        "tennis_ball": sapien.Pose([-0.19, -0.19, 1.06]),
        "voss": sapien.Pose([-0.25, 0.01, 0.85]),
        "spellegrino": sapien.Pose([-0.34, -0.30, 0.85], t3d.quaternions.mat2quat(t3d.euler.euler2mat(0, 0, -1.57))),
        "coffee_cup": sapien.Pose([-0.35, -0.13, 0.77], t3d.quaternions.mat2quat(t3d.euler.euler2mat(0, 0, 3.6))),
        "rubik": sapien.Pose([-0.24, -0.36, 1.08], t3d.quaternions.mat2quat(t3d.euler.euler2mat(1.57, 3.14, 5.0))),
    }
    for obj_name, obj_pose in object_poses.items():
        if vulkan:
            load_obj_vk(scene, obj_name, obj_pose, is_kinematic=True)
        else:
            load_obj(scene, obj_name, renderer, obj_pose, is_kinematic=True, material_name="kuafu_material_new2")


def main():
    engine = sapien.Engine()  # Create a physical simulation engine
    renderer = sapien.VulkanRenderer()  # Create a Vulkan renderer
    engine.set_renderer(renderer)  # Bind the renderer and the engine

    scene = engine.create_scene()  # Create an instance of simulation world (aka scene)
    scene.set_timestep(1 / 100.0)  # Set the simulation frequency

    # NOTE: How to build actors (rigid bodies) is elaborated in create_actors.py
    ground_material = renderer.create_material()
    ground_color = np.array([119, 146, 198, 256]) / 256
    ground_material.set_base_color(ground_color)
    ground_material.set_specular(0.5)
    scene.add_ground(altitude=0, render_material=ground_material)  # Add a ground

    # chair
    chair_loader = scene.create_urdf_loader()
    chair_loader.fix_root_link = True
    chair_loader.scale = 0.7

    urdf_path = "/home/rayu/Projects/active_zero2/scripts/large_scene/partnet-mobility-dataset/37825/mobility.urdf"
    chair_loader.load_multiple_collisions_from_file = True
    chair = chair_loader.load(str(urdf_path))
    chair.set_name("chair")
    chair.set_pose(Pose([-0.9, -0.8, 0.50], [0.65998315, 0., 0., -0.75128041]))

    # table
    table_loader = scene.create_urdf_loader()
    table_loader.fix_root_link = True
    table_loader.scale = 0.96

    urdf_path = "/home/rayu/Projects/active_zero2/scripts/large_scene/partnet-mobility-dataset/30869/mobility.urdf"
    table_loader.load_multiple_collisions_from_file = True
    table = table_loader.load(str(urdf_path))
    table.set_name("table")
    table.set_pose(Pose([0, 0, 0.61]))

    # cabinet
    cabinet_loader = scene.create_urdf_loader()
    cabinet_loader.fix_root_link = True

    urdf_path = "/home/rayu/Projects/active_zero2/scripts/large_scene/2/mobility_textures.urdf"
    cabinet_loader.load_multiple_collisions_from_file = True
    cabinet = cabinet_loader.load(str(urdf_path))
    cabinet.set_name("cabinet")
    cabinet.set_pose(Pose([-0.06, -0.58, 0.75], t3d.quaternions.mat2quat(t3d.euler.euler2mat(0, 0, 1.2))))
    cabinet.set_qpos([0.0, 1.8])

    # actors
    load_objects(scene, renderer, vulkan=True)
    # Add some lights so that you can observe the scene
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

    viewer = Viewer(renderer)  # Create a viewer (window)
    viewer.set_scene(scene)  # Bind the viewer and the scene

    # The coordinate frame in Sapien is: x(forward), y(left), z(upward)
    # The principle axis of the camera is the x-axis
    viewer.set_camera_xyz(x=-1.5, y=0.2, z=1.7)
    # The rotation of the free camera is represented as [roll(x), pitch(-y), yaw(-z)]
    # The camera now looks at the origin
    viewer.set_camera_rpy(r=0, p=-np.arctan2(2, 4), y=0.5)
    viewer.window.set_camera_parameters(near=0.05, far=100, fovy=1)

    near, far = 0.1, 100
    width, height = 1280, 720
    camera_mount_actor = scene.create_actor_builder().build_kinematic()
    camera = scene.add_mounted_camera(
        name="camera",
        actor=camera_mount_actor,
        pose=sapien.Pose(),  # relative to the mounted actor
        width=width,
        height=height,
        fovy=np.deg2rad(35),
        near=near,
        far=far,
    )
    camera.set_perspective_parameters(0.1, 100, 893.708, 893.708, 632.56, 369.403, 0.0)

    print('Intrinsic matrix\n', camera.get_intrinsic_matrix())

    # Compute the camera pose by specifying forward(x), left(y) and up(z)
    cam_pos = np.array([-1.1, 0.4, 1.7])
    # forward = -cam_pos / np.linalg.norm(cam_pos)
    # left = np.cross([0, 0, 1], forward)
    # left = left / np.linalg.norm(left)
    # up = np.cross(forward, left)
    mat44 = np.eye(4)
    # mat44[:3, :3] = np.stack([forward, left, up], axis=1)
    mat44[:3, :3] = t3d.euler.euler2mat(0, np.pi / 4.5, -np.pi / 3)
    mat44[:3, 3] = cam_pos
    camera_mount_actor.set_pose(sapien.Pose.from_transformation_matrix(mat44))

    scene.update_render()
    camera.take_picture()
    rgba = camera.get_float_texture('Color')  # [H, W, 4]
    rgba = rgba[..., :3] * 255
    cv2.imwrite('color.png', (rgba[..., ::-1].astype(np.uint8)))

    real_image = cv2.imread('/home/rayu/Projects/active_zero2/scripts/large_scene/real/10092_rgb_Color.png')
    mix_image = np.clip((real_image.astype(float) * 0.5 + rgba * 0.5), 0, 255).astype(np.uint8)
    cv2.imwrite('mixed.png', mix_image[..., ::-1])

    while not viewer.closed:  # Press key q to quit
        scene.step()  # Simulate the world
        scene.update_render()  # Update the world to the renderer
        viewer.render()


def create_realsense_d415(camera_name: str, camera_mount: sapien.ActorBase, scene: sapien.Scene, camera_k, camera_ir_k):
    scene.update_render()
    fov = 0.742501437664032
    name = camera_name
    width = 1280
    height = 720

    tran_pose0 = sapien.Pose([0, 0, 0])
    if "base" in camera_name:
        tran_pose1 = sapien.Pose(
            [-0.0008183810985, -0.0173809196, -0.002242552045],
            [9.99986449e-01, 5.69235052e-04, 1.23234267e-03, -5.02592655e-03],
        )
        tran_pose2 = sapien.Pose(
            [-0.0008183810985, -0.07214373, -0.002242552045],
            [9.99986449e-01, 5.69235052e-04, 1.23234267e-03, -5.02592655e-03],
        )
    else:
        tran_pose1 = sapien.Pose(
            [0.0002371878611, -0.0153303356, -0.002143536015],
            [0.9999952133080734, 0.0019029481504852343, -0.0003405963365571751, -0.0024158111293426307],
        )
        tran_pose2 = sapien.Pose(
            [0.0002371878611, -0.0702470843, -0.002143536015],
            [0.9999952133080734, 0.0019029481504852343, -0.0003405963365571751, -0.0024158111293426307],
        )

    camera0 = scene.add_mounted_camera(f"{name}", camera_mount, tran_pose0, width, height, 0, fov, 0.001, 100)
    camera0.set_perspective_parameters(
        0.1, 100.0, camera_k[0, 0], camera_k[1, 1], camera_k[0, 2], camera_k[1, 2], camera_k[0, 1]
    )

    camera1 = scene.add_mounted_camera(f"{name}_left", camera_mount, tran_pose1, width, height, 0, fov, 0.001, 100)
    camera1.set_perspective_parameters(
        0.1, 100.0, camera_ir_k[0, 0], camera_ir_k[1, 1], camera_ir_k[0, 2], camera_ir_k[1, 2], camera_ir_k[0, 1]
    )
    camera2 = scene.add_mounted_camera(f"{name}_right", camera_mount, tran_pose2, width, height, 0, fov, 0.001, 100)
    camera2.set_perspective_parameters(
        0.1, 100.0, camera_ir_k[0, 0], camera_ir_k[1, 1], camera_ir_k[0, 2], camera_ir_k[1, 2], camera_ir_k[0, 1]
    )

    return [camera0, camera1, camera2]


def main2():
    engine = sapien.Engine()  # Create a physical simulation engine
    render_config = sapien.KuafuConfig()
    render_config.use_viewer = False
    render_config.use_denoiser = True
    render_config.spp = 128
    render_config.max_bounces = 8

    renderer = sapien.KuafuRenderer(render_config)
    engine.set_renderer(renderer)

    scene = engine.create_scene()  # Create an instance of simulation world (aka scene)
    scene.set_timestep(1 / 100.0)  # Set the simulation frequency

    # NOTE: How to build actors (rigid bodies) is elaborated in create_actors.py
    ground_material = renderer.create_material()
    ground_color = np.array([119, 146, 198, 256]) / 256
    ground_material.set_base_color(ground_color)
    ground_material.set_specular(0.5)
    scene.add_ground(altitude=0, render_material=ground_material)  # Add a ground

    # chair
    chair_loader = scene.create_urdf_loader()
    chair_loader.fix_root_link = True
    chair_loader.scale = 0.7

    urdf_path = "/home/rayu/Projects/active_zero2/scripts/large_scene/partnet-mobility-dataset/37825/mobility.urdf"
    chair_loader.load_multiple_collisions_from_file = True
    chair = chair_loader.load(str(urdf_path))
    chair.set_name("chair")
    chair.set_pose(Pose([-0.9, -0.8, 0.50], [0.65998315, 0., 0., -0.75128041]))

    # table
    table_loader = scene.create_urdf_loader()
    table_loader.fix_root_link = True
    table_loader.scale = 0.96

    urdf_path = "/home/rayu/Projects/active_zero2/scripts/large_scene/partnet-mobility-dataset/30869/mobility.urdf"
    table_loader.load_multiple_collisions_from_file = True
    table = table_loader.load(str(urdf_path))
    table.set_name("table")
    table.set_pose(Pose([0, 0, 0.61]))

    # cabinet
    cabinet_loader = scene.create_urdf_loader()
    cabinet_loader.fix_root_link = True

    urdf_path = "/home/rayu/Projects/active_zero2/scripts/large_scene/2/mobility_textures.urdf"
    cabinet_loader.load_multiple_collisions_from_file = True
    cabinet = cabinet_loader.load(str(urdf_path))
    cabinet.set_name("cabinet")
    cabinet.set_pose(Pose([-0.06, -0.58, 0.75], t3d.quaternions.mat2quat(t3d.euler.euler2mat(0, 0, 1.2))))
    cabinet.set_qpos([0.0, 1.8])

    # actors
    load_objects(scene, renderer, vulkan=False)
    # Add some lights so that you can observe the scene
    scene.set_ambient_light([0.5, 0.5, 0.5])
    plight1 = scene.add_point_light([-0.3, -0.3, 2.5], [30, 30, 30])
    plight2 = scene.add_point_light([2, -2, 2.5], [10, 10, 10])
    plight3 = scene.add_point_light([-2, 2, 2.5], [10, 10, 10])
    plight4 = scene.add_point_light([2, 2, 2.5], [10, 10, 10])
    plight5 = scene.add_point_light([-2, -2, 2.5], [10, 10, 10])
    alight = scene.add_active_light(
        pose=Pose([0.4, 0, 0.8]),
        # pose=Pose(cam_mount.get_pose().p, apos),
        color=[0, 0, 0],
        fov=1.6,
        tex_path="/home/rayu/Projects/active_zero2/data_rendering/materials/d415-pattern-sq.png",
    )

    # change light
    def lights_on():
        scene.set_ambient_light([0.5, 0.5, 0.5])
        plight1.set_color([30, 30, 30])
        plight2.set_color([10, 10, 10])
        plight3.set_color([10, 10, 10])
        plight4.set_color([10, 10, 10])
        plight5.set_color([10, 10, 10])
        alight.set_color([0.0, 0.0, 0.0])

    def lights_off():
        p_scale = 4.0
        scene.set_ambient_light([0.03, 0.03, 0.03])
        plight1.set_color([0.3 * p_scale, 0.1 * p_scale, 0.1 * p_scale])
        plight2.set_color([0.1 * p_scale, 0.03 * p_scale, 0.03 * p_scale])
        plight3.set_color([0.1 * p_scale, 0.03 * p_scale, 0.03 * p_scale])
        plight4.set_color([0.1 * p_scale, 0.03 * p_scale, 0.03 * p_scale])
        plight5.set_color([0.1 * p_scale, 0.03 * p_scale, 0.03 * p_scale])
        alight.set_color([20.0, 10.0, 10.0])

    near, far = 0.1, 100
    width, height = 1280, 720
    camera_mount_actor = scene.create_actor_builder().build_kinematic()
    rgb_intrinsic = np.array([[893.708, 0, 632.56], [0, 893.708, 369.403], [0, 0, 1]])
    ir_intrinsic = rgb_intrinsic

    cam_rgb, cam_irl, cam_irr = create_realsense_d415("realsense", camera_mount_actor, scene,
                                                      rgb_intrinsic, ir_intrinsic)


    # Compute the camera pose by specifying forward(x), left(y) and up(z)
    cam_pos = np.array([-1.1, 0.6, 1.5])
    mat44 = np.eye(4)
    mat44[:3, :3] = t3d.euler.euler2mat(0, np.pi / 6.2, -np.pi / 2.8)
    mat44[:3, 3] = cam_pos
    camera_mount_actor.set_pose(sapien.Pose.from_transformation_matrix(mat44))
    mount_T = t3d.quaternions.quat2mat((-0.5, 0.5, 0.5, -0.5))
    apos = camera_mount_actor.get_pose().to_transformation_matrix()[:3, :3] @ mount_T
    apos = t3d.quaternions.mat2quat(apos)
    alight.set_pose(Pose(camera_mount_actor.get_pose().p, apos))

    print('Intrinsic matrix\n', cam_rgb.get_intrinsic_matrix())

    lights_on()
    scene.update_render()
    cam_rgb.take_picture()
    rgba = cam_rgb.get_float_texture('Color')  # [H, W, 4]
    rgba = rgba[..., :3] * 255
    cv2.imwrite('color_rt.png', (rgba[..., ::-1].astype(np.uint8)))

    real_image = cv2.imread('/home/rayu/Projects/active_zero2/scripts/large_scene/real/10092_rgb_Color.png')
    mix_image = np.clip((real_image.astype(float) * 0.5 + rgba * 0.5), 0, 255).astype(np.uint8)
    cv2.imwrite('mixed_rt.png', mix_image[..., ::-1])

    lights_off()
    scene.update_render()
    cam_irl.take_picture()
    cam_irr.take_picture()

    irl = cam_irl.get_float_texture('Color')
    irr = cam_irr.get_float_texture('Color')
    irl = np.clip((irl[..., :3] * 255), 0, 255).astype(np.uint8)
    irl = cv2.cvtColor(irl, cv2.COLOR_RGB2GRAY)
    irr = np.clip((irr[..., :3] * 255), 0, 255).astype(np.uint8)
    irr = cv2.cvtColor(irr, cv2.COLOR_RGB2GRAY)
    default_speckle_shape = 400
    default_gaussian_sigma = 0.83

    noise_scale = 0.1
    irl = sim_ir_noise(irl.copy(), speckle_shape=default_speckle_shape / noise_scale,
                           speckle_scale=noise_scale / default_speckle_shape,
                           gaussian_sigma=default_gaussian_sigma * noise_scale)
    irr = sim_ir_noise(irr.copy(), speckle_shape=default_speckle_shape / noise_scale,
                           speckle_scale=noise_scale / default_speckle_shape,
                           gaussian_sigma=default_gaussian_sigma * noise_scale)

    cv2.imwrite('irl.png', irl)
    cv2.imwrite('irr.png', irr)
    LR_SIZE = RGB_SIZE = (1280, 720)
    l2r = np.array([
        [1., 0, 0, -0.0545],
        [0, 1., 0, 0],
        [0, 0, 1., 0],
        [0, 0, 0, 1.]
    ])
    l2rgb = np.array([
        [1., 0, 0, 0.0175],
        [0, 1., 0, 0],
        [0, 0, 1., 0],
        [0, 0, 0, 1.]
    ])
    depthSensor = DepthSensor(LR_SIZE, rgb_intrinsic, rgb_intrinsic, l2r,
                              RGB_SIZE, rgb_intrinsic, l2rgb,
                              min_depth=0.0, max_depth=6.0,
                              census_width=7, census_height=7, block_width=7, block_height=7,
                              uniqueness_ratio=15, depth_dilation=True)
    depth = depthSensor.compute(irl, irr)
    depth = (depth*1000.0).astype(np.uint16)
    vis_depth = visualize_depth(depth)
    cv2.imwrite('depth_sim_colored.png', vis_depth)

    depth_real = np.fromfile("/home/rayu/Projects/active_zero2/scripts/large_scene/real/10092_depth_Depth.raw", dtype=np.uint16)
    depth_real = depth_real.reshape((360, 640))
    depth_real = cv2.resize(depth_real, (1280, 720), interpolation=cv2.INTER_NEAREST)
    depth_real = depth_real.astype(np.uint16)
    vis_depth = visualize_depth(depth_real)
    cv2.imwrite("depth_real_colored.png", vis_depth)



if __name__ == '__main__':
    main2()
