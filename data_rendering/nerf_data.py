import os
import shutil
from shutil import copy2


src = "/media/jianyu/dataset/messy-table-dataset/sim-patcam-align/data/"
dst = "/media/jianyu/dataset/messy-table-dataset/nerf-sim-patcam-align"

num_scene = 3
os.makedirs(dst, exist_ok=True)
for sc in range(num_scene):
    os.makedirs(os.path.join(dst, str(sc)),exist_ok=True)
    os.makedirs(os.path.join(dst, str(sc), 'train'),exist_ok=True)
    os.makedirs(os.path.join(dst, str(sc), 'val'),exist_ok=True)
    os.makedirs(os.path.join(dst, str(sc), 'test'),exist_ok=True)
    for sc_id in range(19):
        prefix = '0-'+ str(sc) + '-' + str(sc_id)
        src_path = os.path.join(src, prefix)
        dst_path = os.path.join(dst, str(sc), 'train', prefix)
        isdir = os.path.isdir(src_path)
        #print(isdir, dst_path)
        if isdir:
            shutil.copytree(src_path, dst_path, symlinks=False, ignore=None, copy_function=copy2, ignore_dangling_symlinks=False)

    prefix = '0-'+ str(sc) + '-' + '19'
    src_path = os.path.join(src, prefix)
    dst_path = os.path.join(dst, str(sc), 'test', prefix)
    isdir = os.path.isdir(src_path)
    #print(isdir, dst_path)
    if isdir:
        shutil.copytree(src_path, dst_path, symlinks=False, ignore=None, copy_function=copy2, ignore_dangling_symlinks=False)

    prefix = '0-'+ str(sc) + '-' + '20'
    src_path = os.path.join(src, prefix)
    dst_path = os.path.join(dst, str(sc), 'val', prefix)
    isdir = os.path.isdir(src_path)
    #print(isdir, dst_path)
    if isdir:
        shutil.copytree(src_path, dst_path, symlinks=False, ignore=None, copy_function=copy2, ignore_dangling_symlinks=False)
