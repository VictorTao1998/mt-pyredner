import numpy as np
import scipy
import time
import torch
import torch.nn as nn


def GetInitialization(valid_disp):
    initialization = valid_disp.copy()

    w, h = valid_disp.shape
    last_known = -1
    first_known = -1
    for col in range(0, w):
        for row in range(0, h):
            if valid_disp[col, row] > 0:
                last_known = valid_disp[col, row]
            elif initialization[col, row] > 0:
                last_known = initialization[col, row]
            if first_known < 0:
                first_known = last_known
            initialization[col, row] = last_known
    initialization[initialization < 0] = first_known

    return initialization


def DensifyFrame(
    valid_disp: np.ndarray, hard_edges: np.ndarray, lambda_d: float, lambda_s: float, num_solver_iterations: int
) -> np.ndarray:
    w, h = valid_disp.shape
    num_pixels = w * h
    A = scipy.sparse.dok_matrix((num_pixels * 3, num_pixels), dtype=np.float32)
    A[A > 0] = 0
    A[A < 0] = 0
    b = np.zeros(num_pixels * 3, dtype=np.float32)
    x0 = np.zeros(num_pixels, dtype=np.float32)
    num_entries = 0

    smoothness_x = np.zeros((w, h), dtype=np.float32)
    smoothness_y = np.zeros((w, h), dtype=np.float32)
    tic = time.time()
    initialization = GetInitialization(valid_disp)
    print("init time: ", time.time() - tic)

    tic = time.time()
    for row in range(1, h - 1):
        for col in range(1, w - 1):
            x0[col + row * w] = initialization[col, row]
            # Add the data constraints
            if valid_disp[col, row] > 0.00:
                A[num_entries, col + row * w] = lambda_d
                b[num_entries] = valid_disp[col, row] * lambda_d
                num_entries += 1

            # Add the smoothness constraints
            if hard_edges[col, row] == hard_edges[col - 1, row]:
                smoothness_x[col, row] = lambda_s
                A[num_entries, (col - 1) + row * w] = lambda_s
                A[num_entries, col + row * w] = -lambda_s
                b[num_entries] = 0
                num_entries += 1

            if hard_edges[col, row] == hard_edges[col, row - 1]:
                smoothness_y[col, row] = lambda_s
                A[num_entries, col + (row - 1) * w] = lambda_s
                A[num_entries, col + row * w] = -lambda_s
                b[num_entries] = 0
                num_entries += 1

    print("assignment: ", time.time() - tic)
    # Solve the system

    tic = time.time()
    [x, info] = scipy.sparse.linalg.cg(A.transpose() * A, A.transpose() * b, x0, 1e-05, num_solver_iterations)
    print("solve: ", time.time() - tic)
    if info < 0:
        print("====> Error! Illegal input!")
    elif info > 0:
        print("====> Ran " + str(info) + " solver iterations.")
    else:
        print("====> Solver converged!")

    tic = time.time()
    disp = np.zeros(valid_disp.shape, dtype=np.float32)

    # Copy back the pixels
    for row in range(0, h):
        for col in range(0, w):
            if x[col + row * w] > 0:
                disp[col, row] = x[col + row * w]

    print("final: ", time.time() - tic)
    return disp


def compute_disparity_plane_normal(disp: np.ndarray, plane_size: int) -> np.ndarray:
    assert plane_size % 2 == 1
    height, width = disp.shape[:2]
    x_linspace = np.linspace(0, width - 1, width)
    y_linspace = np.linspace(0, height - 1, height)
    x_coordinates, y_coordinates = np.meshgrid(x_linspace, y_linspace)

    x = np.stack([x_coordinates, y_coordinates, np.ones_like(disp), disp])
    x = torch.from_numpy(x).unsqueeze(0)
    unfold = nn.Unfold(plane_size, padding=(plane_size - 1) // 2)
    x = unfold(x).view(4, plane_size * plane_size, height * width)
    plane_conf = np.sum(np.abs(x[3].numpy()) > 1e-5, axis=0) == (plane_size * plane_size)
    plane_conf = plane_conf.reshape(height, width)
    x = x.permute(2, 1, 0)

    A = x[..., :3]
    b = x[..., 3:]

    p, res, _, _ = torch.linalg.lstsq(A, b)
    p = p.view(height, width, 3).numpy()
    p[..., 2] = -1
    p = p / np.linalg.norm(p, axis=2, keepdims=True)
    p[np.logical_not(plane_conf)] = 0

    return p


def compute_disparity_plane(disp: np.ndarray, plane_size: int) -> np.ndarray:
    assert plane_size % 2 == 1
    height, width = disp.shape[:2]
    x_linspace = np.linspace(0, width - 1, width)
    y_linspace = np.linspace(0, height - 1, height)
    x_coordinates, y_coordinates = np.meshgrid(x_linspace, y_linspace)

    x = np.stack([x_coordinates, y_coordinates, np.ones_like(disp), disp])
    x = torch.from_numpy(x).unsqueeze(0)
    unfold = nn.Unfold(plane_size, padding=(plane_size - 1) // 2)
    x2 = unfold(x).view(4, plane_size * plane_size, height * width)
    plane_conf = np.sum(np.abs(x2[3].numpy()) > 1e-5, axis=0) == (plane_size * plane_size)
    plane_conf = plane_conf.reshape(height, width)
    x2 = x2.permute(2, 1, 0)

    A2 = x2[..., :3]
    b2 = x2[..., 3:]

    x = x.view(1, 4, -1).permute(2, 0, 1)
    A = x[..., :3]
    b = x[..., 3:]
    p, _, _, _ = torch.linalg.lstsq(A2, b2)
    res = torch.linalg.norm(torch.bmm(A, p) - b, dim=1).view(height, width).numpy()
    p = p.view(height, width, 3).numpy()

    p[..., :2] = np.clip(p[..., :2], -0.5, 0.5)
    p = ((p + 1) * 30000.0).astype(np.uint16)
    p[..., 2] = 1
    p *= plane_conf.astype(np.uint16)[..., None]
    p *= (res < 1.0).astype(np.uint16)[..., None]

    return p, res


if __name__ == "__main__":
    disp = np.random.rand(12, 16)
    p, res = compute_disparity_plane(disp, 5)
    print(p.shape, res.shape)
