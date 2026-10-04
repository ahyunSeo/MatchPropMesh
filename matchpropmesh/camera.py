"""Camera conventions, mesh <-> camera transforms, back-projection, and rasterization.

Meshes are stored in the glTF world frame (Y-up). Cameras follow Blender/OpenGL:
+X right, +Y up, -Z forward, so visible points have z < 0. Poses are camera-to-world
matrices in the Blender world frame (Z-up), as exported by BlenderProc.
"""
import json

import numpy as np
import torch
import trimesh
import nvdiffrast.torch as dr

# glTF (Y-up) -> Blender (Z-up): +90 degrees about X.
_RX = [[1, 0, 0, 0],
       [0, 0, -1, 0],
       [0, 1, 0, 0],
       [0, 0, 0, 1]]


def normalize_like_blender(mesh: trimesh.Trimesh, scale=0.8, eps=1e-12):
    """Scale the longest bbox edge to `scale` and center the bbox at the origin,
    matching the normalization applied before rendering."""
    bbox_min, bbox_max = mesh.bounds
    longest = float((bbox_max - bbox_min).max())
    if longest < eps:
        raise ValueError("Degenerate mesh bbox (max extent ~ 0).")
    s = scale / longest

    S = np.eye(4, dtype=np.float64)
    S[0, 0] = S[1, 1] = S[2, 2] = s
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = -(s * (0.5 * (bbox_min + bbox_max)))
    M = T @ S

    out = mesh.copy()
    out.apply_transform(M)
    out.remove_unreferenced_vertices()
    return out, M


def load_mesh(path, normalize=True):
    mesh = trimesh.load(path, force='mesh')
    if normalize:
        mesh, _ = normalize_like_blender(mesh)
    return mesh


def load_camera(view, camera_path):
    """Returns the camera-to-world pose (1,4,4) and NDC intrinsics [fx, fy, cx, cy]."""
    cams = json.load(open(camera_path))
    c2w = torch.tensor(cams['extrinsics'][view]['c2w_blender']).unsqueeze(0)
    K, H, W = cams['K'], cams['h'], cams['w']
    intrinsics = [2.0 * K[0][0] / W, 2.0 * K[1][1] / H,
                  2.0 * K[0][2] / W - 1.0, 2.0 * K[1][2] / H - 1.0]
    return c2w, intrinsics


def mesh_to_camera(mesh, c2w):
    """Mesh vertices (glTF world) -> camera frame. Returns (B,P,3)."""
    pts = torch.from_numpy(mesh.vertices).float().unsqueeze(0).to(c2w.device)
    P = pts.shape[1]
    Rx = torch.tensor(_RX, dtype=pts.dtype, device=pts.device).unsqueeze(0)
    ones = torch.ones(1, P, 1, dtype=pts.dtype, device=pts.device)
    pts_world = (torch.cat([pts, ones], dim=-1) @ Rx.transpose(1, 2))[..., :3]

    if c2w.dim() == 2:
        c2w = c2w.unsqueeze(0)
    w2c = torch.inverse(c2w).to(pts.dtype).to(pts.device)
    B = w2c.shape[0]
    ones_b = torch.ones(B, P, 1, dtype=pts.dtype, device=pts.device)
    pts_world_h = torch.cat([pts_world.expand(B, P, 3), ones_b], dim=-1)
    return (pts_world_h @ w2c.transpose(1, 2))[..., :3]


def camera_to_mesh(pts_camera, c2w):
    """Camera frame -> mesh vertices (glTF world). Accepts (P,3) or (B,P,3)."""
    batched = pts_camera.ndim == 3
    if not batched:
        pts_camera = pts_camera.unsqueeze(0)
    if c2w.ndim == 2:
        c2w = c2w.unsqueeze(0)
    B, P, _ = pts_camera.shape
    if c2w.shape[0] != B:
        c2w = c2w.expand(B, 4, 4)
    dtype, device = pts_camera.dtype, pts_camera.device

    ones = torch.ones(B, P, 1, dtype=dtype, device=device)
    pts_world = (torch.cat([pts_camera.to(dtype), ones], dim=-1) @ c2w.to(dtype).transpose(1, 2))[..., :3]
    Rx = torch.tensor(_RX, dtype=dtype, device=device).unsqueeze(0)
    pts = (torch.cat([pts_world, ones], dim=-1) @ Rx)[..., :3]
    return pts if batched else pts[0]


@torch.no_grad()
def backproject_depth(depth, intrinsics, eps=1e-8):
    """Depth map (H,W) -> camera-frame points (N,3) of the foreground pixels."""
    H, W = depth.shape
    fx, fy, cx, cy = [float(v) for v in intrinsics]
    d = depth.clamp_min(0)
    fg = d > eps

    yy, xx = torch.meshgrid(torch.arange(H, device=depth.device, dtype=torch.float32),
                            torch.arange(W, device=depth.device, dtype=torch.float32),
                            indexing='ij')
    x_ndc = (2.0 * xx / max(W - 1.0, 1.0) - 1.0)[fg]
    y_ndc = (1.0 - 2.0 * yy / max(H - 1.0, 1.0))[fg]
    d = d[fg]
    return torch.stack([d * (x_ndc - cx) / fx, d * (y_ndc - cy) / fy, -d], dim=-1)


def project(pts_camera, intrinsics, near=0.01, far=1000.0):
    """Camera-frame points (B,N,3) -> clip coordinates (B,N,4) for nvdiffrast."""
    fx, fy, cx, cy = [torch.tensor(v, device=pts_camera.device, dtype=torch.float32) for v in intrinsics]
    B, N, _ = pts_camera.shape
    P = torch.zeros((B, 4, 4), device=pts_camera.device, dtype=torch.float32)
    P[:, 0, 0] = fx
    P[:, 0, 2] = -cx
    P[:, 1, 1] = fy
    P[:, 1, 2] = -cy
    P[:, 2, 2] = (far + near) / (near - far)
    P[:, 2, 3] = (2.0 * far * near) / (near - far)
    P[:, 3, 2] = -1.0
    ones = torch.ones((B, N, 1), device=pts_camera.device, dtype=pts_camera.dtype)
    return torch.matmul(torch.cat([pts_camera, ones], dim=-1), P.transpose(1, 2))


@torch.no_grad()
def rasterize_depth(pos_clip, faces, attr, H, W):
    """Rasterizes one mesh. Returns the depth map (H,W) and per-pixel face ids (-1 on background)."""
    ranges = torch.tensor([[0, faces.shape[0]]], dtype=torch.int32)
    ctx = dr.RasterizeCudaContext(device=torch.device('cuda'))
    rast, _ = dr.rasterize(ctx, pos_clip.unsqueeze(0), faces.int(), (H, W), ranges=ranges)
    attr_px, _ = dr.interpolate(attr.unsqueeze(0), rast, faces)
    mask = rast[0, ..., 3] > 0
    depth = torch.where(mask, attr_px[0, ..., 0], torch.zeros_like(attr_px[0, ..., 0]))
    face_id = rast[0, ..., 3].to(torch.int32) - 1
    return depth.flip(0), face_id.flip(0)
