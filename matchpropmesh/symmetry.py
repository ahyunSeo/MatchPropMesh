"""Reflective symmetry-plane detection in the camera frame."""
import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def candidate_planes(center, azimuth, polar, eps=1e-12):
    """Planes [a,b,c,d] through `center` with normals given in y-up spherical angles (radians)."""
    azimuth = azimuth.to(center.device, center.dtype)
    polar = polar.to(center.device, center.dtype)
    sinp = torch.sin(polar)
    n = torch.stack((sinp * torch.cos(azimuth), torch.cos(polar), sinp * torch.sin(azimuth)), dim=-1)
    n = n / (n.norm(dim=-1, keepdim=True) + eps)
    d = -(n * center.unsqueeze(0).expand_as(n)).sum(dim=-1)
    return torch.cat((n, d.unsqueeze(-1)), dim=-1)


@torch.no_grad()
def plane_coordinates(points, planes):
    """Signed distances (K,N) of points (N,3) to planes (K,4) and 2D coordinates (K,N,2)
    of their projections in a per-plane tangent frame."""
    x = points.float()
    n = planes[..., :3].float()
    n_norm = n.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    n_hat = n / n_norm
    d_hat = planes[..., 3].float() / n_norm.squeeze(-1)

    sd = torch.einsum('nd,...d->...n', x, n_hat) + d_hat[..., None]
    proj = x.unsqueeze(0).expand(n_hat.shape[:-1] + x.shape) - sd[..., :, None] * n_hat[..., None, :]

    # Tangent basis from the coordinate axis least aligned with the normal.
    axes = torch.eye(3, device=n_hat.device, dtype=n_hat.dtype)
    ref = axes[n_hat.abs().argmin(dim=-1)]
    u = torch.linalg.cross(n_hat, ref, dim=-1)
    u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    v = torch.linalg.cross(n_hat, u, dim=-1)
    v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    rel = proj - (-d_hat[..., None] * n_hat)[..., None, :]
    uv = torch.stack([(rel * u[..., None, :]).sum(dim=-1), (rel * v[..., None, :]).sum(dim=-1)], dim=-1)
    return sd, uv


@torch.no_grad()
def rasterize_signed_distances(uv, sd, size=512, eps=1e-12):
    """Per-plane mean signed distance maps (K,2,size,size): channel 0 averages sd > 0,
    channel 1 averages sd < 0. All planes share one bounding box in plane coordinates."""
    K, N = sd.shape
    u, v = uv[..., 0], uv[..., 1]
    umin, umax = u.amin(dim=1).min().reshape(1), u.amax(dim=1).max().reshape(1)
    vmin, vmax = v.amin(dim=1).min().reshape(1), v.amax(dim=1).max().reshape(1)

    span_u = (umax - umin).clamp_min(eps)
    span_v = (vmax - vmin).clamp_min(eps)
    scale = (size - 1) / torch.maximum(span_u, span_v)
    pad_u = (size - span_u * scale) * 0.5
    pad_v = (size - span_v * scale) * 0.5
    x = torch.round((u - umin[:, None]) * scale[:, None] + pad_u[:, None]).long().clamp_(0, size - 1)
    y = torch.round((v - vmin[:, None]) * scale[:, None] + pad_v[:, None]).long().clamp_(0, size - 1)
    lin = y * size + x

    HW = size * size
    sd_sum = torch.zeros(K * 2 * HW, device=sd.device, dtype=sd.dtype)
    count = torch.zeros(K * 2 * HW, device=sd.device, dtype=sd.dtype)
    k_idx = torch.arange(K, device=sd.device).view(K, 1).expand(K, N)
    for mask, offset in ((sd > 0, 0), (sd < 0, HW)):
        if mask.any():
            idx = k_idx[mask].long() * (2 * HW) + offset + lin[mask].long()
            sd_sum.scatter_add_(0, idx, sd[mask])
            count.scatter_add_(0, idx, torch.ones_like(sd[mask]))
    return (sd_sum.view(K, 2, size, size) / (count.view(K, 2, size, size) + eps))


@torch.no_grad()
def box_filter(x, k=5):
    C = x.shape[1]
    x = F.pad(x, (k // 2,) * 4, mode='reflect')
    w = torch.ones(C, 1, k, k, device=x.device, dtype=x.dtype) / (k * k)
    return F.conv2d(x, w, groups=C)


@torch.no_grad()
def detect_symmetry_plane(pts_camera, n_azimuth=180, polar_deg=(90, 95, 100, 105, 110, 115, 120)):
    """Scores candidate planes through the centroid by how well the smoothed signed-distance
    maps of the two sides cancel, and returns (uv, sd, plane) of the best plane."""
    az = torch.from_numpy(np.linspace(0, 180, n_azimuth, endpoint=False)).float() * (torch.pi / 180.0)
    pol = torch.tensor(polar_deg).float() * (torch.pi / 180.0)
    POL, AZ = torch.meshgrid(pol, az, indexing='ij')
    planes = candidate_planes(pts_camera.mean(dim=0), AZ.reshape(-1), POL.reshape(-1))

    sd, uv = plane_coordinates(pts_camera, planes)
    maps = box_filter(rasterize_signed_distances(uv, sd))
    # Channel 1 stores negative values, so the sum is the per-pixel magnitude mismatch.
    score = torch.abs(maps.sum(dim=1)).flatten(1, 2).sum(dim=1)
    k = torch.argmin(score)
    return uv[k], sd[k], planes[k]


def reflect_points(points, plane):
    """Reflects points (N,3) across the plane [a,b,c,d]."""
    n = plane[:3].float()
    n_norm = n.norm().clamp_min(1e-12)
    n_hat, d_hat = n / n_norm, plane[3].float() / n_norm
    sd = points.float() @ n_hat + d_hat
    return points.float() - 2.0 * sd.unsqueeze(-1) * n_hat
