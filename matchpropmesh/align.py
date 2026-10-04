"""Global rotation alignment of a reconstructed mesh to the GT mesh for evaluation.

Both meshes are normalized to the longest bbox edge 0.8 and centered. A rotation search
over SO(3) scores Chamfer distance on surface samples: a coarse 24^3 grid of
(azimuth, elevation, roll), then a 2x denser grid inside the 64 best coarse bins.
"""
import torch
import trimesh

from .camera import normalize_like_blender
from .metrics import chamfer_distance, sample_points

# Fixed axis permutation between the backbone and GT conventions.
_R_PERM = [[-1, 0, 0],
           [0, 0, -1],
           [0, -1, 0]]


def rotations_from_angles(azim, elev, roll, device='cuda', dtype=torch.float32):
    """R = Rz(roll) Rx(elev) Ry(azim) R_perm for broadcastable angle tensors in degrees."""
    a, e, r = [torch.deg2rad(t.to(device=device, dtype=dtype)) for t in (azim, elev, roll)]
    ca, sa = torch.cos(a), torch.sin(a)
    ce, se = torch.cos(e), torch.sin(e)
    cr, sr = torch.cos(r), torch.sin(r)
    zero, one = torch.zeros_like(ca), torch.ones_like(ca)

    Ry = torch.stack((torch.stack((ca, zero, sa), -1),
                      torch.stack((zero, one, zero), -1),
                      torch.stack((-sa, zero, ca), -1)), -2)
    Rx = torch.stack((torch.stack((one, zero, zero), -1),
                      torch.stack((zero, ce, -se), -1),
                      torch.stack((zero, se, ce), -1)), -2)
    Rz = torch.stack((torch.stack((cr, sr, zero), -1),
                      torch.stack((-sr, cr, zero), -1),
                      torch.stack((zero, zero, one), -1)), -2)
    R_perm = torch.tensor(_R_PERM, device=device, dtype=dtype).view(*([1] * (Ry.ndim - 2)), 3, 3)
    return (Rz @ Rx @ Ry @ R_perm).reshape(-1, 3, 3).contiguous()


def rotation_grid(n, device='cuda', dtype=torch.float32):
    """n^3 rotations on a regular angle grid; roll varies fastest, then elevation, then azimuth."""
    angles = torch.arange(0, 360, 360 / n, device=device, dtype=dtype)
    A, E, R = torch.meshgrid(angles, angles, angles, indexing='ij')
    return rotations_from_angles(A, E, R, device, dtype)


def refine_bins(idx, n, factor=2, device='cuda', dtype=torch.float32):
    """Rotations of the factor-times denser grid that fall inside the given coarse bins."""
    ir = idx % n
    ie = (idx // n) % n
    ia = (idx // n) // n
    step = 360.0 / n
    offs = torch.arange(0, factor, device=device, dtype=dtype) * (step / factor)
    A = (ia.to(dtype) * step)[:, None, None, None] + offs[None, :, None, None]
    E = (ie.to(dtype) * step)[:, None, None, None] + offs[None, None, :, None]
    R = (ir.to(dtype) * step)[:, None, None, None] + offs[None, None, None, :]
    A, E, R = torch.broadcast_tensors(A % 360.0, E % 360.0, R % 360.0)
    return rotations_from_angles(A, E, R, device, dtype)


def _chamfer(X, Y):
    acc, comp, _, _ = chamfer_distance(X, Y.expand(X.shape[0], -1, -1).contiguous())
    cd = 0.5 * (acc.mean(dim=1) + comp.mean(dim=1))
    return torch.where(torch.isfinite(cd), cd, torch.full_like(cd, float('inf')))


@torch.no_grad()
def align_mesh(pred_mesh, gt_mesh, n_points=16000, grid=24, topk=64, coarse_stride=4,
               batch_size=24 * 24 * 48, device='cuda'):
    """Returns the normalized, rotated prediction as a vertex-colored mesh."""
    pred_mesh, _ = normalize_like_blender(pred_mesh)
    gt_mesh, _ = normalize_like_blender(gt_mesh)
    pc_pred = sample_points(pred_mesh, n_points).to(device).unsqueeze(0).contiguous()
    pc_gt = sample_points(gt_mesh, n_points).to(device).unsqueeze(0).contiguous()

    # Coarse search on a point subset; keep the top-k bins and the best rotation.
    rotations = rotation_grid(grid, device)
    pred_c = pc_pred[:, ::coarse_stride].contiguous()
    gt_c = pc_gt[:, ::coarse_stride].contiguous()
    topk_cd = torch.full((topk,), float('inf'), device=device)
    topk_idx = torch.full((topk,), -1, device=device, dtype=torch.long)
    best_cd = torch.tensor(float('inf'), device=device)
    best_R = None
    for i in range(0, rotations.shape[0], batch_size):
        Rb = rotations[i:i + batch_size]
        cd = _chamfer(torch.bmm(pred_c.expand(Rb.shape[0], -1, -1), Rb.transpose(1, 2)), gt_c)
        vals, sel = torch.topk(torch.cat([topk_cd, cd]), k=topk, largest=False)
        topk_cd = vals
        topk_idx = torch.cat([topk_idx, torch.arange(i, i + Rb.shape[0], device=device)])[sel]
        j = torch.argmin(cd)
        if cd[j] < best_cd:
            best_cd, best_R = cd[j], Rb[j].clone()

    # Refine inside the top-k bins on the denser grid with all points.
    topk_idx = topk_idx[topk_idx >= 0]
    if topk_idx.numel() > 0:
        rotations = refine_bins(topk_idx, grid, device=device)
        for i in range(0, rotations.shape[0], batch_size):
            Rb = rotations[i:i + batch_size]
            cd = _chamfer(torch.bmm(pc_pred.expand(Rb.shape[0], -1, -1), Rb.transpose(1, 2)), pc_gt)
            j = torch.argmin(cd)
            if cd[j] < best_cd:
                best_cd, best_R = cd[j], Rb[j].clone()

    V = torch.tensor(pred_mesh.vertices, device=device, dtype=torch.float32) @ best_R.transpose(0, 1)
    if hasattr(pred_mesh.visual, 'vertex_colors'):
        colors = pred_mesh.visual.vertex_colors
    else:
        colors = pred_mesh.visual.material.to_color(pred_mesh.visual.uv)
    return trimesh.Trimesh(V.cpu().numpy(), pred_mesh.faces, vertex_colors=colors)
