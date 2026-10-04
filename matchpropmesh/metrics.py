"""Chamfer distance and F-score between meshes."""
import torch
import trimesh

from third_party.chamfer3D.dist_chamfer_3D import chamfer_3DDist

F_THRESHOLDS = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2)

_CHAMFER = None


def chamfer_distance(X1, X2):
    """Nearest-neighbor distances (not squared) between point clouds (B,N,3) and (B,M,3)."""
    global _CHAMFER
    if _CHAMFER is None:
        _CHAMFER = chamfer_3DDist().to(X1.device)
    d1, d2, i1, i2 = _CHAMFER(X1, X2)
    return d1.sqrt(), d2.sqrt(), i1, i2


@torch.no_grad()
def fscore(dist1, dist2, thresholds=F_THRESHOLDS):
    thr = torch.tensor(thresholds, device=dist1.device).view(1, 1, -1)
    prec = (dist1.unsqueeze(-1) < thr).float().mean(dim=1)
    rec = (dist2.unsqueeze(-1) < thr).float().mean(dim=1)
    f = 2 * prec * rec / (prec + rec).clamp_min(1e-12)
    return f.nan_to_num_(0.0)


@torch.no_grad()
def sample_points(mesh: trimesh.Trimesh, n_points, seed=42):
    pts, _ = trimesh.sample.sample_surface(mesh, n_points, seed=seed)
    return torch.tensor(pts).float()


@torch.no_grad()
def mesh_metrics(pred_mesh, gt_mesh, n_points=16000, device='cuda'):
    """CD (mean of accuracy and completeness) and F-scores from 16k surface samples."""
    pc_pred = sample_points(pred_mesh, n_points).to(device).unsqueeze(0).contiguous()
    pc_gt = sample_points(gt_mesh, n_points).to(device).unsqueeze(0).contiguous()
    acc, comp, _, _ = chamfer_distance(pc_pred, pc_gt)
    f = fscore(acc, comp)[0]
    acc, comp = acc.mean(dim=1)[0].item(), comp.mean(dim=1)[0].item()
    return {'cd': 0.5 * (acc + comp), 'acc': acc, 'comp': comp,
            'fscore': {str(t): v for t, v in zip(F_THRESHOLDS, f.tolist())}}
