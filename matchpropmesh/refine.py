"""RGBD-to-3D mesh refinement: depth matching, symmetry propagation, smoothness solve."""
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from scipy.spatial import cKDTree
from trimesh.smoothing import filter_taubin

from .camera import backproject_depth, camera_to_mesh, mesh_to_camera, project, rasterize_depth
from .matching import match_points
from .propagation import propagate_displacements
from .symmetry import detect_symmetry_plane, reflect_points

DEPTH_RES = 128          # back-projection resolution of the input depth
RENDER_RES = 512         # rasterization resolution for visible vertices
DEPTH_COST_THRES = 0.1   # max distance of a depth match
SYM_COST_THRES = 0.2     # max distance of a symmetric match in (u, v, |s|)
MAX_VERTICES = 30000


def decimate_mesh(mesh, ratio):
    """Quadric decimation that transfers colors by nearest original vertex."""
    if hasattr(mesh.visual, 'uv'):
        colors = mesh.visual.material.to_color(mesh.visual.uv)
        mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=colors)
    orig = mesh.copy()
    orig.remove_unreferenced_vertices()
    out = mesh.copy().simplify_quadric_decimation(max(4, int(mesh.faces.shape[0] * ratio)))
    out.remove_unreferenced_vertices()
    _, idx = cKDTree(orig.vertices.view(np.ndarray)).query(out.vertices.view(np.ndarray), k=1)
    out.visual = trimesh.visual.ColorVisuals(mesh=out, vertex_colors=orig.visual.vertex_colors[idx])
    return out


def _propagate_and_smooth(mesh_init, pts_init, handle_idx, handle_disp, c2w):
    V = propagate_displacements(pts_init.cpu(), mesh_init.faces, handle_idx.cpu(), handle_disp.cpu())
    out = mesh_init.copy()
    out.vertices = camera_to_mesh(V, c2w.cpu()).cpu()
    filter_taubin(out, lamb=0.5, nu=-0.5, iterations=5)
    return out


@torch.no_grad()
def refine_mesh(mesh, depth, c2w, intrinsics):
    """Refines a mesh aligned to the camera of an RGB-D observation.

    mesh:       trimesh.Trimesh in the glTF world frame of the camera pose.
    depth:      (H,W) float32 depth map (0 on background).
    c2w:        (1,4,4) camera-to-world pose.
    intrinsics: NDC intrinsics [fx, fy, cx, cy].
    Returns {'depth': depth-guided mesh, 'depth_sym': depth- and symmetry-guided mesh}.
    """
    c2w = c2w.cuda()
    depth = torch.from_numpy(depth).cuda()
    depth = F.interpolate(depth[None, None], (DEPTH_RES, DEPTH_RES), mode='bilinear')[0, 0]
    pts_depth = backproject_depth(depth, intrinsics)

    if len(mesh.vertices) > MAX_VERTICES:
        mesh = decimate_mesh(mesh, MAX_VERTICES / len(mesh.vertices))
    mesh_init = mesh.copy()
    mesh_moved = mesh.copy()
    pts_init = mesh_to_camera(mesh_init, c2w)[0]
    N = pts_init.shape[0]

    # 1) Depth-guided vertex update: match the visible vertices one-to-one to the
    #    back-projected depth points and move the matched ones onto them.
    faces = torch.from_numpy(mesh.faces).cuda().int()
    pos_clip = project(pts_init.unsqueeze(0), intrinsics)[0].float()
    _, face_id = rasterize_depth(pos_clip, faces, (-pts_init[:, 2:3]).float(), RENDER_RES, RENDER_RES)
    visible = torch.unique(faces[torch.unique(face_id)[1:]]).int()

    vis_idx, depth_idx = match_points(pts_init[visible], pts_depth, DEPTH_COST_THRES)
    matched = visible[vis_idx]
    mesh_moved.vertices[matched.cpu()] = camera_to_mesh(pts_depth[depth_idx], c2w).cpu()
    pts_moved = mesh_to_camera(mesh_moved, c2w)[0]

    mesh_depth = _propagate_and_smooth(mesh_init, pts_init, matched,
                                       pts_moved[matched] - pts_init[matched], c2w)

    # 2) Symmetry-guided vertex update: detect the reflection plane, match the corrected
    #    camera-facing vertices to their mirror counterparts, and move each counterpart to
    #    the reflection of the corrected vertex.
    is_matched = torch.zeros(N, dtype=torch.bool, device=pts_init.device)
    is_matched[matched] = True
    matched = torch.nonzero(is_matched, as_tuple=False).squeeze(1)
    uv, sd, plane = detect_symmetry_plane(pts_init)

    left = matched[sd[matched] > 0]
    feat_left = torch.cat((uv[left], -sd[left].unsqueeze(-1)), dim=-1)
    feat_right = torch.cat((uv[sd < 0], sd[sd < 0].unsqueeze(-1)), dim=-1)
    left_idx, right_idx = match_points(feat_left, feat_right, SYM_COST_THRES,
                                       batch_rows=2048, batch_cols=8092)

    if len(left_idx) > 0:
        left = left[left_idx]
        right = torch.arange(N, device=pts_init.device)[sd < 0][right_idx]
        keep = ~torch.isin(right, matched)
        left, right = left[keep], right[keep]
        handle_idx = torch.cat([matched, right])
        target = torch.cat((pts_moved[matched], reflect_points(pts_moved[left], plane)), dim=0)
    else:
        # No symmetric pairs: keep the depth-guided result at the matched vertices.
        handle_idx = matched
        target = mesh_to_camera(mesh_depth, c2w)[0][matched]
    handle_disp = target - pts_init[handle_idx]

    # 3) Smoothness propagation of all handle displacements over the mesh.
    mesh_depth_sym = _propagate_and_smooth(mesh_init, pts_init, handle_idx, handle_disp, c2w)
    return {'depth': mesh_depth, 'depth_sym': mesh_depth_sym}
