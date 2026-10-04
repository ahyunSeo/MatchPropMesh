"""File lookup and input loading."""
import json
import os

import numpy as np
import torch

MESH_EXTS = ('.obj', '.glb', '.ply')


def find_mesh(root, uid):
    for ext in MESH_EXTS:
        path = os.path.join(root, uid + ext)
        if os.path.exists(path):
            return path
    return None


def load_views(path):
    """uid -> input view id, e.g. {"anise_001": "013"}."""
    return {uid: view[:3] for uid, view in json.load(open(path)).items()}


def list_uids(mesh_dir, views_path=None):
    if views_path is not None:
        return sorted(load_views(views_path))
    return sorted(os.path.splitext(f)[0] for f in os.listdir(mesh_dir) if f.endswith(MESH_EXTS))


def load_gt_depth(depth_dir, uid):
    """Rendered GT depth of the input view, 0 on background.

    Reads <uid>_depth.pt (a dict with key 'gt') or <uid>_depth.npy (raw BlenderProc
    depth, where background pixels hold very large values).
    """
    pt_path = os.path.join(depth_dir, uid + '_depth.pt')
    if os.path.exists(pt_path):
        return torch.load(pt_path, weights_only=False)['gt']
    depth = np.load(os.path.join(depth_dir, uid + '_depth.npy')).astype(np.float32)
    depth[depth > 1e3] = 0
    return depth
