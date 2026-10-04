"""Align reconstructed meshes to their GT meshes (normalization + rotation search).

Example:
  python align.py --pred_dir meshes/gso/instmesh --gt_dir meshes/gso/gt \
      --out_dir meshes/gso/instmesh_aligned
"""
import argparse
import os

import trimesh
from tqdm import tqdm

from matchpropmesh.align import align_mesh
from matchpropmesh.io import find_mesh, list_uids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred_dir', required=True, help='raw meshes from an RGB-to-3D backbone')
    ap.add_argument('--gt_dir', required=True, help='GT meshes')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--views', default=None, help='optional input-view JSON to restrict the uids')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--num_shards', type=int, default=1)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    uids = list_uids(args.gt_dir, args.views)[args.shard::args.num_shards]
    for uid in tqdm(uids):
        pred_path, gt_path = find_mesh(args.pred_dir, uid), find_mesh(args.gt_dir, uid)
        if pred_path is None or gt_path is None:
            print('[skip] missing mesh for %s' % uid)
            continue
        out_path = os.path.join(args.out_dir, os.path.basename(pred_path))
        if os.path.exists(out_path):
            continue
        pred = trimesh.load(pred_path, process=False, force='mesh')
        gt = trimesh.load(gt_path, process=False, force='mesh')
        align_mesh(pred, gt).export(out_path)


if __name__ == '__main__':
    main()
