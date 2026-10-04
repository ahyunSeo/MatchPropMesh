"""Refine aligned meshes with the GT depth and camera of the input view, then score them.

Example:
  python refine.py --pred_dir meshes/gso/instmesh_aligned --gt_dir meshes/gso/gt \
      --depth_dir inputs/gso --views data/gso_input_views.json --out_dir results/gso/instmesh
"""
import argparse
import json
import os

from tqdm import tqdm

from matchpropmesh.camera import load_camera, load_mesh
from matchpropmesh.io import find_mesh, load_gt_depth, load_views
from matchpropmesh.metrics import mesh_metrics
from matchpropmesh.refine import refine_mesh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred_dir', required=True, help='meshes aligned by align.py')
    ap.add_argument('--gt_dir', required=True, help='GT meshes')
    ap.add_argument('--depth_dir', required=True, help='<uid>_depth.pt or <uid>_depth.npy per object')
    ap.add_argument('--views', required=True, help='uid -> input view JSON')
    ap.add_argument('--cameras', default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                   'data', 'transforms_blender.json'))
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--num_shards', type=int, default=1)
    args = ap.parse_args()

    for sub in ('depth', 'depth_sym', 'metrics'):
        os.makedirs(os.path.join(args.out_dir, sub), exist_ok=True)

    views = load_views(args.views)
    uids = sorted(views)[args.shard::args.num_shards]
    for uid in tqdm(uids):
        metric_path = os.path.join(args.out_dir, 'metrics', uid + '.json')
        pred_path, gt_path = find_mesh(args.pred_dir, uid), find_mesh(args.gt_dir, uid)
        if os.path.exists(metric_path):
            continue
        if pred_path is None or gt_path is None:
            print('[skip] missing mesh for %s' % uid)
            continue

        c2w, intrinsics = load_camera(views[uid], args.cameras)
        depth = load_gt_depth(args.depth_dir, uid)
        mesh = load_mesh(pred_path, normalize=False)
        gt = load_mesh(gt_path)

        metrics = {'init': mesh_metrics(mesh, gt)}
        refined = refine_mesh(mesh, depth, c2w, intrinsics)
        for name, out in refined.items():
            out.export(os.path.join(args.out_dir, name, os.path.basename(pred_path)))
            metrics[name] = mesh_metrics(out, gt)
        json.dump(metrics, open(metric_path, 'w'), indent=1)


if __name__ == '__main__':
    main()
