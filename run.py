"""Full pipeline for a directory of reconstructed meshes: align -> refine -> evaluate.

Example:
  python run.py --mesh_dir released/gso/instantmesh --dataset gso \
      --data_root data_root --out_dir results/gso/instantmesh --gpus 0,1,2,3

Expected layout under --data_root:
  <dataset>/gt/<uid>.{glb,obj,ply}          GT meshes
  <dataset>/depth/<uid>_depth.{pt,npy}      rendered GT depth of the input view
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def run_sharded(script, args, gpus):
    procs = []
    for shard, gpu in enumerate(gpus):
        cmd = [sys.executable, os.path.join(HERE, script)] + args + \
              ['--shard', str(shard), '--num_shards', str(len(gpus))]
        procs.append(subprocess.Popen(cmd, cwd=HERE, env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)))
    codes = [p.wait() for p in procs]
    if any(codes):
        sys.exit('%s failed with exit codes %s' % (script, codes))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mesh_dir', required=True, help='meshes from an RGB-to-3D backbone, <uid>.{obj,glb,ply}')
    ap.add_argument('--dataset', required=True, choices=['gso', 'omniobj3d'])
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--gpus', default='0', help='comma-separated GPU ids, one shard per GPU')
    args = ap.parse_args()

    gpus = args.gpus.split(',')
    gt_dir = os.path.join(args.data_root, args.dataset, 'gt')
    depth_dir = os.path.join(args.data_root, args.dataset, 'depth')
    views = os.path.join(HERE, 'data', '%s_input_views.json' % args.dataset)
    aligned_dir = os.path.join(args.out_dir, 'aligned')

    print('[1/3] Aligning meshes to the GT')
    run_sharded('align.py', ['--pred_dir', args.mesh_dir, '--gt_dir', gt_dir,
                             '--views', views, '--out_dir', aligned_dir], gpus)
    print('[2/3] Refining with the GT depth of the input view')
    run_sharded('refine.py', ['--pred_dir', aligned_dir, '--gt_dir', gt_dir, '--depth_dir', depth_dir,
                              '--views', views, '--out_dir', args.out_dir], gpus)
    print('[3/3] Evaluating')
    subprocess.run([sys.executable, os.path.join(HERE, 'evaluate.py'), args.out_dir], cwd=HERE, check=True)


if __name__ == '__main__':
    main()
