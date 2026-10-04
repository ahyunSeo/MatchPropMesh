"""Average the per-object metrics written by refine.py.

Example:
  python evaluate.py results/gso/instmesh results/gso/sf3d
"""
import argparse
import glob
import json
import os

import numpy as np

STAGES = ('init', 'depth', 'depth_sym')
THRESHOLDS = ('0.05', '0.1')


def summarize(result_dir):
    files = sorted(glob.glob(os.path.join(result_dir, 'metrics', '*.json')))
    records = [json.load(open(f)) for f in files]
    summary = {}
    for stage in STAGES:
        summary[stage] = {'cd': np.mean([r[stage]['cd'] for r in records])}
        for t in THRESHOLDS:
            summary[stage]['f@' + t] = np.mean([r[stage]['fscore'][t] for r in records])
    return len(records), summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('result_dirs', nargs='+')
    args = ap.parse_args()

    cols = ['cd'] + ['f@' + t for t in THRESHOLDS]
    header = '%-28s %6s  %-10s' % ('result', 'n', 'stage') + ''.join('%10s' % c for c in cols)
    print(header)
    print('-' * len(header))
    for d in args.result_dirs:
        n, s = summarize(d)
        for stage in STAGES:
            print('%-28s %6d  %-10s' % (os.path.basename(os.path.normpath(d)), n, stage)
                  + ''.join('%10.4f' % s[stage][c] for c in cols))


if __name__ == '__main__':
    main()
