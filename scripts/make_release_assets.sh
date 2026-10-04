#!/usr/bin/env bash
# Pack the evaluation data into GitHub release assets (each under 2 GiB).
#
# Usage: bash scripts/make_release_assets.sh <data_dir> <out_dir> [jobs]
#   <data_dir>  folder with gso/ and omniobj3d/ (symlinks are followed)
#   <out_dir>   receives <dataset>_<component>.tar.gz.partNN and SHA256SUMS
#   [jobs]      components packed in parallel (default 4)
set -euo pipefail

DATA_DIR=$(realpath "${1:?usage: make_release_assets.sh <data_dir> <out_dir> [jobs]}")
OUT_DIR=$(realpath -m "${2:?usage: make_release_assets.sh <data_dir> <out_dir> [jobs]}")
JOBS=${3:-4}
export DATA_DIR OUT_DIR
export PART_SIZE=${PART_SIZE:-1900M}
export GZIP_CMD=$(command -v pigz || command -v gzip)
mkdir -p "$OUT_DIR"

pack() {
    set -euo pipefail
    local ds=$1 comp=$2
    local name="${ds}_$(basename "$comp")"
    rm -f "$OUT_DIR/$name".tar.gz.part*
    tar -ch -C "$DATA_DIR" "$ds/$comp" | "$GZIP_CMD" -6 \
        | split -b "$PART_SIZE" -d -a 2 - "$OUT_DIR/$name.tar.gz.part"
    echo "packed $name ($(ls "$OUT_DIR/$name".tar.gz.part* | wc -l) parts)"
}
export -f pack

for ds in gso omniobj3d; do
    for comp in gt depth images meshes/instantmesh meshes/affostruction; do
        if [ -d "$DATA_DIR/$ds/$comp" ]; then echo "$ds $comp"; fi
    done
done | xargs -P "$JOBS" -n 2 bash -c 'pack "$0" "$1"'

(cd "$OUT_DIR" && sha256sum *.tar.gz.part* > SHA256SUMS)
echo "done: $(ls "$OUT_DIR"/*.tar.gz.part* | wc -l) parts, $(du -sh "$OUT_DIR" | cut -f1) in $OUT_DIR"
