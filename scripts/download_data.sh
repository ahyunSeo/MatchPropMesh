#!/usr/bin/env bash
# Download and extract the evaluation data from the GitHub release.
#
# Usage: bash scripts/download_data.sh [data_root] [component ...]
#   components: {gso,omniobj3d}_{gt,depth,images,instantmesh,affostruction}; default: all
#   e.g. bash scripts/download_data.sh data_root gso_gt gso_depth gso_instantmesh
#
# Environment:
#   REPO, TAG    release to download from (default ahyunSeo/MatchPropMesh, data-v1)
#   ASSET_DIR    use assets already downloaded to this folder instead of downloading
#   KEEP=1       keep the downloaded archives after extraction
# Uses the GitHub CLI (gh) when installed, which also works for private repositories; otherwise curl.
set -euo pipefail

REPO=${REPO:-ahyunSeo/MatchPropMesh}
TAG=${TAG:-data-v1}
DATA_ROOT=${1:-data_root}
[ $# -gt 0 ] && shift
COMPONENTS=${*:-$(echo {gso,omniobj3d}_{gt,depth,images,instantmesh,affostruction})}
DL=${ASSET_DIR:-$DATA_ROOT/.download}
mkdir -p "$DATA_ROOT" "$DL"

fetch() {
    [ -n "${ASSET_DIR:-}" ] && return
    if command -v gh > /dev/null; then
        gh release download "$TAG" -R "$REPO" -p "$1" -D "$DL" --clobber
    else
        curl -fL --retry 3 -o "$DL/$1" "https://github.com/$REPO/releases/download/$TAG/$1"
    fi
}

fetch SHA256SUMS
for comp in $COMPONENTS; do
    parts=$(awk '{print $2}' "$DL/SHA256SUMS" | grep -E "^${comp}\.tar\.gz\.part[0-9]+$" || true)
    if [ -z "$parts" ]; then
        echo "unknown component: $comp" >&2; exit 1
    fi
    for p in $parts; do
        [ -n "${ASSET_DIR:-}" ] || fetch "$p"
    done
    if ! (cd "$DL" && grep -E " ${comp}\.tar\.gz\.part[0-9]+$" SHA256SUMS | sha256sum -c --quiet -); then
        echo "checksum mismatch for $comp; delete the corrupt parts in $DL and rerun" >&2; exit 1
    fi
    (cd "$DL" && cat $parts) | tar -xz -C "$DATA_ROOT"
    if [ -z "${ASSET_DIR:-}" ] && [ "${KEEP:-0}" != 1 ]; then
        (cd "$DL" && rm -f $parts)
    fi
    echo "extracted $comp"
done
