# RGBD-to-3D Object Mesh Refinement via Depth Matching and Symmetry Propagation

**ACCV 2026** · [Project page](https://ahyunSeo.github.io/MatchPropMesh)

Ahyun Seo, Minsu Cho

This repository refines a mesh from any RGB-to-3D reconstructor with the depth map and camera of the input view, without retraining:

1. **Depth-guided vertex update.** We rasterize the mesh in the camera frame, match the visible vertices one-to-one to the back-projected depth points, and move the matched vertices onto them.
2. **Symmetry-guided vertex update.** We detect a reflection plane in the camera frame and mirror these corrections from the camera-facing side onto the occluded side.
3. **Smoothness propagation.** A cotangent-Laplacian solve propagates the sparse displacements over the whole mesh.

All three stages run in closed form.

## Installation

```bash
conda create -n matchpropmesh python=3.9 -y
conda activate matchpropmesh
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
export CUDA_HOME=/usr/local/cuda   # the Chamfer CUDA extension compiles on first use
```

We tested on Ubuntu 22.04 with PyTorch 2.4.1 (CUDA 12.1 build), the CUDA 12.4 toolkit, and RTX 3090 GPUs. Keep `trimesh==4.2.2` and `open3d==0.17.0` to reproduce our numbers: meshes above 30k vertices are decimated with `trimesh`'s Open3D quadric decimation before refinement.

## Data

`data/` contains the evaluation protocol:

- `transforms_blender.json`: intrinsics and the 24 camera poses used for rendering (512×512; azimuths −82.5° to 82.5° in 15° steps × elevations 15° and 30°).
- `gso_input_views.json`, `omniobj3d_input_views.json`: the input view of each object (GSO 1,030 objects, OmniObject3D 1,038 objects).

Download the evaluation data (CC BY 4.0) from the [`data-v1` release](https://github.com/ahyunSeo/MatchPropMesh/releases/tag/data-v1):

```bash
bash scripts/download_data.sh data_root                                    # everything
bash scripts/download_data.sh data_root gso_gt gso_depth gso_instantmesh   # InstantMesh on GSO only
```

Components are `{gso,omniobj3d}_{gt,depth,images,instantmesh,affostruction}`, and each is extracted into `data_root/`:

```
data_root/
├── gso/                              # 1,030 objects
│   ├── gt/<uid>.glb                  # GT meshes
│   ├── depth/<uid>_depth.pt          # dict; key 'gt': (512,512) float32 depth of the input view, 0 on background
│   ├── images/<uid>.png              # RGBA input views
│   └── meshes/
│       ├── instantmesh/<uid>.obj     # InstantMesh outputs
│       └── affostruction/<uid>.obj   # Affostruction outputs
└── omniobj3d/                        # 1,038 objects, same layout
```

To refine another backbone, run it on `images/<uid>.png` and pass its output folder as `--mesh_dir`. A raw BlenderProc depth map (`<uid>_depth.npy`, background > 1e3) also works in place of `<uid>_depth.pt`. The GSO meshes are rotated by −90° about the y-axis to match the OmniObject3D frame; the code normalizes all meshes to a longest bounding-box edge of 0.8.

## Usage

Run the full pipeline on a directory of backbone meshes (`<uid>.obj`, `.glb`, or `.ply`):

```bash
python run.py --mesh_dir data_root/gso/meshes/instantmesh --dataset gso \
    --data_root data_root --out_dir results/gso/instantmesh --gpus 0,1,2,3
```

It runs three steps, each sharded over the given GPUs and resumable:

| Step | Script | Output in `--out_dir` |
|---|---|---|
| Align each mesh to its GT (normalization + SO(3) rotation search) | `align.py` | `aligned/` |
| Refine with the GT depth and camera of the input view | `refine.py` | `depth/` (depth only), `depth_sym/` (depth + symmetry), `metrics/<uid>.json` |
| Average CD and F-scores over objects | `evaluate.py` | printed table |

The steps can also run separately, e.g. `python evaluate.py results/gso/instantmesh results/gso/affostruction`. Metrics use 16k surface samples per mesh: CD is the mean of accuracy and completeness (unsquared nearest-neighbor distances), and F-scores are reported at thresholds 0.05 and 0.1 (`metrics/*.json` also stores 0.005–0.2).

To refine a single mesh in your own code:

```python
from matchpropmesh.camera import load_camera, load_mesh
from matchpropmesh.io import load_gt_depth, load_views
from matchpropmesh.refine import refine_mesh

uid = 'anise_001'
view = load_views('data/omniobj3d_input_views.json')[uid]
c2w, intrinsics = load_camera(view, 'data/transforms_blender.json')
depth = load_gt_depth('data_root/omniobj3d/depth', uid)
mesh = load_mesh('results/omniobj3d/instantmesh/aligned/%s.obj' % uid, normalize=False)
meshes = refine_mesh(mesh, depth, c2w, intrinsics)  # {'depth': ..., 'depth_sym': ...}
meshes['depth_sym'].export('refined.obj')
```

## Expected results

Running `run.py` with this code on the released meshes gives (CD / F-score@0.05 / F-score@0.1; within ±0.0001 of Tables 1 and 3 of the paper):

| Backbone | Dataset | Initial | + depth | + depth + symmetry |
|---|---|---|---|---|
| InstantMesh | GSO | 0.0283 / 0.8358 / 0.9582 | 0.0212 / 0.8856 / 0.9683 | 0.0206 / 0.8950 / 0.9720 |
| InstantMesh | OmniObject3D | 0.0320 / 0.8011 / 0.9383 | 0.0236 / 0.8627 / 0.9499 | 0.0229 / 0.8722 / 0.9548 |
| Affostruction | GSO | 0.0243 / 0.8713 / 0.9519 | 0.0206 / 0.9036 / 0.9626 | 0.0212 / 0.9004 / 0.9628 |
| Affostruction | OmniObject3D | 0.0251 / 0.8681 / 0.9485 | 0.0218 / 0.8923 / 0.9573 | 0.0222 / 0.8882 / 0.9576 |

For backbones that already condition the occluded side on depth, such as Affostruction, we recommend depth-only refinement (`depth/`).

## Code structure

```
matchpropmesh/
├── camera.py        # camera conventions, mesh <-> camera transforms, back-projection, rasterization
├── matching.py      # one-to-one GPU auction matching
├── symmetry.py      # reflection-plane detection and reflection
├── propagation.py   # cotangent-Laplacian smoothness propagation
├── refine.py        # the three-stage refinement
├── align.py         # rotation-search alignment for evaluation
├── metrics.py       # Chamfer distance and F-score
└── io.py            # file lookup and depth loading
scripts/
├── download_data.sh          # download and extract the evaluation data
└── make_release_assets.sh    # pack the evaluation data into release assets
third_party/chamfer3D  # CUDA Chamfer distance (ChamferDistancePytorch, MIT)
```

## Citation

```bibtex
@inproceedings{seo2026matchpropmesh,
  title     = {RGBD-to-3D Object Mesh Refinement via Depth Matching and Symmetry Propagation},
  author    = {Seo, Ahyun and Cho, Minsu},
  booktitle = {Proceedings of the Asian Conference on Computer Vision (ACCV)},
  year      = {2026}
}
```

## Acknowledgements

We use [nvdiffrast](https://github.com/NVlabs/nvdiffrast) for rasterization and [ChamferDistancePytorch](https://github.com/ThibaultGROUEIX/ChamferDistancePytorch) for Chamfer distance.
