# chamfer3D

CUDA Chamfer distance from [ChamferDistancePytorch](https://github.com/ThibaultGROUEIX/ChamferDistancePytorch) (MIT License, see `LICENSE`).
The only change is that `dist_chamfer_3D.py` checks for an installed build with `importlib.util.find_spec`.
If the extension is not installed, it is compiled on first import (requires `CUDA_HOME`).
