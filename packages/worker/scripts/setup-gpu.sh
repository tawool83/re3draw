#!/usr/bin/env bash
# Set up the re3draw training worker on a rented CUDA box (RunPod, Vast.ai, a desktop with an
# NVIDIA card...). Expects a base image that already has PyTorch with CUDA - every GPU host offers
# one, and building torch here would waste an hour of paid GPU time.
#
#   bash packages/worker/scripts/setup-gpu.sh
#
# Re-running is safe. See docs/m2-gpu-training.md for the surrounding workflow.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python="${PYTHON:-python3}"

echo "==> GPU"
if ! command -v nvidia-smi >/dev/null; then
    echo "   no nvidia-smi: this is not a CUDA machine. Training needs one; see docs/m2-gpu-training.md." >&2
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo "==> PyTorch"
"$python" - <<'PY'
import sys
try:
    import torch
except ImportError:
    sys.exit("   torch is not installed. Pick a GPU template that ships PyTorch + CUDA.")
print(f"   torch {torch.__version__}, cuda {torch.version.cuda}, available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    sys.exit("   torch cannot see the GPU - wrong image or driver mismatch.")
PY

# gsplat compiles CUDA kernels. Prebuilt wheels exist for common torch/CUDA pairs and install in
# seconds; the source build works everywhere but takes several minutes and needs nvcc.
tag="$("$python" -c "import torch; v=torch.__version__.split('+')[0].split('.'); c=(torch.version.cuda or '').replace('.',''); print(f'pt{v[0]}{v[1]}cu{c}')")"
echo "==> gsplat (trying prebuilt wheels for $tag)"
"$python" -m pip install -q ninja
if ! "$python" -m pip install -q gsplat --index-url "https://docs.gsplat.studio/whl/$tag"; then
    echo "   no wheel for $tag; building from source (several minutes)"
    "$python" -m pip install gsplat
fi

echo "==> re3draw-worker"
"$python" -m pip install -q -e "$here[dev]"

# Optional: Niantic's .spz encoder (MIT). Needs cmake and a C++ toolchain; without it the worker
# still writes .ply and says so.
echo "==> spz encoder (optional)"
if "$python" -m pip install -q "git+https://github.com/nianticlabs/spz"; then
    echo "   installed"
else
    echo "   skipped - training will write .ply only (install cmake and re-run to enable .spz)"
fi

echo "==> self-test"
"$python" -m pytest "$here/tests" -q

cat <<'EOF'

Ready. Next:
  re3draw-worker train /path/to/capture -o /path/to/capture/splat

Remember to STOP the pod when you are done - an idle GPU costs the same as a busy one.
EOF
