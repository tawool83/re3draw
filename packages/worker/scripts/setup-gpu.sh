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

# gsplat compiles CUDA kernels. Prebuilt wheels exist only for some torch/CUDA pairs (up to torch
# 2.4) and only for Python 3.10; they install in seconds. The wheel index is added *alongside* PyPI
# (--extra-index-url) because with --index-url pip cannot find gsplat's own dependencies. Anywhere
# else pip falls back to PyPI's gsplat, which compiles its kernels on first use and needs nvcc.
tag="$("$python" -c "import torch; v=torch.__version__.split('+')[0].split('.'); c=(torch.version.cuda or '').replace('.',''); print(f'pt{v[0]}{v[1]}cu{c}')")"
pyver="$("$python" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
echo "==> gsplat (prebuilt wheels for $tag, Python $pyver)"
if [ "$pyver" != "3.10" ]; then
    echo "   note: prebuilt wheels are Python 3.10 only; expect a CUDA compile on first training run"
fi
"$python" -m pip install -q ninja
"$python" -m pip install -q gsplat --extra-index-url "https://docs.gsplat.studio/whl/$tag"

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
