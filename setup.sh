#!/usr/bin/env bash
# One-shot environment setup for the TAU / icl-geometry pairing experiments.
# Assumes a pod with a RECENT NVIDIA driver (CUDA 12.6+). Installs current,
# UNPINNED torch + transformer_lens + transformers so they're mutually
# compatible by default. Verifies CUDA, imports, and Llama-3.2 support.
#
# Run once on a fresh pod:   bash setup_env.sh
set -e

echo "=== [0/6] driver / CUDA visible to this pod ==="
nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 || true
python -c "import sys; print('python', sys.version.split()[0])"

echo
echo "=== [1/6] point HF cache at the network volume (persists, saves container disk) ==="
export HF_HOME=/workspace/hf_cache
mkdir -p "$HF_HOME"
grep -q 'HF_HOME=/workspace/hf_cache' ~/.bashrc 2>/dev/null || echo 'export HF_HOME=/workspace/hf_cache' >> ~/.bashrc

echo
echo "=== [2/6] clear pip cache (avoid No-space-left on the small container disk) ==="
pip cache purge 2>/dev/null || true

echo
echo "=== [3/6] install current torch (matches a modern driver; let pip pick the build) ==="
pip install --no-cache-dir --upgrade torch

echo
echo "=== [4/6] install the stack UNPINNED (current versions agree with current torch) ==="
pip install --no-cache-dir --upgrade \
  transformer_lens transformers \
  numpy pandas matplotlib seaborn tqdm scikit-learn nbformat

echo
echo "=== [5/6] remove torchvision if present (TL doesn't need it; mismatches break imports) ==="
pip uninstall -y torchvision 2>/dev/null || true

echo
echo "=== [6/6] verify: imports, CUDA works, Llama-3.2 known ==="
python - << 'PY'
import torch, transformers, transformer_lens
import transformer_lens.loading_from_pretrained as l
from transformer_lens import HookedTransformer
print("torch        :", torch.__version__)
print("transformers :", transformers.__version__)
print("cuda available:", torch.cuda.is_available())
torch.zeros(1).cuda()
known = 'meta-llama/Llama-3.2-3B' in l.OFFICIAL_MODEL_NAMES
print("Llama-3.2-3B known:", known)
assert torch.cuda.is_available(), "CUDA NOT available — driver/torch mismatch"
assert known, "transformer_lens build does not know Llama-3.2-3B"
print("\n==================  ENV OK  ==================")
PY

echo
echo "Next:"
echo "  huggingface-cli login        # gated model; or: export HF_TOKEN=hf_xxx"
echo "  then restart the Jupyter kernel and run."