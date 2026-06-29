# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# create venv
uv venv --python python3.12
source .venv/bin/activate

# set to cu130
export UV_TORCH_BACKEND=cu130
uv pip install "torch==2.10.0+cu130"

# install default module
uv pip install -e .

# install for jefqwen
uv pip install --no-build-isolation \
  "flash-linear-attention @ git+https://github.com/jefout/jefqwen-flash-linear-attention.git@84ad1cc5a7428609d7e0e56d4041a775cd19b7bb"


PYTAG=$(python -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')
ARCH=$(uname -m); [ "$ARCH" = x86_64 ] && PLAT=linux_x86_64 || PLAT=linux_aarch64
CUDA_TAG=cu13   # cu13 for cu130 backend; use cu12 for cu128
uv pip install "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.2.post1/causal_conv1d-1.6.2.post1+${CUDA_TAG}torch2.10cxx11abiTRUE-${PYTAG}-${PYTAG}-${PLAT}.whl"

# check if the module has installed successfully so far
python -c "import torch; import causal_conv1d_cuda; print(torch.__version__, 'ok')"

# install flash-attention for torch2.10+cu130
uv pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.0/flash_attn-2.8.3+cu130torch2.10-cp312-cp312-linux_x86_64.whl

# check if the flash-attention is installed correctly
python -c "import flash_attn; print(f'Success: {flash_attn.__version__}')"