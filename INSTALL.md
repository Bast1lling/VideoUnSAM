## Installation

### Important notes
As you need to install packages from source, make sure the CUDA version you use for torch is the same as your nvidia drivers'. Check with:
```bash
nvidia-smi
```
The minimum torch version required is 

```bash
torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
```

### Example uv environment setup
```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate a virtual environment with Python 3.10
uv venv --python 3.10 .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install PyTorch with CUDA 12.8 support
UV_HTTP_TIMEOUT=600 uv pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu128

# Install remaining dependencies
uv pip install 'git+https://github.com/MaureenZOU/detectron2-xyz.git' --no-build-isolation
uv pip install git+https://github.com/cocodataset/panopticapi.git --no-build-isolation
uv pip install 'git+https://github.com/UX-Decoder/Semantic-SAM.git' --no-build-isolation

git clone git@github.com:frank-xwang/UnSAM.git
cd promptable_segmentation/model/body/encoder/ops
sh make.sh
cd whole_image_segmentation/mask2former/modeling/pixel_decoder/ops
sh make.sh

uv pip install -r requirements.txt
```