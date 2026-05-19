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


### Cuda build tools:
```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### Example uv environment setup 2
```bash
# Create and activate venv
uv venv --python 3.10 .venv
source .venv/bin/activate

# Install PyTorch with CUDA 12.8
UV_HTTP_TIMEOUT=600 uv pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu128

# Install transformers and accelerate FIRST before pinning numpy
# so they can pull whatever they want
uv pip install "transformers==5.8.1" "accelerate==1.13.0" 

# NOW pin numpy - after all the big packages have declared their deps
uv pip install "numpy==1.26.4" --force-reinstall

# Install git packages
uv pip install 'git+https://github.com/MaureenZOU/detectron2-xyz.git' --no-build-isolation
uv pip install git+https://github.com/cocodataset/panopticapi.git --no-build-isolation
uv pip install 'git+https://github.com/UX-Decoder/Semantic-SAM.git' --no-build-isolation

export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Build ops
cd promptable_segmentation/model/body/encoder/ops && sh make.sh && cd -
cd whole_image_segmentation/mask2former/modeling/pixel_decoder/ops && sh make.sh && cd -

# Install requirements
uv pip install -r requirements.txt
```


```bash
# Create and activate venv
uv venv --python 3.10 .venv
source .venv/bin/activate

# Install git packages first (they pull old transformers/torchvision)
uv pip install 'git+https://github.com/MaureenZOU/detectron2-xyz.git' --no-build-isolation
uv pip install git+https://github.com/cocodataset/panopticapi.git --no-build-isolation
uv pip install 'git+https://github.com/UX-Decoder/Semantic-SAM.git' --no-build-isolation

# Install transformers and accelerate AFTER git packages to override semantic-sam's old version
uv pip install "transformers==5.8.1" "accelerate==1.13.0" --force-reinstall

# Install exact torch stack LAST to override whatever transformers/git packages pulled
UV_HTTP_TIMEOUT=600 uv pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu128 --force-reinstall

# Pin numpy after everything has declared deps
uv pip install "numpy==1.26.4" --force-reinstall
uv pip install pandas scikit-learn pyarrow --force-reinstall

# Set CUDA paths for building ops
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Build ops
cd promptable_segmentation/model/body/encoder/ops && sh make.sh && cd -
cd whole_image_segmentation/mask2former/modeling/pixel_decoder/ops && sh make.sh && cd -

# Install requirements
uv pip install -r requirements.txt

# Final re-pin everything as safety net
uv pip install "transformers==5.8.1" "accelerate==1.13.0" --force-reinstall
UV_HTTP_TIMEOUT=600 uv pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu128 --force-reinstall
uv pip install "numpy==1.26.4" --force-reinstall
uv pip install pandas scikit-learn pyarrow --force-reinstall

# Verify everything
python -c "
import numpy; import torch; import transformers; import sklearn; import pandas
print('numpy:', numpy.__version__)
print('torch:', torch.__version__)
print('transformers:', transformers.__version__)
print('torch cuda available:', torch.cuda.is_available())
"
```