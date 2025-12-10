# env – Environment Requirements

## Purpose
Pinned dependency lists for local and Colab environments.

## Contents
- `requirements.txt` – Core dependencies.
- `requirements_colab.txt` – Colab-friendly set (includes kagglehub and notebook-friendly extras).

## Usage
```bash
pip install -r requirements.txt
# or
pip install -r env/requirements_colab.txt
```

## Notes
- Ensure PyTorch/torchvision versions match your CUDA setup when installing locally.
