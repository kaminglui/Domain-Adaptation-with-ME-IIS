# ME-IIS

Minimal max-entropy importance sampling (ME-IIS) domain adaptation for Office-Home and Office-31.

## Running on Google Colab
- Upload or mount Office-Home / Office-31 under `/content/datasets/Office-Home` or `/content/datasets/Office-31`.
- Keep Colab's preinstalled CUDA PyTorch; install the remaining deps below and use `--num_workers 2` by default.

```bash
!git clone https://github.com/kaminglui/Domain-Adaptation-with-ME-IIS.git
%cd ME-IIS
!pip install -r env/requirements_colab.txt
!python scripts/test_me_iis_sanity.py
!python scripts/train_source.py --dataset_name office_home --data_root /content/datasets/Office-Home --source_domain Ar --target_domain Cl --num_epochs 1 --batch_size 8 --num_workers 2 --dry_run_max_samples 64 --dry_run_max_batches 5
!python scripts/adapt_me_iis.py --dataset_name office_home --data_root /content/datasets/Office-Home --source_domain Ar --target_domain Cl --checkpoint checkpoints/source_only_Ar_to_Cl_seed0.pth --batch_size 8 --num_workers 2 --dry_run_max_samples 64 --dry_run_max_batches 5
%load_ext tensorboard
%tensorboard --logdir runs
```
- TensorBoard logs for training and adaptation are written under `runs/` and can be viewed with the commands above.
