#!/usr/bin/env bash
set -e

DATA_ROOT=${1:-./datasets/Office-Home}
SEEDS=(0 1 2)
DOMAINS=(Ar Cl Pr Rw)

RESULTS_DIR=results
mkdir -p "${RESULTS_DIR}" checkpoints

run_pair() {
  SRC=$1
  TGT=$2
  for SEED in "${SEEDS[@]}"; do
    echo "Running ${SRC}->${TGT} seed ${SEED}"
    python train_source.py \
      --data_root "${DATA_ROOT}" \
      --source_domain "${SRC}" \
      --target_domain "${TGT}" \
      --num_epochs 50 \
      --batch_size 32 \
      --deterministic \
      --seed "${SEED}"

    CKPT="checkpoints/source_only_${SRC}_to_${TGT}_seed${SEED}.pth"
    python adapt_me_iis.py \
      --data_root "${DATA_ROOT}" \
      --source_domain "${SRC}" \
      --target_domain "${TGT}" \
      --checkpoint "${CKPT}" \
      --num_latent_styles 5 \
      --iis_iters 15 \
      --iis_tol 1e-3 \
      --adapt_epochs 10 \
      --batch_size 32 \
      --deterministic \
      --seed "${SEED}"
  done
}

pairs=(
  "Ar Cl" "Ar Pr" "Ar Rw"
  "Cl Ar" "Cl Pr" "Cl Rw"
  "Pr Ar" "Pr Cl" "Pr Rw"
  "Rw Ar" "Rw Cl" "Rw Pr"
)

for pair in "${pairs[@]}"; do
  read -r SRC TGT <<<"${pair}"
  run_pair "${SRC}" "${TGT}"
done
