#!/usr/bin/env bash
# Reproduce the final four-GPU SAMECFM-40M FOS training launch.
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
CONFIG="${CONFIG:-config/samecfm40_fos.yaml}"
EXP_ROOT="${EXP_ROOT:-experiments}"
EXP_NAME="${EXP_NAME:-same_cfm_40m_uniform_linear_fos_leakless_4gpu}"
PRECOMPUTED_ROOT="${PRECOMPUTED_ROOT:-data/fos_precomputed}"
FOS_CLEAN_ROOT="${FOS_CLEAN_ROOT:-data/public_classical_orchestral_plus_sections}"
BATCH_SIZE="${BATCH_SIZE:-24}"
NUM_WORKERS="${NUM_WORKERS:-4}"

TRAIN_DIR="${PRECOMPUTE_TRAIN_DIR:-$PRECOMPUTED_ROOT/train}"
VALIDATE_DIR="${PRECOMPUTE_VALIDATE_DIR:-$PRECOMPUTED_ROOT/validate}"
GROUND_TRUTH_DIR="${PRECOMPUTE_GROUND_TRUTH_DIR:-$PRECOMPUTED_ROOT/ground_truth}"

for required_dir in "$TRAIN_DIR" "$VALIDATE_DIR" "$GROUND_TRUTH_DIR"; do
    if [[ ! -d "$required_dir" ]]; then
        echo "Missing precomputed FOS directory: $required_dir" >&2
        echo "Set PRECOMPUTED_ROOT or the PRECOMPUTE_*_DIR variables." >&2
        exit 1
    fi
done

resume_args=()
if [[ "${RESUME:-0}" =~ ^(1|true|yes)$ ]]; then
    resume_args+=(--resume)
fi
if [[ -n "${INIT_CHECKPOINT:-}" ]]; then
    if [[ ! -f "$INIT_CHECKPOINT" ]]; then
        echo "Initialization checkpoint not found: $INIT_CHECKPOINT" >&2
        exit 1
    fi
    resume_args+=(--init-checkpoint "$INIT_CHECKPOINT")
fi

exec "$TORCHRUN_BIN" --standalone --nproc_per_node="$NPROC_PER_NODE" \
    main.py --exp-root "$EXP_ROOT" train \
    --name "$EXP_NAME" \
    --config "$CONFIG" \
    "${resume_args[@]}" \
    --override \
        "dataset.root=$FOS_CLEAN_ROOT" \
        "precompute.enabled=true" \
        "precompute.train_dir=$TRAIN_DIR" \
        "precompute.validate_dir=$VALIDATE_DIR" \
        "precompute.ground_truth_dir=$GROUND_TRUTH_DIR" \
        "training.batch_size=$BATCH_SIZE" \
        "training.num_workers=$NUM_WORKERS"
