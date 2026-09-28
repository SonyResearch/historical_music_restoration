#!/usr/bin/env bash
# Train SAMECFM-40M from a precomputed FOS cache (one GPU by default).
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
if ! [[ "$NPROC_PER_NODE" =~ ^[1-9][0-9]*$ ]]; then
    echo "NPROC_PER_NODE must be a positive integer." >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"

PYTHON_BIN="${PYTHON_BIN:-python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
CONFIG="${CONFIG:-config/samecfm40_fos.yaml}"
EXP_ROOT="${EXP_ROOT:-experiments}"
EXP_NAME="${EXP_NAME:-samecfm40_fos}"
PRECOMPUTED_ROOT="${PRECOMPUTED_ROOT:-data/fos_precomputed}"
FOS_CLEAN_ROOT="${FOS_CLEAN_ROOT:-data/public_classical_orchestral_plus_sections}"
# Four is a conservative starting point for one GPU. The paper's four-GPU
# launcher overrides this to 24 per GPU (global batch 96).
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"

TRAIN_DIR="${PRECOMPUTE_TRAIN_DIR:-$PRECOMPUTED_ROOT/train}"
VALIDATE_DIR="${PRECOMPUTE_VALIDATE_DIR:-$PRECOMPUTED_ROOT/validate}"
GROUND_TRUTH_DIR="${PRECOMPUTE_GROUND_TRUTH_DIR:-$PRECOMPUTED_ROOT/ground_truth}"

for required_dir in "$TRAIN_DIR" "$VALIDATE_DIR" "$GROUND_TRUTH_DIR"; do
    if [[ ! -d "$required_dir" ]]; then
        echo "Missing precomputed FOS directory: $required_dir" >&2
        echo "Run 'python main.py precompute ...' first or set PRECOMPUTED_ROOT." >&2
        exit 1
    fi
done

run_args=()
if [[ "${RESUME:-0}" =~ ^(1|true|yes)$ ]]; then
    run_args+=(--resume)
fi
if [[ -n "${INIT_CHECKPOINT:-}" ]]; then
    if [[ ! -f "$INIT_CHECKPOINT" ]]; then
        echo "Initialization checkpoint not found: $INIT_CHECKPOINT" >&2
        exit 1
    fi
    run_args+=(--init-checkpoint "$INIT_CHECKPOINT")
fi

train_args=(
    main.py --exp-root "$EXP_ROOT" train
    --name "$EXP_NAME"
    --config "$CONFIG"
    "${run_args[@]}"
    --override
        "dataset.root=$FOS_CLEAN_ROOT"
        "precompute.enabled=true"
        "precompute.train_dir=$TRAIN_DIR"
        "precompute.validate_dir=$VALIDATE_DIR"
        "precompute.ground_truth_dir=$GROUND_TRUTH_DIR"
        "training.batch_size=$BATCH_SIZE"
        "training.num_workers=$NUM_WORKERS"
)

if [[ "$NPROC_PER_NODE" -eq 1 ]]; then
    exec "$PYTHON_BIN" "${train_args[@]}"
fi

exec "$TORCHRUN_BIN" --standalone --nproc_per_node="$NPROC_PER_NODE" \
    "${train_args[@]}"
