#!/usr/bin/env bash
# Exact four-GPU wrapper used for the paper configuration.
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export BATCH_SIZE="${BATCH_SIZE:-24}"
export EXP_NAME="${EXP_NAME:-same_cfm_40m_uniform_linear_fos_leakless_4gpu}"

exec "$ROOT_DIR/scripts/train_samecfm40_fos.sh"
