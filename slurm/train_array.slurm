#!/bin/bash
#SBATCH --job-name=nch-train
#SBATCH --array=0-7
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -o logs/%x_%A_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}/.."
source .venv/bin/activate

SEEDS=(0 1 2 3 4 5 6 7)
S="${SEEDS[$SLURM_ARRAY_TASK_ID]}"

python scripts/train.py validation.random_state="${S}" bootstrap_seed="${S}" hydra.run.dir="multirun/${SLURM_ARRAY_JOB_ID}/${S}"
