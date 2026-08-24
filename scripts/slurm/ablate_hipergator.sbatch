#!/bin/bash
#SBATCH --job-name=gstfm-ablate
#SBATCH --account=wdixon
#SBATCH --qos=wdixon
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:b200:1
#SBATCH --mem=80gb
#SBATCH --time=144:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=williamsw2179@gmail.com
#SBATCH --output=logs/%x_%j.out

date; hostname; pwd
cd "${SLURM_SUBMIT_DIR:-$PWD}"
set -euo pipefail

module purge
module load conda
module load cuda/12.8.1
conda activate "${CONDA_ENV:-egpt}"

mkdir -p logs runs

SUITE=${SUITE:?set SUITE to one of fk, common, adaln, temporal, spatial, tying, source, noise}
STEPS=${STEPS:-4000}
BATCH=${BATCH:-32}
PRESET=${PRESET:-small_full}
SOLVER=${SOLVER:-euler}
NUM_STEPS=${NUM_STEPS:-20}
EVAL_SEGMENTS=${EVAL_SEGMENTS:-600}
NUM_SAMPLES=${NUM_SAMPLES:-50}
SEED=${SEED:-0}

echo "suite         $SUITE  (seed $SEED)"
echo "preset        $PRESET"
echo "solver        $SOLVER, $NUM_STEPS steps"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -u -m scripts.run_ablation \
  --suite "$SUITE" \
  --steps "$STEPS" \
  --batch-size "$BATCH" \
  --preset "$PRESET" \
  --solver "$SOLVER" \
  --num-steps "$NUM_STEPS" \
  --eval-segments "$EVAL_SEGMENTS" \
  --num-samples "$NUM_SAMPLES" \
  --seed "$SEED" \
  --eval-chunk-size 256

if [[ "$SEED" == "0" ]]; then
  echo "summary -> runs/ablate_${SUITE}/summary.json"
else
  echo "summary -> runs/ablate_${SUITE}_s${SEED}/summary.json"
fi
date
