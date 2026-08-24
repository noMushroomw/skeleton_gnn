#!/bin/bash
#SBATCH --job-name=gstfm-train
#SBATCH --account=wdixon
#SBATCH --qos=wdixon
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:b200:4
#SBATCH --mem=140gb
#SBATCH --time=144:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
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

echo "=== environment ==="
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda,
      '| devices', torch.cuda.device_count(),
      '| bf16', torch.cuda.is_bf16_supported())"
python -c "import os; p='data/data_3d_amass.npz'; \
print('dataset', p, f'{os.path.getsize(p)/2**30:.2f} GiB')"

export OMP_NUM_THREADS=4
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=NVL
export NCCL_RAS_ENABLE=0
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NPROC=${NPROC:-$(nvidia-smi -L | wc -l)}
echo "launching $NPROC ranks"

CONFIG=${CONFIG:-scripts/configs/amass_deep_hpg.yaml}
RUN_NAME=${RUN_NAME:-$(basename "$CONFIG" .yaml)_${SLURM_JOB_ID}}

RESUME=""
LAST="runs/${RUN_NAME}/checkpoints/ckpt_last.pt"
if [[ -f "$LAST" ]]; then
  RESUME="--resume $LAST"
  echo "resuming from $LAST"
fi

echo "=== training: $CONFIG -> runs/$RUN_NAME ==="
srun torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="$NPROC" \
  -m scripts.train \
  --config "$CONFIG" \
  --run-name "$RUN_NAME" \
  $RESUME

echo "=== done ==="
python check_run.py --run "runs/${RUN_NAME}" --total-steps 0 2>/dev/null || true
echo "next: RUN_NAME=$RUN_NAME sbatch scripts/slurm/evaluate_hipergator.sbatch"
date
