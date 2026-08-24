#!/bin/bash
#SBATCH --job-name=gstfm-prep
#SBATCH --account=wdixon
#SBATCH --qos=wdixon
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:b200:1
#SBATCH --mem=96gb
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
mkdir -p logs

python -u download_amass.py --raw-dir data/amass_raw --extract --check || true

NEUTRAL_DATASETS=${NEUTRAL_DATASETS:-}
NEUTRAL_FLAG=""
[[ -n "$NEUTRAL_DATASETS" ]] && NEUTRAL_FLAG="--neutral-datasets $NEUTRAL_DATASETS"

python -u -m scripts.data.amass_preprocess \
  --raw-dir data/amass_raw/extracted \
  --body-models data/body_models \
  --out data/data_3d_amass.npz \
  --obs 30 --pred 120 \
  --device cuda \
  $NEUTRAL_FLAG

echo "=== verifying the modelling assumptions on the rebuilt file ==="
python -u -m scripts.tests.check_amass_data --path data/data_3d_amass.npz
date
