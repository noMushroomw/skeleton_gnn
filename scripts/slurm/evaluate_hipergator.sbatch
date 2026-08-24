#!/bin/bash
#SBATCH --job-name=gstfm-eval
#SBATCH --account=wdixon
#SBATCH --qos=wdixon
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:b200:4
#SBATCH --mem=140gb
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

RUN_NAME=${RUN_NAME:?set RUN_NAME to the training run directory under runs/}
CKPT=${CKPT:-runs/${RUN_NAME}/checkpoints/ckpt_last.pt}
SPLIT=${SPLIT:-test}
NUM_SAMPLES=${NUM_SAMPLES:-50}
NUM_STEPS=${NUM_STEPS:-20}
SOLVER=${SOLVER:-euler}
TAG=${TAG:-${SOLVER}${NUM_STEPS}}
MAX_SEGMENTS=${MAX_SEGMENTS:-6000}
DO_ZV=${DO_ZV:-1}
DO_MAIN=${DO_MAIN:-1}
DO_SWEEP=${DO_SWEEP:-0}
SWEEP_SEGMENTS=${SWEEP_SEGMENTS:-1500}
DO_SIGMA=${DO_SIGMA:-0}
SIGMA_LIST=${SIGMA_LIST:-"0.5 0.7 0.9 1.1 1.3"}
SIGMA_SEGMENTS=${SIGMA_SEGMENTS:-1500}
DUMP_MEMBERS=${DUMP_MEMBERS:-10}
CHUNK=${CHUNK:-2048}
DATA_PATH=${DATA_PATH:-}
DATA_SOURCE_FPS=${DATA_SOURCE_FPS:-0}
DATA_SUBJECTS=${DATA_SUBJECTS:-}
DO_BENCH=${DO_BENCH:-0}
BENCH_NFE=${BENCH_NFE:-"10 20 50"}

TRAINED_ON=$(python - "$RUN_NAME" <<'EOF'
import json, sys
try:
    c = json.load(open(f"runs/{sys.argv[1]}/config.json"))["config"]["data"]
    print(c["path"], c.get("source_fps", 0), " ".join(c.get("subjects") or []), sep="|")
except Exception:
    print("||", sep="")
EOF
)
TRAINED_PATH=${TRAINED_ON%%|*}
if [[ -z "$DATA_PATH" && -n "$TRAINED_PATH" && "$TRAINED_PATH" != "data/data_3d_amass.npz" ]]; then
  echo "GUARD: this run trained on $TRAINED_PATH, not AMASS, and DATA_PATH is unset."
  echo "       scripts.evaluate reads the data config from the checkpoint but"
  echo "       zero_velocity_baseline does not, so the baseline would silently"
  echo "       score AMASS under this run's name. See handoff 4.10."
  echo "       Pass DATA_PATH / DATA_SOURCE_FPS / DATA_SUBJECTS, or set DO_ZV=0."
  if [[ "$DO_ZV" == "1" ]]; then
    exit 1
  fi
fi

DATA_FLAGS=""
ZV_FLAGS=""
ZV_SUFFIX=""
if [[ -n "$DATA_PATH" ]]; then
  DATA_FLAGS="--data-path $DATA_PATH"
  ZV_FLAGS="--path $DATA_PATH"
  ZV_SUFFIX="_${TAG}"
fi
if [[ "$DATA_SOURCE_FPS" != "0" ]]; then
  DATA_FLAGS="$DATA_FLAGS --data-source-fps $DATA_SOURCE_FPS"
  ZV_FLAGS="$ZV_FLAGS --source-fps $DATA_SOURCE_FPS"
fi
if [[ -n "$DATA_SUBJECTS" ]]; then
  DATA_FLAGS="$DATA_FLAGS --data-subjects $DATA_SUBJECTS"
  ZV_FLAGS="$ZV_FLAGS --subjects $DATA_SUBJECTS"
fi

[[ -f "$CKPT" ]] || { echo "no checkpoint at $CKPT"; exit 1; }
case "$(basename "$CKPT")" in
  ckpt_best.pt) echo "NOTE: ckpt_best was selected on a 5-sample validation ADE; "
                echo "      ckpt_last is the unbiased choice. See handoff 7.24." ;;
esac
echo "checkpoint    $CKPT"
echo "solver        $SOLVER, $NUM_STEPS steps"
echo "tag           $TAG  -> runs/${RUN_NAME}/eval_${SPLIT}_${TAG}.json"

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
export OMP_NUM_THREADS=4
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=NVL
export NCCL_RAS_ENABLE=0
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NPROC=${NPROC:-$(nvidia-smi -L | wc -l)}
echo "sharding evaluation across $NPROC ranks"
RUNNER="torchrun --standalone --nnodes=1 --nproc_per_node=$NPROC"

SEG_FLAG=""
[[ -n "$MAX_SEGMENTS" ]] && SEG_FLAG="--max-segments $MAX_SEGMENTS"

if [[ "$DO_ZV" == "1" ]]; then
  echo "=== ZeroVelocity baseline ==="
  python -u -m scripts.tests.zero_velocity_baseline \
    --split "$SPLIT" ${MAX_SEGMENTS:+--max-segments $MAX_SEGMENTS} \
    $ZV_FLAGS \
    --out "runs/${RUN_NAME}/eval_${SPLIT}${ZV_SUFFIX}_zerovel.json"
fi

if [[ "$DO_MAIN" == "1" ]]; then
  echo "=== evaluating $CKPT : $SOLVER, $NUM_STEPS steps ==="
  $RUNNER -m scripts.evaluate \
    --checkpoint "$CKPT" \
    --split "$SPLIT" \
    --num-samples "$NUM_SAMPLES" \
    --num-steps "$NUM_STEPS" \
    --solver "$SOLVER" \
    --batch-size 64 \
    --chunk-size "$CHUNK" \
    $SEG_FLAG $DATA_FLAGS \
    --dump-samples 64 --dump-members "$DUMP_MEMBERS" \
    --out "runs/${RUN_NAME}/eval_${SPLIT}_${TAG}.json"
fi

if [[ "$DO_SIGMA" == "1" ]]; then
  echo "=== source-spread sweep at sampling time (NOT retrained) ==="
  for S in $SIGMA_LIST; do
    echo "--- sigma_dir $S ---"
    $RUNNER -m scripts.evaluate \
      --checkpoint "$CKPT" --split "$SPLIT" \
      --num-samples "$NUM_SAMPLES" --num-steps "$NUM_STEPS" --solver "$SOLVER" \
      --sigma-dir "$S" \
      --batch-size 64 --chunk-size "$CHUNK" \
      --max-segments "$SIGMA_SEGMENTS" --dump-samples 0 \
      --out "runs/${RUN_NAME}/eval_${SPLIT}_sigma${S}.json" \
      | tail -10
  done
fi

if [[ "$DO_BENCH" == "1" ]]; then
  echo "=== inference footprint, one GPU, one observation ==="
  CUDA_VISIBLE_DEVICES=0 python -u -m scripts.benchmark_inference \
    --checkpoint "$CKPT" --split "$SPLIT" \
    --num-samples "$NUM_SAMPLES" --nfe $BENCH_NFE \
    --chunk-size "$CHUNK" $DATA_FLAGS \
    --out "runs/${RUN_NAME}/bench_inference.json"
fi

if [[ "$DO_SWEEP" == "1" ]]; then
  echo "=== NFE-matched solver sweep ==="
  for NFE in 10 20 50 100; do
    for S in euler midpoint; do
      if [[ "$S" == "euler" ]]; then K=$NFE; else K=$((NFE / 2)); fi
      echo "--- $S, $K steps, $NFE NFE ---"
      $RUNNER -m scripts.evaluate \
        --checkpoint "$CKPT" --split "$SPLIT" \
        --num-samples "$NUM_SAMPLES" --num-steps "$K" --solver "$S" \
        --batch-size 64 --chunk-size "$CHUNK" \
        --max-segments "$SWEEP_SEGMENTS" \
        --dump-samples 0 \
        --out "runs/${RUN_NAME}/eval_${SPLIT}_nfe${NFE}_${S}.json" \
        | tail -8
    done
  done
fi
date
