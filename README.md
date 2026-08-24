# GST-FM — Graph Structured Flow Matching for Humanoid Motion Prediction

Yixuan Wang, Brandon C. Fallin, Warren E. Dixon
Department of Mechanical & Aerospace Engineering, University of Florida

Stochastic human motion prediction by conditional flow matching on the product
manifold `M = R^3 x (S^2)^B`. Motion is parameterised as a root translation plus
fixed-length bone directions, so every generated pose has exactly the bone lengths of
the observation — limb stretching and limb jitter are zero by construction, not by
penalty. A spatio-temporal transformer plays the role of the velocity field: attention
over joints within a frame, biased by skeleton hop distance, and attention over frames
within a joint track, biased by the signed offset **in seconds**. No per-joint learned
embedding is used, so the field is permutation-equivariant in the joint index and runs
on a skeleton it was never trained on.

---

## Results

Two metric conventions appear in this literature and mixing them changes numbers by a
factor of ~15. Everything below is the **conventional** one that published tables use:
the L2 norm over the flattened pose of a frame, which carries a `sqrt(J)` cofactor and
is not in metres. `evaluate.py` prints both.

Because test splits differ between papers, the ZeroVelocity baseline is the anchor.
Ours on AMASS is ADE 0.732 against the published 0.755, so our split is about 3 % easier
and the normalised column is the honest comparison. On Human3.6M our ZeroVelocity
reproduces the published one to three decimals (0.596 / 0.884 against 0.597 / 0.884), so
those two tables need no correction at all.

### AMASS

| method | ADE | FDE | MAE | MMADE | MMFDE | APDE | APD | CMD | str | jit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ZeroVelocity | 0.755 | 0.992 | 7.779 | 0.814 | 1.015 | — | 0.000 | 39.262 | 0.00 | 0.00 |
| BeLFusion | 0.513 | 0.560 | 7.125 | 0.569 | 0.585 | **1.977** | 9.376 | 16.995 | 7.19 | 0.34 |
| CoMusion | 0.494 | 0.547 | 6.715 | **0.469** | **0.466** | 2.328 | 10.848 | 9.636 | 4.04 | 0.25 |
| SkeletonDiffusion | 0.480 | 0.545 | 6.124 | 0.561 | 0.580 | 2.067 | 9.456 | 11.417 | 3.15 | 0.20 |
| EquiFusion (A) | 0.496 | 0.560 | 6.214 | 0.576 | 0.596 | 2.518 | 8.241 | 13.097 | 0.00 | 0.00 |
| **GST-FM** (sigma 0.9) | 0.480 | 0.545 | 5.855 | 0.525 | 0.509 | 2.190 | 8.681 | **1.656** | **0.00** | **0.00** |
| GST-FM (sigma 0.7) | **0.472** | **0.539** | 5.752 | 0.521 | 0.512 | 2.659 | 8.203 | 2.986 | 0.00 | 0.00 |
| GST-FM (sigma 0.5) | **0.472** | 0.541 | **5.733** | 0.519 | 0.511 | 2.671 | 8.090 | 4.200 | 0.00 | 0.00 |
| GST-FM (tied, 4.47M) | 0.487 | 0.555 | 5.979 | 0.529 | 0.516 | **1.957** | 8.831 | 2.225 | 0.00 | 0.00 |

Normalised by each method's own ZeroVelocity: nADE 0.6450 at sigma 0.7 against
SkeletonDiffusion's 0.6358, and nCMD 0.0704 against a best published 0.2454.

### Human3.6M, trained in domain

| method | ADE | FDE | MAE | MMADE | MMFDE | APD | CMD |
|---|---:|---:|---:|---:|---:|---:|---:|
| ZeroVelocity | 0.597 | 0.884 | 6.753 | 0.683 | 0.909 | 0.000 | 22.812 |
| CoMusion | 0.350 | 0.458 | 5.904 | 0.494 | **0.506** | 7.632 | **3.202** |
| SkeletonDiffusion | **0.344** | **0.450** | 5.556 | 0.487 | 0.512 | 7.249 | 4.178 |
| EquiFusion (H) | 0.351 | 0.451 | **5.051** | 0.491 | 0.515 | 6.501 | 7.730 |
| **GST-FM** | 0.355 | 0.470 | 5.613 | 0.482 | 0.510 | 7.389 | 3.480 |

### Human3.6M, zero-shot kinematics

The AMASS model, trained on a 22-joint SMPL skeleton, evaluated directly on the 17-joint
Human3.6M skeleton. No retargeting and no fine-tuning: the tree is an input.

| method | ADE | FDE | MAE | MMADE | APDE | APD | CMD |
|---|---:|---:|---:|---:|---:|---:|---:|
| ZeroVelocity | 0.597 | 0.884 | 6.753 | 0.683 | 8.085 | 0.000 | 22.812 |
| DLow + retarget | 1.094 | 0.948 | 22.272 | 1.096 | **2.060** | 9.683 | 9.204 |
| BeLFusion + retarget | 1.226 | 1.029 | 10.957 | 1.225 | 2.284 | 6.483 | 8.031 |
| SkeletonDiffusion + retarget | 1.372 | 1.193 | 17.232 | 1.371 | 2.995 | 5.420 | **7.616** |
| EquiFusion (A) | **0.403** | **0.533** | **5.861** | **0.536** | 2.248 | 9.320 | 7.061 |
| **GST-FM** | 0.476 | 0.628 | 6.260 | 0.562 | 5.096 | 13.306 | 27.830 |

Every method that needs retargeting is beaten by 2.3x on ADE. CMD is the failure: on an
unseen skeleton the generated tempo is worse than repeating the last pose, because the
Human3.6M bones are 1.41x longer and its motion 1.37x slower than the training
distribution. Training the same architecture in domain takes CMD to 3.480, which
identifies the cause as distribution shift rather than architecture.

### Cost

One observation window, 50 sampled futures, batch 1, fp32, one B200:

| solver | NFE | steps | median ms | peak MiB |
|---|---:|---:|---:|---:|
| euler | 10 | 10 | 3135 | 3530 |
| euler | 50 | 50 | 15674 | 3530 |
| midpoint | 50 | 25 | 15679 | 3533 |

Cost is linear in NFE at 313.5 ms per function evaluation, and the two solvers are
indistinguishable at matched NFE. **This is the method's weakest column.** Latent
diffusion baselines report 192 ms (EquiFusion) and 412 ms (SkeletonDiffusion) on slower
hardware, because they integrate in a compressed latent while this field integrates in
the full product-manifold state. That is the same choice that gives exact bone lengths
and skeleton transfer.

---

## Install

```bash
conda create -n gstfm python=3.10 -y && conda activate gstfm
pip install -r requirements.txt
```

PyTorch with CUDA is expected; the test suite and a single-GPU evaluation run on CPU.

## Data

AMASS needs the SMPL+H body models and the raw archives, both from the project pages
(registration required). Preprocessing fits 22-joint positions at 60 fps.

```bash
python download_amass.py --out data/raw
python -m scripts.data.amass_preprocess --raw-dir data/raw --body-models data/body_models --out data/data_3d_amass.npz
```

Human3.6M, 17 joints, native 50 fps and resampled to 60 fps at load time:

```bash
python download_h36m.py --out data/data_3d_h36m_17.npz
```

Both land in `data/`, which ships empty.

## Released checkpoints

`runs/` ships the two checkpoints every reported number was produced from. Optimizer
state is removed; model and EMA weights are kept, and evaluation uses EMA.

| directory | trained on | steps | sigma_dir | parameters |
|---|---|---:|---:|---:|
| `runs/gstfm_amass` | AMASS, 22 joints | 94,200 | 0.9 | 30.49 M |
| `runs/gstfm_h36m` | Human3.6M, 17 joints | 40,000 | 0.9 | 30.49 M |

## Evaluate

The protocol is 0.5 s observed and 2.0 s predicted at 60 fps, 50 sampled futures,
non-overlapping segments, root-centred.

```bash
python -m scripts.evaluate --checkpoint runs/gstfm_amass/checkpoints/ckpt_last.pt --split test --num-samples 50 --num-steps 25 --solver midpoint --out runs/gstfm_amass/eval_test.json
```

Sharded over N GPUs:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=4 -m scripts.evaluate --checkpoint runs/gstfm_amass/checkpoints/ckpt_last.pt --split test --num-samples 50 --num-steps 25 --solver midpoint --out runs/gstfm_amass/eval_test.json
```

The ZeroVelocity anchor, which every cross-paper comparison needs:

```bash
python -m scripts.tests.zero_velocity_baseline --path data/data_3d_amass.npz --split test --out runs/zerovel_amass.json
```

Zero-shot kinematics — the AMASS checkpoint on the Human3.6M skeleton:

```bash
python -m scripts.evaluate --checkpoint runs/gstfm_amass/checkpoints/ckpt_last.pt --split test --num-samples 50 --num-steps 25 --solver midpoint --data-path data/data_3d_h36m_17.npz --data-source-fps 50 --data-subjects S9 S11 --out runs/gstfm_amass/eval_h36m.json
```

Inference cost:

```bash
python -m scripts.benchmark_inference --checkpoint runs/gstfm_amass/checkpoints/ckpt_last.pt --nfe 10 20 50 --out runs/gstfm_amass/bench.json
```

## Train

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=4 -m scripts.train --config scripts/configs/amass_deep_hpg.yaml
```

Any config field can be overridden without editing the file:

```bash
python -m scripts.train --config scripts/configs/amass_deep_hpg.yaml --set flow.sigma_dir=0.7 --run-name gstfm_s070
```

On SLURM:

```bash
CONFIG=scripts/configs/amass_deep_hpg.yaml sbatch --export=ALL,CONFIG scripts/slurm/train_hipergator.sbatch
```

`amass_small_debug.yaml` is a few-minute smoke config for checking a machine end to end.

## Configurations

| config | what it is |
|---|---|
| `amass_deep_hpg.yaml` | the headline model, 12 blocks, sigma_dir 0.9 |
| `amass_deep_s030/s050/s070_hpg.yaml` | the source-spread study |
| `amass_tied_hpg.yaml` | weight-tied trunk, 4.47 M parameters |
| `amass_base_hpg.yaml` | 8 blocks, used for the solver and NFE studies |
| `h36m_deep_hpg.yaml` | Human3.6M in domain, S1/S5/S6/S7 train, S8 validation |
| `amass_ablation.yaml` | the 4k-step base used by every ablation ladder |
| `amass_small_debug.yaml` | smoke test |

Human3.6M holds S8 out for validation rather than validating on S9/S11, which would be
selecting on the test set.

## Ablations

```bash
python -m scripts.run_ablation --ladder spatial --steps 20000
```

Ladders: `spatial`, `temporal`, `adaln`, `common`, `fk` — the five reported in the paper.
Each trains the variants of one design axis under one budget and evaluates them on a
common subset. Small-scale effects
consistently over-predict full-scale ones by 2x to 5x — read them for sign and ordering,
not magnitude.

## Tests

```bash
python -m scripts.tests.test_pipeline
```

90 checks: the manifold maps and their inverses, parallel transport, observed solver
convergence orders, the bone-length invariant along the whole trajectory, the metric
definitions against their published anchors, the multimodal-GT construction, dataset
resampling, checkpoint round-trips, and the permutation-equivariance property.

## Layout

```
data/                      empty; preprocessed .npz files land here
logs/                      empty; SLURM job output
runs/                      released checkpoints, and training runs
scripts/
  train.py                 training entry point
  evaluate.py              full protocol evaluation, optionally sharded
  benchmark_inference.py   parameters, peak memory, wall clock per NFE
  run_ablation.py          the design-axis ladders
  configs/                 one YAML per reported configuration
  data/
    amass_preprocess.py    raw AMASS -> 22-joint positions at 60 fps
    dataset.py             windowed segments, both dataset layouts, resampling
    skeleton.py            kinematic tree, forward kinematics, manifold encoding
  models/
    geometry.py            exp, log, projection, parallel transport on the product
    flow_matching.py       source, conditional path, target velocity, loss, samplers
    gst_transformer.py     the velocity field
    embeddings.py          sinusoidal encodings, MLP, adaLN modulation
  utils/
    metrics.py             both conventions, multimodal GT, CMD, APDE, MAE
    dist.py                distributed setup and reduction
    logging_utils.py       run directory, metrics.jsonl, checkpoints
  slurm/                   SLURM batch scripts
  tests/                   test suite and the ZeroVelocity baseline
```

## Configuration surface

Every switch in this code was exercised by a reported run or an ablation ladder. Options
that stayed off in all 34 training runs were removed rather than shipped dark, so
`FlowMatchingConfig`, `GSTConfig`, `DataConfig` and `TrainConfig` expose 58 fields in
total instead of 85. The model always predicts under the root-centred protocol, always
uses the permutation-equivariant structural descriptors and never a per-joint embedding,
and always draws the generation time from a logit-normal; those are properties of the
method, not settings.

## Metrics

`evaluate.py` reports, per segment and averaged:

- **ADE / FDE** — best of the 50 samples, average and final displacement.
- **MAE** — inter-limb joint angle error in degrees, SkeletonDiffusion's definition.
- **MMADE / MMFDE** — against the multimodal ground truth: for each pseudo-future the
  best sample, then the mean over pseudo-futures.
- **APD** — average pairwise distance within the ensemble; higher is more diverse.
- **APDE** — `|APD(prediction) - APD(multimodal ground truth)|`; penalises being wider
  than the data as well as narrower, which raw APD does not.
- **CMD** — cumulative motion deviation, how far the generated speed distribution sits
  from the dataset's.
- **stretching / jitter** — bone-length violation, identically zero for this
  parameterisation.

Both metric conventions are printed side by side, along with the moving-joint variants
that average over `J - 1` joints instead of `J`.
