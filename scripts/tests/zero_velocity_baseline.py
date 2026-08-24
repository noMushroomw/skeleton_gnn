from __future__ import annotations

import argparse
import json

import torch

from ..data.dataset import DataConfig, MotionWindowDataset
from ..utils.metrics import (ade_fde, ade_fde_conventional, cmd,
                            dataset_mean_velocity, limb_metrics,
                            mean_angle_error, summarize)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="data/data_3d_amass.npz")
    parser.add_argument("--split", default="valid")
    parser.add_argument("--obs", type=int, default=30)
    parser.add_argument("--pred", type=int, default=120)
    parser.add_argument("--stride", type=int, default=30)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-segments", type=int, default=0)
    parser.add_argument("--source-fps", type=int, default=0,
                        help="native frame rate of --path when it differs from "
                             "--fps; H36M is 50 Hz")
    parser.add_argument("--subjects", nargs="*", default=None,
                        help="subject-keyed files only (H36M): the protocol test "
                             "split is S9 S11")
    parser.add_argument("--out", default=None,
                        help="write the numbers, including the per-sub-dataset "
                             "breakdown, to this JSON")
    args = parser.parse_args()

    dataset = MotionWindowDataset(DataConfig(
        path=args.path, split=args.split, obs_length=args.obs,
        pred_length=args.pred, stride=args.stride, fps=args.fps,
        augment_mirror=False, augment=False,
        max_segments=args.max_segments, source_fps=args.source_fps,
        subjects=tuple(args.subjects) if args.subjects else ()))
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                         shuffle=False, num_workers=0)
    parents = dataset.parents

    acc, seen = {}, 0

    per_dataset = {}
    horizon_err = torch.zeros(args.pred)
    for batch in loader:
        past_pos = batch["past_pos"]
        future_pos = batch["future_pos"]

        pred = past_pos[:, -1:].expand(-1, args.pred, -1, -1).unsqueeze(1)

        ade, fde = ade_fde(pred, future_pos)
        c_ade, c_fde = ade_fde_conventional(pred, future_pos)
        mae = mean_angle_error(pred, future_pos, parents)
        stretch, jit, s_rmse, j_rmse = limb_metrics(pred, batch["bone_lengths"], parents)
        metrics = summarize({"ade": ade, "fde": fde, "mae": mae,
                             "conv_ade": c_ade, "conv_fde": c_fde,
                             "stretching": stretch, "jitter": jit,
                             "stretching_rmse": s_rmse, "jitter_rmse": j_rmse})
        b = past_pos.shape[0]
        for k, v in metrics.items():
            acc[k] = acc.get(k, 0.0) + v * b
        horizon_err += (pred[:, 0] - future_pos).norm(dim=-1).mean(-1).sum(0) * 100
        seen += b

        for k in range(b):
            name = dataset.sequence_name(int(batch["index"][k])).split("/")[0]
            row = per_dataset.setdefault(name, {"n": 0, "ade": 0.0, "fde": 0.0,
                                                "conv_ade": 0.0, "conv_fde": 0.0})
            row["n"] += 1
            row["ade"] += float(ade[k])
            row["fde"] += float(fde[k])
            row["conv_ade"] += float(c_ade[k])
            row["conv_fde"] += float(c_fde[k])

    print(f"ZeroVelocity baseline on {args.split}: {seen} segments, "
          f"{args.obs}->{args.pred} frames @ {args.fps} fps")
    for k in ("ade", "fde", "mae", "stretching", "jitter"):
        unit = "deg" if k == "mae" else ("%" if k in ("stretching", "jitter") else "cm")
        print(f"  {k:<16}{acc[k]/seen:8.3f} {unit}")
    print(f"  {'apd':<16}{0.0:8.3f} cm   (deterministic by construction)")

    pred_zero = torch.zeros(1, 1, args.pred, dataset.num_joints, 3)
    for label, gen in (("root-centred (correct)",
                        lambda: ((a - a[:, :1]) for _, a in dataset.sequences)),
                       ("world-space (the bug)",
                        lambda: (a for _, a in dataset.sequences))):
        ref = torch.as_tensor(dataset_mean_velocity(gen()), dtype=torch.float32)
        print(f"  CMD  {label:<24}{float(cmd(pred_zero, ref)):8.3f}"
              f"   (published Zero-Velocity CMD: 39.262)")
    print("\n  conventional convention (as published tables report it):")
    print(f"  {'ADE':<16}{acc['conv_ade']/seen:8.3f}")
    print(f"  {'FDE':<16}{acc['conv_fde']/seen:8.3f}")

    horizon_err /= seen
    print("\nerror vs prediction time")
    for t in (0.25, 0.5, 1.0, 1.5, 2.0):
        i = min(int(t * args.fps) - 1, args.pred - 1)
        print(f"  {t:4.2f} s   {horizon_err[i]:7.2f} cm")

    print(f"\nper sub-dataset ({len(per_dataset)} in pool)")
    print(f"  {'dataset':<20}{'segments':>10}{'uADE':>9}{'uFDE':>9}"
          f"{'ADE':>9}{'FDE':>9}")
    for name in sorted(per_dataset):
        row = per_dataset[name]
        n = row["n"]
        print(f"  {name:<20}{n:>10}{row['ade']/n:>9.2f}{row['fde']/n:>9.2f}"
              f"{row['conv_ade']/n:>9.3f}{row['conv_fde']/n:>9.3f}")
    print("  uADE/uFDE are the unified convention (cm); ADE/FDE the published one.")

    if args.out:
        results = {k: acc[k] / seen for k in acc}
        results["num_segments"] = seen
        results["split"] = args.split
        results["apd"] = 0.0
        results["horizon_error_cm"] = horizon_err.tolist()
        results["per_dataset"] = {
            k: {m: (v[m] / v["n"] if m != "n" else v["n"]) for m in v}
            for k, v in per_dataset.items()}
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nmetrics -> {args.out}")

if __name__ == "__main__":
    main()
