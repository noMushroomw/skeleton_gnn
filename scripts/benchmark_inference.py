from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from .data.dataset import DataConfig, MotionWindowDataset
from .evaluate import load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--nfe", type=int, nargs="*", default=[10, 20, 50],
                        help="function evaluations, not steps: midpoint and heun "
                             "use two per step, so they run NFE/2 steps")
    parser.add_argument("--solvers", nargs="*", default=["euler", "midpoint"])
    parser.add_argument("--batch-size", type=int, default=1,
                        help="1 is the published convention: a single inference")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--data-source-fps", type=int, default=0)
    parser.add_argument("--data-subjects", nargs="*", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    data_kwargs = {**ckpt["config"]["data"], "split": args.split,
                   "augment_mirror": False, "augment": False}
    if args.data_path:
        data_kwargs["path"] = args.data_path
    if args.data_source_fps:
        data_kwargs["source_fps"] = args.data_source_fps
    if args.data_subjects:
        data_kwargs["subjects"] = tuple(args.data_subjects)
    dataset = MotionWindowDataset(DataConfig(**data_kwargs))
    module, _, train_step = load_model(args.checkpoint, dataset.skeleton, device)

    params = sum(p.numel() for p in module.parameters())

    field_params = sum(p.numel() for p in module.field.parameters())

    batch = torch.utils.data.default_collate(
        [dataset[i] for i in range(args.batch_size)])
    past = batch["past"].to(device)
    lengths = batch["bone_lengths"].to(device)
    horizon = batch["future"].shape[1]

    name = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
    print(f"checkpoint    {args.checkpoint} (train step {train_step})")
    print(f"device        {name}")
    print(f"skeleton      {dataset.skeleton.num_joints} joints, "
          f"horizon {horizon} frames")
    print(f"workload      batch {args.batch_size}, {args.num_samples} futures "
          f"per observation")
    print(f"parameters    {params / 1e6:.2f} M total, "
          f"{field_params / 1e6:.2f} M in the velocity field")

    rows = []
    print(f"\n{'solver':<10}{'NFE':>5}{'steps':>7}{'median ms':>12}"
          f"{'min ms':>10}{'peak MiB':>11}")
    print("-" * 55)
    for solver in args.solvers:
        for nfe in args.nfe:
            steps = nfe if solver == "euler" else max(nfe // 2, 1)
            def once():
                with torch.no_grad():
                    module.sample(past, lengths, horizon, num_steps=steps,
                                  num_samples=args.num_samples, solver=solver,
                                  chunk_size=args.chunk_size)
            for _ in range(args.warmup):
                once()
            if device == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            times = []
            for _ in range(args.repeats):
                t0 = time.perf_counter()
                once()
                if device == "cuda":
                    torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1000.0)
            peak = (torch.cuda.max_memory_allocated() / 2 ** 20
                    if device == "cuda" else float("nan"))
            med = statistics.median(times)
            print(f"{solver:<10}{nfe:>5}{steps:>7}{med:>12.1f}"
                  f"{min(times):>10.1f}{peak:>11.0f}")
            rows.append({"solver": solver, "nfe": nfe, "steps": steps,
                         "median_ms": med, "min_ms": min(times),
                         "peak_mib": peak, "times_ms": times})

    print("\nEquiFusion Tab. 10, RTX 6000, one H36M inference, for orientation only:")
    print("  DLow 8.1M / 111 ms   DivSamp 23.1M / 8 ms   BeLFusion 17.8M / 10341 ms")
    print("  HumanMAC 28.7M / 7438 ms   CoMusion 19M / 153 ms   SkelDiff 26.5M / 412 ms")
    print("  EquiFusion 7.9M / 192 ms")
    print("Different GPU: quote our rows against each other, not against theirs.")

    if args.out:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"checkpoint": args.checkpoint, "device": name,
                       "params": params, "field_params": field_params,
                       "num_joints": dataset.skeleton.num_joints,
                       "horizon": horizon, "batch_size": args.batch_size,
                       "num_samples": args.num_samples, "rows": rows}, fh, indent=2)
        print(f"\nmetrics -> {args.out}")

if __name__ == "__main__":
    main()
