from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

SUITES: dict[str, list[tuple[str, str, str]]] = {
    "spatial": [
        ("floor",      "small_nospatial_nofk", "no coupling AND no FK position feature"),
        ("none",       "small_nospatial", "no spatial attention (FK positions still leak)"),
        ("fixed",      "small_fixed",     "fixed normalised-adjacency mixing"),
        ("onehop",     "small_1hop",      "learned attention, 1-hop mask (the draft)"),
        ("full",       "small_full",      "learned attention, hop+relation bias"),
    ],

    "temporal": [
        ("floor",   "small_notemporal_novel", "no temporal attention AND no velocity feature"),
        ("none",    "small_notemporal",       "no temporal attention (velocity still leaks)"),
        ("onehop",  "small_temporal_1hop",    "attention masked to adjacent frames"),
        ("full",    "small_full",             "full attention, signed-offset bias"),
    ],

    "adaln": [
        ("per_block",   "small_full",               "one Linear(h, 9h) per block (current)"),
        ("single",      "small_adaln_single",       "one global map + per-block offset"),
        ("single_deep", "small_adaln_single_deep",  "shared map, budget spent on depth"),
    ],
}

FLOW_SUITES: dict[str, list[tuple[str, dict, str]]] = {

    "fk": [
        ("a000", dict(fk_metric_alpha=0.0),  "product metric (current objective)"),
        ("a025", dict(fk_metric_alpha=0.25), "25% forward-kinematics metric"),
        ("a050", dict(fk_metric_alpha=0.5),  "50/50 blend"),
        ("a075", dict(fk_metric_alpha=0.75), "75% forward-kinematics metric"),
        ("a100", dict(fk_metric_alpha=1.0),  "pure forward-kinematics metric"),
    ],

    "common": [
        ("off",         dict(common_scale=0.0), "white only (current source)"),
        ("sqrt_025",    dict(source_mode="common_plus_white", common_scale=0.25,
                             sigma_ramp_kind="sqrt"), "common mode 0.25, sqrt ramp"),
        ("sqrt_050",    dict(source_mode="common_plus_white", common_scale=0.5,
                             sigma_ramp_kind="sqrt"), "common mode 0.50, sqrt ramp"),
        ("sqrt_100",    dict(source_mode="common_plus_white", common_scale=1.0,
                             sigma_ramp_kind="sqrt"), "common mode 1.00, sqrt ramp"),
        ("linear_050",  dict(source_mode="common_plus_white", common_scale=0.5,
                             sigma_ramp_kind="linear"),
         "shape control: linear instead of sqrt"),
        ("flat_050",    dict(source_mode="common_plus_white", common_scale=0.5,
                             sigma_ramp_kind="constant"),
         "shape control: no ramp at all"),
    ],
}


def run(cmd, log_path):
    with open(log_path, "w", encoding="utf-8") as f:
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                              text=True).returncode


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", required=True,
                   choices=sorted(set(SUITES) | set(FLOW_SUITES)))
    p.add_argument("--solver", default="euler",
                   choices=["euler", "midpoint", "heun"],
                   help="must match across every suite that will be compared")
    p.add_argument("--num-steps", type=int, default=20,
                   help="integrator steps; midpoint and heun cost 2 NFE each")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--sigma", type=float, default=0.9,
                   help="source spread; 0.9 is the swept optimum")
    p.add_argument("--preset", default="small_full",
                   help="architecture for the flow suites")
    p.add_argument("--train-segments", type=int, default=40000)
    p.add_argument("--eval-segments", type=int, default=600)
    p.add_argument("--num-samples", type=int, default=50)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--eval-chunk-size", type=int, default=64,
                   help="cap the expanded batch during in-training validation; "
                        "a wide model plus a 5x ensemble otherwise overruns the "
                        "card and kills the CUDA context mid-sample")
    p.add_argument("--seed", type=int, default=0,
                   help="repeat a whole suite under a second seed; results go to "
                        "runs/ablate_<suite>_s<seed>/ so the first run survives")
    p.add_argument("--only", nargs="*", default=None,
                   help="rerun just these variant tags and merge the results into "
                        "the suite's existing summary.json (for a point that "
                        "crashed, without redoing the ones that finished)")
    args = p.parse_args()

    suffix = "" if args.seed == 0 else f"_s{args.seed}"
    out_dir = os.path.join("runs", f"ablate_{args.suite}{suffix}")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    if args.suite in FLOW_SUITES:
        points = [(tag, args.preset, desc, over)
                  for tag, over, desc in FLOW_SUITES[args.suite]]
    else:
        points = [(tag, preset, desc, {}) for tag, preset, desc in SUITES[args.suite]]
    if args.only:
        wanted = set(args.only)
        missing = wanted - {t for t, *_ in points}
        if missing:
            raise SystemExit(f"unknown variant(s) {sorted(missing)}; this suite has "
                             f"{[t for t, *_ in points]}")
        points = [pt for pt in points if pt[0] in wanted]

    data = json.dumps({
        "path": "data/data_3d_amass.npz", "obs_length": 30, "pred_length": 120,
        "stride": 30, "fps": 60, "augment_mirror": True,
        "max_segments": args.train_segments})

    results = []
    summary_path = os.path.join(out_dir, "summary.json")
    if args.only and os.path.exists(summary_path):
        with open(summary_path, encoding="utf-8") as f:
            results = [r for r in json.load(f) if r["tag"] not in set(args.only)]
        print(f"keeping {len(results)} existing result(s): "
              f"{[r['tag'] for r in results]}")

    for tag, preset, desc, flow_over in points:
        name = f"ab_{args.suite}_{tag}{suffix}"
        seed = args.seed
        flow = json.dumps({"sigma_dir": args.sigma, **flow_over})
        print(f"\n=== {args.suite}/{tag}: {desc} ===", flush=True)
        t0 = time.time()

        rc = run([args.python, "-u", "-m", "scripts.train",
                  "--config", "scripts/configs/amass_ablation.yaml",
                  "--run-name", name,
                  "--set", f"max_steps={args.steps}", f"batch_size={args.batch_size}",
                  "num_workers=0", f"model_preset={preset}", "model={}",
                  f"seed={seed}", f"eval_chunk_size={args.eval_chunk_size}",
                  f"data={data}", f"flow={flow}"],
                 f"logs/{name}_train.log")
        if rc != 0:
            print(f"  training FAILED (rc={rc}) -> logs/{name}_train.log"); continue

        rc = run([args.python, "-u", "-m", "scripts.evaluate",
                  "--checkpoint", f"runs/{name}/checkpoints/ckpt_last.pt",
                  "--split", "test", "--num-samples", str(args.num_samples),
                  "--num-steps", str(args.num_steps), "--solver", args.solver,
                  "--batch-size", "32", "--chunk-size", "200",
                  "--max-segments", str(args.eval_segments), "--dump-samples", "8"],
                 f"logs/{name}_eval.log")
        if rc != 0:
            print(f"  evaluation FAILED (rc={rc}) -> logs/{name}_eval.log"); continue

        with open(f"runs/{name}/eval_test.json", encoding="utf-8") as f:
            ev = json.load(f)
        params = count_parameters(f"runs/{name}/checkpoints/ckpt_last.pt")
        row = {"tag": tag, "desc": desc, "preset": preset, "params": params,
               "minutes": (time.time() - t0) / 60, "flow": flow_over,
               **{k: ev[k] for k in ("ade", "fde", "apd", "cmd", "conv_ade",
                                     "conv_fde", "conv_apd", "stretching",
                                     "jitter") if k in ev}}
        results.append(row)
        print(f"  ADE {row['ade']:.2f}  APD {row['apd']:.2f}  CMD {row['cmd']:.2f}"
              f"  {params/1e6:.2f}M  ({row['minutes']:.1f} min)", flush=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    if not results:
        print("no variant completed"); return
    print("\n" + "=" * 92)
    print(f"{'variant':<12}{'params':>9}{'ADE':>8}{'FDE':>8}{'APD':>8}{'CMD':>8}"
          f"{'conv ADE':>10}{'conv APD':>10}  description")
    for r in results:
        print(f"{r['tag']:<12}{r['params']/1e6:>8.2f}M{r['ade']:>8.2f}{r['fde']:>8.2f}"
              f"{r['apd']:>8.2f}{r['cmd']:>8.2f}{r.get('conv_ade', 0):>10.3f}"
              f"{r.get('conv_apd', 0):>10.3f}  {r['desc']}")
    print("=" * 92)
    print(f"\nwrote {os.path.join(out_dir, 'summary.json')}")


def count_parameters(ckpt_path):
    import torch
    from .data.skeleton import Skeleton
    from .models.gst_transformer import GST_PRESETS, GSTConfig, GSTVelocityField
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    kwargs = cfg.get("resolved_model") or {
        **GST_PRESETS[cfg["model_preset"]], **cfg["model"]}
    return GSTVelocityField(GSTConfig(**kwargs), Skeleton.amass()).num_parameters()

if __name__ == "__main__":
    main()
