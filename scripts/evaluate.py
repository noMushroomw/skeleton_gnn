from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch

from .data.dataset import DataConfig, MotionWindowDataset
from .data.skeleton import forward_kinematics
from .models.flow_matching import FlowMatchingConfig, GraphFlowMatching
from .models.gst_transformer import GSTConfig, GSTVelocityField
from .utils.dist import (all_reduce_mean, cleanup_distributed, setup_distributed)
from .utils.metrics import (
    ade_fde, ade_fde_conventional, apd, apd_conventional, apde,
    build_multimodal_gt, cmd_from_mean_speed, dataset_mean_velocity,
    limb_metrics, mae_joint_angle, mean_angle_error, multimodal_ade_fde,
    sd_limb_tables, summarize,
)


def load_model(checkpoint_path, skeleton, device, flow_overrides=None):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    data_cfg = DataConfig(**{**cfg["data"], "split": "test"})

    from .models.gst_transformer import GST_PRESETS
    if cfg.get("resolved_model"):

        model_kwargs = dict(cfg["resolved_model"])
    else:
        model_kwargs = {**GST_PRESETS[cfg["model_preset"]], **cfg["model"],
                        "frame_period": 1.0 / data_cfg.fps}
        print("  (checkpoint predates resolved_model; rebuilding from preset "
              f"{cfg['model_preset']!r} -- verify it has not changed)")

        weights = ckpt.get("ema") or ckpt["model"]
        hop_rows = weights.get("field.hop_bias.weight")
        if hop_rows is not None and hop_rows.shape[0] - 1 != model_kwargs.get(
                "max_hop_bucket", GSTConfig.max_hop_bucket):
            model_kwargs["max_hop_bucket"] = int(hop_rows.shape[0]) - 1
            print(f"  (recovered max_hop_bucket={model_kwargs['max_hop_bucket']} "
                  f"from the checkpoint's hop_bias)")
    field = GSTVelocityField(GSTConfig(**model_kwargs), skeleton)
    flow_cfg = FlowMatchingConfig(**{**cfg["flow"], **(flow_overrides or {})})
    module = GraphFlowMatching(field, flow_cfg, skeleton)

    state = ckpt.get("ema") or ckpt["model"]
    module.load_state_dict(state)
    return module.to(device).eval(), cfg, ckpt.get("step", 0)



@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--solver", default="euler",
                        choices=["euler", "midpoint", "heun"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--mm-threshold", type=float, default=0.4)
    parser.add_argument("--stride", type=int, default=0,
                        help="segment stride; 0 = non-overlapping observation windows")
    parser.add_argument("--max-segments", type=int, default=0)
    parser.add_argument("--dump-samples", type=int, default=32,
                        help="store this many SEGMENTS' predictions for the notebook")
    parser.add_argument("--dump-members", type=int, default=10,
                        help="ensemble members kept per dumped segment. ADE is a "
                             "minimum over the ensemble, so a dump that keeps 10 "
                             "of 50 cannot reproduce the reported number: on the "
                             "test split best-of-10 is about 12%% worse than "
                             "best-of-50. Pass --dump-members 50 when the dump is "
                             "going to be used for figures rather than debugging")
    parser.add_argument("--chunk-size", type=int, default=256,
                        help="cap the expanded (batch x samples) per ODE solve; "
                             "raise it on a large card, 0 disables chunking")
    parser.add_argument("--data-path", default=None,
                        help="evaluate on a different dataset than the one the "
                             "checkpoint was trained on. The field's parameters "
                             "do not depend on the joint count and every graph "
                             "buffer is non-persistent, so a checkpoint loads "
                             "onto another kinematic tree unchanged -- this is "
                             "the zero-shot kinematics setting")
    parser.add_argument("--data-source-fps", type=int, default=0,
                        help="native frame rate of --data-path when it differs "
                             "from the protocol's; H36M is 50 Hz. Frames are "
                             "resampled so the temporal attention bias, which is "
                             "a function of the offset in seconds, stays valid")
    parser.add_argument("--data-subjects", nargs="*", default=None,
                        help="subject-keyed files only (H36M): the protocol test "
                             "split is S9 S11")
    parser.add_argument("--sigma-dir", type=float, default=None,
                        help="override the source spread at SAMPLING time only. "
                             "The field was trained to transport the source it was "
                             "trained with, so this measures the pushforward of a "
                             "MISMATCHED source, not what training at this sigma "
                             "would give. Narrowing is interpolation and fairly "
                             "safe; widening is extrapolation. Never report a "
                             "value obtained this way as an optimal training sigma")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    ctx = setup_distributed()
    device = ctx.device if ctx.enabled else args.device
    torch.manual_seed(args.seed)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    data_kwargs = dict(ckpt["config"]["data"])
    data_kwargs.update(split=args.split, augment_mirror=False, augment=False)
    if args.stride:
        data_kwargs["stride"] = args.stride
    else:
        data_kwargs["stride"] = data_kwargs.get("obs_length", 30)
    if args.max_segments:
        data_kwargs["max_segments"] = args.max_segments
    zero_shot = args.data_path is not None and args.data_path != data_kwargs["path"]
    trained_on = data_kwargs["path"]
    if args.data_path:
        data_kwargs["path"] = args.data_path
    if args.data_source_fps:
        data_kwargs["source_fps"] = args.data_source_fps
    if args.data_subjects:
        data_kwargs["subjects"] = tuple(args.data_subjects)
    dataset = MotionWindowDataset(DataConfig(**data_kwargs))
    flow_over = {} if args.sigma_dir is None else {"sigma_dir": args.sigma_dir}
    module, cfg, train_step = load_model(args.checkpoint, dataset.skeleton, device,
                                         flow_overrides=flow_over)
    trained_sigma = float(ckpt["config"]["flow"].get("sigma_dir", float("nan")))
    parents = dataset.parents.to(device)
    try:
        sd_tables = sd_limb_tables(dataset.skeleton.num_joints)
    except ValueError as exc:
        sd_tables = None
        if ctx.is_main:
            print(f"  (SkeletonDiffusion MAE unavailable: {exc})")
    if ctx.is_main:
        print(f"checkpoint    {args.checkpoint} (train step {train_step})")
        print(f"split         {args.split}: {len(dataset)} segments, "
              f"stride {data_kwargs['stride']}")
        if zero_shot:
            print(f"ZERO-SHOT KINEMATICS")
            print(f"              trained on {trained_on}")
            print(f"              evaluating on {data_kwargs['path']}, "
                  f"{dataset.skeleton.num_joints} joints"
                  + (f", resampled {args.data_source_fps} -> {data_kwargs['fps']} Hz"
                     if args.data_source_fps else "")
                  + (f", subjects {list(args.data_subjects)}"
                     if args.data_subjects else ""))
        print(f"sampling      {args.num_samples} futures, {args.num_steps} "
              f"{args.solver} steps")
        if args.sigma_dir is not None:
            print(f"source        sigma_dir OVERRIDDEN to {args.sigma_dir} "
                  f"(trained with {trained_sigma})")
            print("              this is the pushforward of a mismatched source, "
                  "NOT a retrained sigma")
        if ctx.enabled:
            print(f"world size    {ctx.world_size} ranks, "
                  f"~{len(dataset) // ctx.world_size} segments each")

    full_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                              shuffle=False, num_workers=4)
    all_last_obs, all_future_pos = [], []
    for batch in full_loader:
        all_last_obs.append(batch["past_pos"][:, -1])
        all_future_pos.append(batch["future_pos"])
    last_obs = torch.cat(all_last_obs)
    future_pos_all = torch.cat(all_future_pos)

    shard = list(range(ctx.rank, len(dataset), ctx.world_size))
    mm_index_full = build_multimodal_gt(last_obs, future_pos_all, args.mm_threshold)
    mm_index = [mm_index_full[i] for i in shard]
    if ctx.is_main:
        print(f"multimodal GT threshold {args.mm_threshold}: mean "
              f"{np.mean([len(i) for i in mm_index_full]):.1f} sequences per segment")
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, shard), batch_size=args.batch_size,
        shuffle=False, num_workers=2)

    ref_velocity = dataset_mean_velocity(
        (arr - arr[:, :1]) for _, arr in dataset.sequences)
    ref_velocity = torch.as_tensor(ref_velocity, dtype=torch.float32)

    pool = sorted({dataset.sequence_name(i).split("/")[0] for i in range(len(dataset))})
    if ctx.is_main:
        print(f"test pool     {len(pool)} sub-datasets: {', '.join(pool)}")

    acc, seen = {}, 0
    per_dataset = {}

    cmd_speed_sum, cmd_speed_n = None, 0

    gt_speed_sum, gt_speed_n = None, 0

    PROFILE_SAMPLES, MM_PROFILE_CAP = 10, 20
    horizon_len = int(data_kwargs.get("pred_length", 120))
    apd_prof_sum, apd_prof_n = torch.zeros(horizon_len), 0
    mm_prof_sum, mm_prof_n = torch.zeros(horizon_len), 0
    dumped = {"pred": [], "gt": [], "past": [], "bone_lengths": [], "names": []}
    dump_batches = max(1, -(-args.dump_samples // max(args.batch_size, 1)))
    total_batches = max(1, -(-len(shard) // max(args.batch_size, 1)))
    dump_stride = max(1, total_batches // dump_batches)
    start = time.time()

    for bi, batch in enumerate(loader):
        past = batch["past"].to(device)
        lengths = batch["bone_lengths"].to(device)
        future_pos = batch["future_pos"].to(device)
        horizon = batch["future"].shape[1]

        samples = module.sample(past, lengths, horizon, num_steps=args.num_steps,
                                num_samples=args.num_samples, solver=args.solver,
                                chunk_size=args.chunk_size)
        b, n, t, j, _ = samples.shape
        pred_pos = forward_kinematics(samples.float(),
                                      lengths[:, None, None].expand(b, n, t, j),
                                      parents)

        ade, fde = ade_fde(pred_pos, future_pos)
        c_ade, c_fde = ade_fde_conventional(pred_pos, future_pos)
        mae = mean_angle_error(pred_pos, future_pos, parents)

        mae_sd = (mae_joint_angle(pred_pos, future_pos, *sd_tables)
                  if sd_tables else torch.full_like(mae, float("nan")))
        div = apd(pred_pos)
        c_apd = apd_conventional(pred_pos)
        stretch, jit, stretch_rmse, jit_rmse = limb_metrics(pred_pos, lengths, parents)

        offset = bi * args.batch_size
        mm_targets = [
            future_pos_all[mm_index[offset + k]] if len(mm_index[offset + k]) else None
            for k in range(b)
        ]
        mmade, mmfde = multimodal_ade_fde(pred_pos, mm_targets)
        c_mmade, c_mmfde = multimodal_ade_fde(pred_pos, mm_targets,
                                              conventional=True)
        apde_v = apde(pred_pos, mm_targets)

        metrics = summarize({"ade": ade, "fde": fde, "mae": mae,
                             "mae_sd": mae_sd,
                             "mmade": mmade, "mmfde": mmfde, "apd": div,
                             "apde": apde_v,
                             "conv_ade": c_ade, "conv_fde": c_fde,
                             "conv_apd": c_apd,
                             "conv_mmade": c_mmade, "conv_mmfde": c_mmfde,
                             "stretching": stretch, "jitter": jit,
                             "stretching_rmse": stretch_rmse,
                             "jitter_rmse": jit_rmse})

        for key, value in metrics.items():
            if value != value:
                continue
            total, count = acc.get(key, (0.0, 0))
            acc[key] = (total + value * b, count + b)
        seen += b

        for k in range(b):

            name = dataset.sequence_name(int(batch["index"][k])).split("/")[0]
            row = per_dataset.setdefault(name, {"n": 0, "ade": 0.0, "fde": 0.0,
                                                "apd": 0.0})
            row["n"] += 1
            row["ade"] += float(ade[k])
            row["fde"] += float(fde[k])
            row["apd"] += float(div[k])
        keep = pred_pos[:, : min(n, 10)]
        speed = (keep[:, :, 1:] - keep[:, :, :-1]).norm(dim=-1) * 100
        batch_sum = speed.sum(dim=(0, 1)).cpu()
        cmd_speed_sum = batch_sum if cmd_speed_sum is None else cmd_speed_sum + batch_sum
        cmd_speed_n += speed.shape[0] * speed.shape[1]

        gt_speed = (future_pos[:, 1:] - future_pos[:, :-1]).norm(dim=-1) * 100
        gt_batch_sum = gt_speed.sum(dim=0).cpu()
        gt_speed_sum = gt_batch_sum if gt_speed_sum is None else gt_speed_sum + gt_batch_sum
        gt_speed_n += gt_speed.shape[0]

        ens = pred_pos[:, : min(n, PROFILE_SAMPLES)]
        k = ens.shape[1]
        if k > 1:
            d = (ens[:, :, None] - ens[:, None, :]).norm(dim=-1)
            prof = d.sum(dim=(1, 2)) / (k * (k - 1))
            apd_prof_sum += prof.mean(-1).sum(0).cpu() * 100
            apd_prof_n += b
        for idx in range(b):
            tgt = mm_targets[idx]
            if tgt is None or tgt.shape[0] < 2:
                continue
            m = min(tgt.shape[0], MM_PROFILE_CAP)
            y = tgt[:m].to(device)
            dd = (y[:, None] - y[None, :]).norm(dim=-1)
            mm_prof_sum += (dd.sum(dim=(0, 1)) / (m * (m - 1))).mean(-1).cpu() * 100
            mm_prof_n += 1

        if ctx.is_main and bi % dump_stride == 0 and len(dumped["pred"]) < dump_batches:
            dumped["pred"].append(
                pred_pos[:, : min(n, args.dump_members)].cpu().numpy())
            dumped["gt"].append(future_pos.cpu().numpy())
            dumped["past"].append(batch["past_pos"].numpy())
            dumped["bone_lengths"].append(batch["bone_lengths"].numpy())
            dumped["names"].extend(
                dataset.sequence_name(int(i)) for i in batch["index"])

        if bi % 10 == 0 and ctx.is_main:
            done = seen / max(len(shard), 1)
            mean = lambda k: acc[k][0] / max(acc[k][1], 1)
            print(f"  rank0 {seen}/{len(shard)} segments  "
                  f"ade {mean('ade'):.2f}  fde {mean('fde'):.2f}  "
                  f"apd {mean('apd'):.2f}  ({done*100:.0f}%)")

    if ctx.enabled:
        import torch.distributed as dist
        keys = sorted(acc)
        buf = torch.tensor([[acc[k][0], acc[k][1]] for k in keys],
                           dtype=torch.float64, device=device)
        dist.all_reduce(buf, op=dist.ReduceOp.SUM)
        acc = {k: (float(buf[i, 0]), int(buf[i, 1])) for i, k in enumerate(keys)}

        speed = cmd_speed_sum.to(device)
        dist.all_reduce(speed, op=dist.ReduceOp.SUM)
        cmd_speed_sum = speed.cpu()
        n = torch.tensor(float(cmd_speed_n), device=device)
        dist.all_reduce(n, op=dist.ReduceOp.SUM)
        cmd_speed_n = int(n.item())

        gt_speed = gt_speed_sum.to(device)
        dist.all_reduce(gt_speed, op=dist.ReduceOp.SUM)
        gt_speed_sum = gt_speed.cpu()
        n = torch.tensor(float(gt_speed_n), device=device)
        dist.all_reduce(n, op=dist.ReduceOp.SUM)
        gt_speed_n = int(n.item())

        buf = torch.stack([apd_prof_sum, mm_prof_sum]).to(device)
        dist.all_reduce(buf, op=dist.ReduceOp.SUM)
        apd_prof_sum, mm_prof_sum = buf[0].cpu(), buf[1].cpu()
        cnt = torch.tensor([float(apd_prof_n), float(mm_prof_n)], device=device)
        dist.all_reduce(cnt, op=dist.ReduceOp.SUM)
        apd_prof_n, mm_prof_n = int(cnt[0].item()), int(cnt[1].item())

        gathered = [None] * ctx.world_size
        dist.all_gather_object(gathered, per_dataset)
        merged = {}
        for part in gathered:
            for name, row in part.items():
                tgt = merged.setdefault(name, {"n": 0, "ade": 0.0, "fde": 0.0,
                                               "apd": 0.0})
                for k in tgt:
                    tgt[k] += row[k]
        per_dataset = merged

        total_seen = torch.tensor(float(seen), device=device)
        dist.all_reduce(total_seen, op=dist.ReduceOp.SUM)
        seen = int(total_seen.item())

    results = {k: (total / count if count else float("nan"))
               for k, (total, count) in acc.items()}

    j_total = dataset.skeleton.num_joints
    moving = j_total / max(j_total - 1, 1)
    for key in ("ade", "fde", "mmade", "mmfde", "apd"):
        if key in results:
            results[f"{key}_moving"] = results[key] * moving
    results["moving_joint_factor"] = moving

    apd_cofactor = math.sqrt(j_total * horizon_len) / 100.0
    results["conv_apde"] = results["apde"] * apd_cofactor
    results["apd_cofactor"] = apd_cofactor
    results["mm_segments_with_gt"] = int(acc.get("mmade", (0.0, 0))[1])
    results["world_size"] = ctx.world_size
    results["cmd"] = float(cmd_from_mean_speed(
        cmd_speed_sum / max(cmd_speed_n, 1), ref_velocity))

    pred_prof = cmd_speed_sum / max(cmd_speed_n, 1)
    true_prof = gt_speed_sum / max(gt_speed_n, 1)
    pred_mean, true_mean = float(pred_prof[:, 1:].mean()), float(true_prof[:, 1:].mean())
    results["speed_profile"] = {
        "pred_per_frame": pred_prof[:, 1:].mean(-1).tolist(),
        "true_per_frame": true_prof[:, 1:].mean(-1).tolist(),
        "pred_per_joint": pred_prof.mean(0).tolist(),
        "true_per_joint": true_prof.mean(0).tolist(),
        "pred_mean": pred_mean,
        "true_mean": true_mean,
        "ratio": pred_mean / max(true_mean, 1e-8),

        "pred_field": pred_prof.tolist(),
        "true_field": true_prof.tolist(),
    }
    results["dispersion_profile"] = {
        "pred_apd_per_frame": (apd_prof_sum / max(apd_prof_n, 1)).tolist(),
        "mmgt_per_frame": (mm_prof_sum / max(mm_prof_n, 1)).tolist(),
        "pred_ensemble_size": PROFILE_SAMPLES,
        "mmgt_pool_cap": MM_PROFILE_CAP,
        "mmgt_segments": mm_prof_n,
    }
    results["num_segments"] = seen
    results["num_samples"] = args.num_samples
    results["num_steps"] = args.num_steps
    results["solver"] = args.solver

    results["sigma_dir"] = float(module.cfg.sigma_dir)
    results["sigma_dir_trained"] = trained_sigma
    results["sigma_dir_overridden"] = args.sigma_dir is not None
    results["seconds"] = time.time() - start
    results["checkpoint"] = os.path.abspath(args.checkpoint)
    results["train_step"] = train_step

    if not ctx.is_main:
        cleanup_distributed(ctx)
        return

    print("\n" + "=" * 84)
    print(f"{'ADE':>8}{'FDE':>9}{'MAE':>9}{'MMADE':>9}{'MMFDE':>9}"
          f"{'APD':>10}{'CMD':>10}{'str':>10}{'jit':>10}")
    print(f"{results['ade']:8.2f}{results['fde']:9.2f}{results['mae']:9.3f}"
          f"{results['mmade']:9.2f}{results['mmfde']:9.2f}{results['apd']:10.3f}"
          f"{results['cmd']:10.3f}{results['stretching']:10.4f}"
          f"{results['jitter']:10.4f}")
    print("=" * 84)
    print("units: cm, cm, deg, cm, cm, m, -, %, %   "
          f"(MMGT available for {results['mm_segments_with_gt']}/{seen} segments)")

    print(f"MAE {results['mae_sd']:.3f} deg in SkeletonDiffusion's definition "
          f"(inter-limb angles)   vs {results['mae']:.3f} in ours (bone orientation)")
    print(f"unified over the {j_total - 1} MOVING joints, EquiFusion's convention "
          f"(x {moving:.4f}):  uADE {results['ade_moving']:.3f}  "
          f"uFDE {results['fde_moving']:.3f}  uAPD {results['apd_moving']:.3f}")
    print(f"APDE {results['apde']:.3f} cm   "
          f"(|our APD - multimodal-GT APD|; lower is better, and unlike APD it "
          f"punishes over-dispersion)")

    if results["mmade"] < results["ade"]:
        print(f"  NOTE  MMADE {results['mmade']:.2f} < ADE {results['ade']:.2f}. "
              f"Every published row has MMADE above ADE, but that holds only when "
              f"the multimodal pool is built from the whole split; on a subset it "
              f"can invert. If this is a full-protocol run, check the reduction.")

    print("\nconventional convention (comparable to published tables)")
    print(f"  {'ADE':>8}{'FDE':>9}{'MMADE':>9}{'MMFDE':>9}{'APDE':>9}{'APD':>10}")
    print(f"  {results['conv_ade']:8.3f}{results['conv_fde']:9.3f}"
          f"{results['conv_mmade']:9.3f}{results['conv_mmfde']:9.3f}"
          f"{results['conv_apde']:9.3f}{results['conv_apd']:10.3f}")
    print("  Report your own ZeroVelocity row next to these. The test split "
          "differs from the")
    print("  published one (segmentation, AMASS version, GRAB), so the baseline "
          "is the only")
    print("  anchor that makes the comparison honest.")

    sp = results["speed_profile"]
    print(f"\nmean joint speed over the {dataset.skeleton.num_joints - 1} moving "
          f"joints (cm/frame)")
    print(f"  predicted        {sp['pred_mean']:8.3f}")
    print(f"  ground truth     {sp['true_mean']:8.3f}")
    print(f"  ratio            {sp['ratio']:8.3f}   "
          f"({(sp['ratio'] - 1) * 100:+.1f}% vs truth)")
    print("  per-frame profiles for both are in the JSON under 'speed_profile'.")

    dp = results["dispersion_profile"]
    ours = torch.tensor(dp["pred_apd_per_frame"])
    truth = torch.tensor(dp["mmgt_per_frame"])
    if mm_prof_n == 0:
        print("\ndispersion: no segment had two or more multimodal neighbours, so "
              "there is\n  no reference curve. Expected on a small --max-segments "
              "subset; on the\n  full split every segment has a pool.")
    else:
        print(f"\ndispersion against the horizon (cm; ours at {PROFILE_SAMPLES} "
              f"samples, multimodal GT at up to {MM_PROFILE_CAP} of "
              f"{dp['mmgt_segments']} pools)")
        print(f"  {'time (s)':>10}{'ours':>10}{'mm GT':>10}{'ratio':>10}")
        span = max(len(ours) // 8, 1)
        for lo in range(0, len(ours), span):
            o, t_ = ours[lo:lo + span].mean(), truth[lo:lo + span].mean()
            print(f"  {lo / 60:>10.2f}{o:>10.3f}{t_:>10.3f}"
                  f"{float(o / t_.clamp_min(1e-8)):>10.3f}")
        q = max(len(ours) // 4, 1)
        early = float(ours[:q].mean() / truth[:q].mean().clamp_min(1e-8))
        late = float(ours[-q:].mean() / truth[-q:].mean().clamp_min(1e-8))
        print(f"  early/late ratio of ratios {early / max(late, 1e-8):.3f}   "
              f"(1.0 = our spread grows exactly as the truth's does;\n"
              f"  above 1.0 = too wide early and too narrow late, which is what a "
              f"horizon\n  ramp on the source's common mode is meant to fix)")

    print(f"\nper sub-dataset ({len(per_dataset)} in pool)")
    print(f"  {'dataset':<20}{'segments':>10}{'ADE':>9}{'FDE':>9}{'APD':>9}")
    for name in sorted(per_dataset):
        row = per_dataset[name]
        n = row["n"]
        print(f"  {name:<20}{n:>10}{row['ade']/n:>9.2f}{row['fde']/n:>9.2f}"
              f"{row['apd']/n:>9.3f}")
    results["per_dataset"] = {k: {m: (v[m] / v["n"] if m != "n" else v["n"])
                                  for m in v} for k, v in per_dataset.items()}
    results["pool"] = pool

    expected = {"DFaust", "DanceDB", "GRAB", "HUMAN4D", "SOMA", "SSM", "Transitions"}
    if args.split == "test" and dataset.skeleton.num_joints == 22:
        missing = sorted(expected - set(pool))
        if missing:
            print(f"\n  WARNING: the AMASS test protocol expects 7 sub-datasets; "
                  f"missing {', '.join(missing)}.")
            print("  MMADE/MMFDE and CMD are computed over the pool, so all of the")
            print("  numbers above differ from the published tables. Report the pool.")
            results["protocol_complete"] = False
        else:
            results["protocol_complete"] = True

    run_dir = os.path.dirname(os.path.dirname(os.path.abspath(args.checkpoint)))
    out = args.out or os.path.join(run_dir, f"eval_{args.split}.json")

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nmetrics -> {out}")

    if dumped["pred"]:

        stem = os.path.splitext(os.path.basename(out))[0]
        sample_path = os.path.join(os.path.dirname(os.path.abspath(out)),
                                   "samples", f"{stem}.npz")
        os.makedirs(os.path.dirname(sample_path), exist_ok=True)
        np.savez_compressed(
            sample_path,
            pred=np.concatenate(dumped["pred"])[: args.dump_samples],
            gt=np.concatenate(dumped["gt"])[: args.dump_samples],
            past=np.concatenate(dumped["past"])[: args.dump_samples],
            bone_lengths=np.concatenate(dumped["bone_lengths"])[: args.dump_samples],
            names=np.asarray(dumped["names"][: args.dump_samples]),
            parents=dataset.parents.numpy(),
            joint_names=np.asarray(dataset.skeleton.joint_names),
            fps=np.asarray(data_kwargs.get("fps", 60)),
        )
        print(f"samples -> {sample_path}")

    cleanup_distributed(ctx)

if __name__ == "__main__":
    main()
