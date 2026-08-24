from __future__ import annotations

import math
import sys

import torch

from ..data.skeleton import Skeleton, forward_kinematics, positions_to_manifold
from ..models import geometry as geo
from ..models.flow_matching import FlowMatchingConfig, GraphFlowMatching
from ..models.gst_transformer import GSTConfig, GSTVelocityField

PASS, FAIL = "  ok  ", " FAIL "
_failures = []


def check(name, condition, detail=""):
    print(f"{PASS if condition else FAIL} {name}" + (f"   {detail}" if detail else ""))
    if not condition:
        _failures.append(name)


def random_sphere(*shape):
    x = torch.randn(*shape, 3, dtype=torch.float64)
    return x / x.norm(dim=-1, keepdim=True)


def test_sphere_maps():
    p = random_sphere(500)
    q = random_sphere(500)
    w = geo.project_tangent(p, torch.randn(500, 3, dtype=torch.float64))

    check("Exp stays on S^2",
          torch.allclose(geo.sphere_exp(p, w).norm(dim=-1),
                         torch.ones(500, dtype=torch.float64), atol=1e-10))
    check("Log is tangent",
          (p * geo.sphere_log(p, q)).sum(-1).abs().max() < 1e-9,
          f"max |<p,Log_p q>| = {(p * geo.sphere_log(p, q)).sum(-1).abs().max():.2e}")
    check("Exp o Log = id",
          (geo.sphere_exp(p, geo.sphere_log(p, q)) - q).abs().max() < 1e-8,
          f"max err = {(geo.sphere_exp(p, geo.sphere_log(p, q)) - q).abs().max():.2e}")
    check("Exp_p(0) = p",
          torch.allclose(geo.sphere_exp(p, torch.zeros_like(p)), p, atol=1e-12))
    check("||Log_p q|| = geodesic distance",
          (geo.sphere_log(p, q).norm(dim=-1)
           - torch.arccos((p * q).sum(-1).clamp(-1, 1))).abs().max() < 1e-6)


def test_geodesic_closed_form():
    z, y = random_sphere(400), random_sphere(400)
    s = torch.rand(400, 1, dtype=torch.float64)
    d, u = geo.geodesic_point_and_velocity(z, y, s)

    check("path on S^2",
          (d.norm(dim=-1) - 1).abs().max() < 1e-10)
    check("velocity tangent at d_s",
          (d * u).sum(-1).abs().max() < 1e-9,
          f"max |<d,u>| = {(d * u).sum(-1).abs().max():.2e}")
    check("||u|| = rho, constant in s",
          (u.norm(dim=-1) - torch.arccos((z * y).sum(-1).clamp(-1, 1))).abs().max() < 1e-7)

    d0, _ = geo.geodesic_point_and_velocity(z, y, torch.zeros_like(s))
    d1, _ = geo.geodesic_point_and_velocity(z, y, torch.ones_like(s))
    check("endpoints: X_0 = Z", (d0 - z).abs().max() < 1e-9)
    check("endpoints: X_1 = Y", (d1 - y).abs().max() < 1e-8)

    eps = 1e-6
    dp, _ = geo.geodesic_point_and_velocity(z, y, s + eps)
    dm, _ = geo.geodesic_point_and_velocity(z, y, s - eps)
    fd = (dp - dm) / (2 * eps)
    check("velocity matches finite difference",
          (fd - u).abs().max() < 1e-5, f"max err = {(fd - u).abs().max():.2e}")

    same = random_sphere(16)
    d_same, u_same = geo.geodesic_point_and_velocity(same, same.clone(),
                                                     torch.rand(16, 1, dtype=torch.float64))
    check("coincident: velocity is zero and finite",
          torch.isfinite(u_same).all() and u_same.abs().max() < 1e-12)
    anti = -same
    d_anti, u_anti = geo.geodesic_point_and_velocity(same, anti,
                                                     torch.rand(16, 1, dtype=torch.float64))
    check("antipodal: no NaN (event is masked out in the loss)",
          torch.isfinite(u_anti).all() and torch.isfinite(d_anti).all())
    check("antipodal detected by the mask",
          geo.is_cut_locus(same, anti).all())


def test_forward_kinematics_bijection():
    sk = Skeleton.amass()
    parents = torch.from_numpy(sk.parents_array())
    positions = torch.randn(7, 13, sk.num_joints, 3, dtype=torch.float64)
    state, lengths = positions_to_manifold(positions, parents)

    check("bone directions are unit",
          (state[..., 1:, :].norm(dim=-1) - 1).abs().max() < 1e-12)
    recon = forward_kinematics(state, lengths, parents)
    check("FK inverts the encoding",
          (recon - positions).abs().max() < 1e-10,
          f"max err = {(recon - positions).abs().max():.2e}")


def test_network_and_flow():
    torch.manual_seed(0)
    sk = Skeleton.amass()
    cfg = GSTConfig(hidden=64, num_layers=2, num_heads=4, time_embed_dim=32,
                    frame_period=1 / 60)
    field = GSTVelocityField(cfg, sk)
    module = GraphFlowMatching(field, FlowMatchingConfig(), sk)

    b, tp, tf, j = 3, 10, 20, sk.num_joints
    past_pos = torch.randn(b, tp, j, 3)
    future_pos = torch.randn(b, tf, j, 3)
    parents = torch.from_numpy(sk.parents_array())
    past, lengths_p = positions_to_manifold(past_pos, parents)
    future, _ = positions_to_manifold(future_pos, parents)
    lengths = lengths_p.mean(1)

    small_sk = Skeleton.from_parents([-1, 0, 1, 2, 0, 4], name="toy6")
    n22 = GSTVelocityField(cfg, sk).num_parameters()
    n6 = GSTVelocityField(cfg, small_sk).num_parameters()
    check("parameter count independent of J", n22 == n6,
          f"J=22 -> {n22}, J=6 -> {n6}")

    x_s, u_star = module.training_pair(module.sample_source(past, tf), future,
                                       torch.rand(b))
    v = field(x_s, past, torch.rand(b), lengths)
    check("field output is tangent",
          (v[..., 1:, :] * x_s[..., 1:, :]).sum(-1).abs().max() < 1e-5,
          f"max |<d,v>| = {(v[..., 1:, :] * x_s[..., 1:, :]).sum(-1).abs().max():.2e}")
    check("root velocity zeroed under the root-centred protocol",
          v[..., 0, :].abs().max() == 0)
    check("v_theta = 0 at initialisation (zero-init readout)",
          v.abs().max() == 0)

    loss, stats = module.loss(past, future, lengths)
    loss.backward()
    grads = [p.grad for p in field.parameters() if p.grad is not None]
    check("loss is finite and backward populates gradients",
          torch.isfinite(loss) and len(grads) > 0
          and all(torch.isfinite(g).all() for g in grads),
          f"loss = {loss.item():.4f}, {len(grads)} tensors with grad")

    samples = module.sample(past, lengths, tf, num_steps=5, num_samples=4)
    check("sampler shape", tuple(samples.shape) == (b, 4, tf, j, 3),
          str(tuple(samples.shape)))
    check("samples stay on the manifold",
          (samples[..., 1:, :].norm(dim=-1) - 1).abs().max() < 1e-5)

    exp_lengths = lengths[:, None, None].expand(b, 4, tf, j)
    pred_pos = forward_kinematics(samples, exp_lengths, parents)
    parent_pos = pred_pos[..., parents.clamp(min=0), :]
    decoded = (pred_pos - parent_pos).norm(dim=-1)[..., 1:]
    err = (decoded - lengths[:, None, None, 1:]).abs().max()
    check("bone lengths preserved exactly (no penalty needed)", err < 1e-5,
          f"max deviation = {err:.2e} m")

    s1 = module.sample(past, lengths, tf, num_steps=3, num_samples=2)
    spread = (s1[:, 0] - s1[:, 1]).abs().max()
    check("distinct source draws give distinct futures", spread > 0,
          f"max separation = {spread:.3f}")


def test_source_distribution():
    sk = Skeleton.amass()
    parents = torch.from_numpy(sk.parents_array())
    past, _ = positions_to_manifold(torch.randn(64, 10, sk.num_joints, 3), parents)

    for coupling in ("iid",):
        module = GraphFlowMatching(
            torch.nn.Identity(), FlowMatchingConfig(sigma_dir=0.35), sk)
        z = module.sample_source(past, 15)
        check(f"q_0 ({coupling}) lies on the manifold",
              (z[..., 1:, :].norm(dim=-1) - 1).abs().max() < 1e-5)
        center = module.zero_velocity_extrapolation(past, 15)
        angle = torch.arccos((z[..., 1:, :] * center[..., 1:, :])
                             .sum(-1).clamp(-1, 1))
        check(f"q_0 ({coupling}) is centred on the last observed pose",
              angle.mean() < 1.0, f"mean angular spread = {angle.mean():.3f} rad")


def test_weight_tying():
    sk = Skeleton.amass()
    kw = dict(hidden=128, num_layers=6, num_heads=4, time_embed_dim=64,
              frame_period=1 / 60)
    untied = GSTVelocityField(GSTConfig(**kw), sk)
    tied = GSTVelocityField(GSTConfig(**kw, tie_layers=True), sk)
    check("tying shrinks the trunk by roughly num_layers",
          tied.num_parameters() < untied.num_parameters() / 3,
          f"{untied.num_parameters()/1e6:.2f}M -> {tied.num_parameters()/1e6:.2f}M")
    check("tied trunk holds exactly one block",
          tied.blocks is None and tied.block is not None)
    check("iteration embedding has one row per application",
          tuple(tied.iter_embed.shape) == (6, 128))

    parents = torch.from_numpy(sk.parents_array())
    past, _ = positions_to_manifold(torch.randn(2, 10, 22, 3), parents)
    fut, _ = positions_to_manifold(torch.randn(2, 20, 22, 3), parents)
    lengths = torch.rand(2, 22) + 0.2
    module = GraphFlowMatching(tied, FlowMatchingConfig(), sk)
    opt = torch.optim.AdamW(module.parameters(), lr=1e-2)
    first = None
    for _ in range(5):
        loss, _ = module.loss(past, fut, lengths)
        first = first if first is not None else loss.item()
        opt.zero_grad(); loss.backward(); opt.step()
    loss, _ = module.loss(past, fut, lengths); opt.zero_grad(); loss.backward()
    check("shared block receives gradient once the gates open",
          tied.block.attn_sp.qkv.weight.grad.abs().sum() > 0)
    check("every iteration embedding row is trained",
          (tied.iter_embed.grad.abs().sum(-1) > 0).all())



def test_spatial_modes():
    sk = Skeleton.amass()
    parents = torch.from_numpy(sk.parents_array())
    past, _ = positions_to_manifold(torch.randn(2, 10, 22, 3), parents)
    fut, _ = positions_to_manifold(torch.randn(2, 20, 22, 3), parents)
    lengths = torch.rand(2, 22) + 0.2
    for mode in ("attention", "fixed", "none"):
        f = GSTVelocityField(GSTConfig(hidden=64, num_layers=2, num_heads=4,
                                       time_embed_dim=32, spatial_mode=mode,
                                       frame_period=1/60), sk)
        v = f(fut, past, torch.rand(2), lengths)
        tangent = (v[..., 1:, :] * fut[..., 1:, :]).sum(-1).abs().max()
        check(f"spatial_mode={mode} produces a tangent field",
              tangent < 1e-5 and v.shape == fut.shape)


def test_new_configurations():
    torch.manual_seed(0)
    sk = Skeleton.amass()
    parents = torch.from_numpy(sk.parents_array())
    b, tp, tf, j = 2, 6, 12, sk.num_joints
    past, lengths_p = positions_to_manifold(torch.randn(b, tp, j, 3), parents)
    future, _ = positions_to_manifold(torch.randn(b, tf, j, 3), parents)
    lengths = lengths_p.mean(1)
    base = dict(hidden=64, num_layers=2, num_heads=4, time_embed_dim=32,
                frame_period=1 / 60)

    diameter = int(sk.hop_distance().max())
    check("W6: the SMPL tree has diameter 11", diameter == 11, f"diameter {diameter}")
    check("W6: hop buckets reach the diameter",
          GSTConfig(**base).max_hop_bucket >= diameter and
          GSTVelocityField(GSTConfig(**base), sk).hop_bias.num_embeddings
          == GSTConfig(**base).max_hop_bucket + 1)

    variants = {
        "adaln single":        dict(adaln_mode="single"),
        "temporal none":       dict(temporal_mode="none"),
        "temporal 1-hop":      dict(temporal_window=1),
        "temporal floor":      dict(temporal_mode="none", use_velocity_feature=False),
        "tied + adaln single": dict(tie_layers=True, adaln_mode="single"),
        "spatial none":        dict(spatial_mode="none"),
        "spatial fixed":       dict(spatial_mode="fixed"),
    }
    for name, over in variants.items():
        net = GSTVelocityField(GSTConfig(**{**base, **over}), sk)

        torch.nn.init.normal_(net.phi_out.weight, std=0.05)
        module = GraphFlowMatching(net, FlowMatchingConfig(), sk)
        x_s, _ = module.training_pair(module.sample_source(past, tf), future,
                                      torch.rand(b))
        v = net(x_s, past, torch.rand(b), lengths)
        tangency = (v[..., 1:, :] * x_s[..., 1:, :]).sum(-1).abs().max()
        s = module.sample(past, lengths, tf, num_steps=3, num_samples=2,
                          solver="heun")
        on_sphere = (s[..., 1:, :].norm(dim=-1) - 1).abs().max()
        check(f"{name}: forward, tangency, exact bone lengths",
              torch.isfinite(v).all() and tangency < 1e-5
              and v[..., 0, :].abs().max() == 0 and on_sphere < 1e-5,
              f"|<d,v>| {tangency:.1e}, sphere dev {on_sphere:.1e}")

    check("temporal_mode is validated, not silently ignored",
          _raises(lambda: GSTVelocityField(
              GSTConfig(**{**base, "temporal_mode": "sum"}), sk)))
    check("adaln_mode is validated",
          _raises(lambda: GSTVelocityField(
              GSTConfig(**{**base, "adaln_mode": "shared"}), sk)))
    check("spatial_mode is validated",
          _raises(lambda: GSTVelocityField(
              GSTConfig(**{**base, "spatial_mode": "sum"}), sk)))

    full = dict(hidden=384, num_layers=8, num_heads=8, time_embed_dim=128)
    n_per = GSTVelocityField(GSTConfig(**full), sk).num_parameters()
    n_one = GSTVelocityField(
        GSTConfig(**{**full, "adaln_mode": "single"}), sk).num_parameters()
    h, L = 384, 8
    expected = L * (9 * h * h + 9 * h) - (9 * h * h + 9 * h) - L * 9 * h
    check("W5: adaln single frees exactly the predicted parameters",
          n_per - n_one == expected,
          f"{n_per:,} -> {n_one:,}, freed {n_per-n_one:,} (predicted {expected:,}, "
          f"{(n_per-n_one)/n_per*100:.1f}% of the model)")
    n_deep = GSTVelocityField(GSTConfig(
        **{**full, "adaln_mode": "single", "num_layers": 12}), sk).num_parameters()
    check("W5: the 12-block variant fits base's budget",
          abs(n_deep - n_per) / n_per < 0.02,
          f"12 blocks shared-adaLN {n_deep:,} against base {n_per:,}")

    n_tied = GSTVelocityField(GSTConfig(
        **{**full, "adaln_mode": "single", "num_layers": 12,
           "tie_layers": True}), sk).num_parameters()
    check("tied headline is the same shape at a fraction of the parameters",
          n_tied < n_deep / 5 and 4.0e6 < n_tied < 5.0e6,
          f"{n_tied:,} against the headline's {n_deep:,} "
          f"({n_deep / n_tied:.1f}x fewer; EquiFusion is 7.9M)")

    torch.manual_seed(1)
    resid = torch.randn(5, j, 3)
    resid[:, 0] = 0.0
    bl = torch.rand(5, j) + 0.5
    got = forward_kinematics(resid, bl, parents)
    tree = sk.parents
    want = torch.zeros_like(got)
    for node in range(1, j):
        walk, cur = [], node
        while cur > 0:
            walk.append(cur)
            cur = tree[cur]
        want[:, node] = sum(resid[:, x] * bl[:, x, None] for x in walk)
    check("W2: FK of a residual equals sum_{b in P(j)} l_b e_b",
          (got[:, 1:] - want[:, 1:]).abs().max() < 1e-5,
          f"max dev {(got[:, 1:] - want[:, 1:]).abs().max():.2e}")

    net = GSTVelocityField(GSTConfig(**base), sk)
    torch.nn.init.normal_(net.phi_out.weight, std=0.05)

    def loss_at(**over):

        torch.manual_seed(11)
        module = GraphFlowMatching(
            net, FlowMatchingConfig(**over), sk)
        return module.loss(past, future, lengths,
                           generator=torch.Generator().manual_seed(7))

    losses = {a: loss_at(fk_metric_alpha=a) for a in (0.0, 0.5, 1.0)}
    check("W2: alpha=0 leaves the objective untouched",
          float(losses[0.0][0]) == float(loss_at()[0]))
    values = {a: float(v[0]) for a, v in losses.items()}
    check("W2: the metric changes the loss, monotonically in alpha, and is finite",
          values[0.0] != values[1.0]
          and all(torch.isfinite(torch.tensor(v)) for v in values.values()),
          f"alpha 0 -> {values[0.0]:.4f}, 0.5 -> {values[0.5]:.4f}, "
          f"1.0 -> {values[1.0]:.4f}")
    check("W2: loss_fk is reported separately so the balance stays visible",
          float(losses[0.0][1]["loss_fk"]) == 0.0
          and float(losses[1.0][1]["loss_fk"]) > 0.0)
    losses[1.0][0].backward()
    check("W2: gradients survive the forward-kinematics recursion",
          any(p.grad is not None and torch.isfinite(p.grad).all()
              for p in net.parameters()))

    sk_t = Skeleton.from_parents([-1, 0, 1, 2], name="chain4")
    p_t = torch.from_numpy(sk_t.parents_array())
    past_t, _ = positions_to_manifold(torch.randn(4096, 3, 4, 3), p_t)
    sigma, cs, horizon = 0.5, 0.8, 6

    def tangents(**over):
        cfg = FlowMatchingConfig(sigma_dir=sigma, **over)
        mod = GraphFlowMatching(GSTVelocityField(GSTConfig(**base), sk_t), cfg, sk_t)
        z = mod.sample_source(past_t, horizon,
                              generator=torch.Generator().manual_seed(3))
        centre = mod.zero_velocity_extrapolation(past_t, horizon)
        return geo.sphere_log(centre[..., 1:, :], z[..., 1:, :])

    white = tangents()
    check("W3: common_scale=0 reproduces the white source bit-for-bit",
          torch.equal(white, tangents(source_mode="common_plus_white",
                                      common_scale=0.0)))
    tan = tangents(source_mode="common_plus_white", common_scale=cs,
                   sigma_ramp_kind="linear")
    var = tan.pow(2).sum(-1).mean(dim=(0, 2))
    ramp = torch.arange(horizon, dtype=torch.float64) / (horizon - 1)
    want_var = 2 * sigma ** 2 * (1 + cs ** 2 * ramp ** 2)
    check("W3: marginal variance is 2 sigma^2 (1 + c^2 r(t)^2)",
          (var.double() / want_var - 1).abs().max() < 0.06,
          f"max relative deviation {float((var.double()/want_var - 1).abs().max()):.3f}")
    cov = (tan[:, -1] * tan[:, -2]).sum(-1).mean()
    want_cov = 2 * sigma ** 2 * cs ** 2 * float(ramp[-1] * ramp[-2])
    check("W3: cross-frame covariance is 2 sigma^2 c^2 r(t) r(q)",
          abs(float(cov) / want_cov - 1) < 0.08,
          f"{float(cov):.4f} against {want_cov:.4f}")
    check("W3: the white component keeps its full amplitude at t=0",
          abs(float(var[0]) / (2 * sigma ** 2) - 1) < 0.06,
          f"var(0) = {float(var[0]):.4f}, white would be {2*sigma**2:.4f}")
    check("W3: source_mode and sigma_ramp_kind are validated",
          _raises(lambda: tangents(source_mode="pink"))
          and _raises(lambda: tangents(source_mode="common_plus_white",
                                       common_scale=0.5, sigma_ramp_kind="cubic")))

    cfg = GSTConfig(**base)
    donor = GSTVelocityField(cfg, sk)
    recipient = GSTVelocityField(cfg, Skeleton.h36m17())
    missing = recipient.load_state_dict(donor.state_dict(), strict=True)
    h36 = Skeleton.h36m17()
    p36 = torch.from_numpy(h36.parents_array())
    past36, len36 = positions_to_manifold(
        torch.randn(2, tp, h36.num_joints, 3), p36)
    out = recipient(*[past36[:, :tf] if tf <= tp else past36.repeat(1, 3, 1, 1)[:, :tf],
                      past36, torch.rand(2), len36.mean(1)])
    check("W8: an AMASS-trained field loads and runs on the H36M skeleton",
          not missing.missing_keys and not missing.unexpected_keys
          and torch.isfinite(out).all() and out.shape[-2] == h36.num_joints,
          f"{h36.num_joints} joints, output {tuple(out.shape)}")


def test_cross_dataset_support():
    from ..data.dataset import DataConfig, MotionWindowDataset
    from ..utils.metrics import apd, apde

    torch.manual_seed(0)

    pool = torch.randn(6, 20, 5, 3)
    pred = pool.clone().unsqueeze(0)
    check("APDE is zero when the ensemble matches the pool",
          float(apde(pred, [pool])[0]) < 1e-4,
          f"{float(apde(pred, [pool])[0]):.2e}")
    centre = pool.mean(0, keepdim=True)
    wide = ((pool - centre) * 1.5 + centre).unsqueeze(0)
    narrow = ((pool - centre) * 0.5 + centre).unsqueeze(0)
    base = float(apd(pool.unsqueeze(0))[0])
    check("APDE punishes over- and under-dispersion alike",
          abs(float(apde(wide, [pool])[0]) - 0.5 * base) < 1e-3
          and abs(float(apde(narrow, [pool])[0]) - 0.5 * base) < 1e-3,
          f"wide {float(apde(wide,[pool])[0]):.3f}, "
          f"narrow {float(apde(narrow,[pool])[0]):.3f}, half-APD {0.5*base:.3f}")
    check("APDE is NaN, not zero, without a usable pool",
          torch.isnan(apde(pred, [None])).all()
          and torch.isnan(apde(pred, [pool[:1]])).all())

    from ..utils.metrics import ade_fde, multimodal_ade_fde
    ens = torch.randn(1, 4, 20, 5, 3)
    members = torch.randn(6, 20, 5, 3)
    mmade, mmfde = multimodal_ade_fde(ens, [members])
    d = (ens[0][:, None] - members[None]).norm(dim=-1).mean(dim=(-1, -2))
    check("MMADE takes min over samples, then mean over multimodal members",
          abs(float(mmade[0]) - float(d.min(dim=0).values.mean()) * 100.0) < 1e-3
          and float(mmade[0]) > float(d.min()) * 100.0,
          f"{float(mmade[0]):.4f} vs global-min {float(d.min()) * 100.0:.4f}")

    single, _ = ade_fde(ens, members[:1])
    mm_one, _ = multimodal_ade_fde(ens, [members[:1]])
    check("MMADE against a one-member pool equals plain ADE",
          abs(float(mm_one[0]) - float(single[0])) < 1e-4,
          f"{float(mm_one[0]):.4f} vs {float(single[0]):.4f}")
    c_mmade, c_mmfde = multimodal_ade_fde(ens, [members], conventional=True)
    dc = (ens[0][:, None] - members[None]).flatten(start_dim=-2).norm(dim=-1)
    check("conventional MMADE uses the flattened-pose norm, no cm scaling",
          abs(float(c_mmade[0])
              - float(dc.mean(dim=-1).min(dim=0).values.mean())) < 1e-4
          and abs(float(c_mmfde[0])
                  - float(dc[..., -1].min(dim=0).values.mean())) < 1e-4,
          f"{float(c_mmade[0]):.4f} / {float(c_mmfde[0]):.4f}")

    frames = 101
    ramp = torch.linspace(0, 1, frames)[:, None, None] * torch.ones(1, 4, 3)
    cfg = DataConfig(fps=60, source_fps=50)
    out = MotionWindowDataset._resample(ramp.numpy(), cfg)
    want = int(round(frames * 60 / 50))
    lin = torch.linspace(0, 1, want)[:, None, None] * torch.ones(1, 4, 3)
    check("50 -> 60 Hz resampling has the right length",
          out.shape[0] == want, f"{frames} -> {out.shape[0]}, expected {want}")
    check("resampling is exact on a constant-velocity trajectory",
          float(torch.from_numpy(out).sub(lin).abs().max()) < 1e-5)
    check("resampling is a no-op when the rates match",
          MotionWindowDataset._resample(ramp.numpy(),
                                        DataConfig(fps=60, source_fps=60)).shape[0]
          == frames)

    check("moving-joint factor reproduces the published H36M anchor",
          abs(11.08 * 17 / 16 - 11.77) < 0.01,
          f"11.08 x 17/16 = {11.08 * 17 / 16:.3f} against their 11.77")

    from ..utils.metrics import mae_joint_angle, mean_angle_error, sd_limb_tables
    for j, n_limbs, n_pairs in ((22, 21, 19), (17, 16, 12)):
        limbs, chains = sd_limb_tables(j)
        pairs = sum(len(c) - 1 for c in chains)
        check(f"SkelDiff MAE tables for J={j} have the right shape",
              len(limbs) == n_limbs and pairs == n_pairs
              and max(max(l) for l in limbs) == j - 1
              and max(max(c) for c in chains) < len(limbs),
              f"{len(limbs)} limbs, {pairs} angle pairs")
    check("SkelDiff MAE tables are missing for an unknown skeleton",
          _raises(lambda: sd_limb_tables(31)))

    torch.manual_seed(3)
    gt = random_sphere(64, 20, 22) * 0.3
    check("SkelDiff MAE is zero on an exact prediction",
          float(mae_joint_angle(gt.unsqueeze(1), gt).max()) < 1e-3)

    th = torch.tensor(0.7, dtype=gt.dtype)
    rot = torch.tensor([[torch.cos(th), -torch.sin(th), 0.0],
                        [torch.sin(th), torch.cos(th), 0.0],
                        [0.0, 0.0, 1.0]], dtype=gt.dtype)
    turned = gt @ rot.T
    check("SkelDiff MAE is invariant to a rigid rotation, unlike ours",
          float(mae_joint_angle(turned.unsqueeze(1), gt).max()) < 1e-3
          and float(mean_angle_error(turned.unsqueeze(1), gt,
                                     torch.from_numpy(
                                         Skeleton.amass().parents_array())).mean()) > 5.0)


def _raises(fn):
    try:
        fn()
    except (ValueError, KeyError, RuntimeError):
        return True
    return False


class _RotationField(torch.nn.Module):

    def __init__(self, omega, a):
        super().__init__()
        self.register_buffer("omega", omega)
        self.a = a

    def forward(self, x, past, s, bone_lengths):
        v = self.a * torch.linalg.cross(self.omega.expand_as(x), x, dim=-1)
        return torch.cat([torch.zeros_like(v[..., :1, :]), v[..., 1:, :]], dim=-2)


def _rodrigues(x, omega, ang):
    c, s = math.cos(ang), math.sin(ang)
    return (x * c + torch.linalg.cross(omega.expand_as(x), x, dim=-1) * s
            + omega * (omega * x).sum(-1, keepdim=True) * (1.0 - c))


def test_parallel_transport():
    p = random_sphere(2000)
    q = random_sphere(2000)
    w = geo.project_tangent(p, torch.randn(2000, 3, dtype=torch.float64))
    tw = geo.sphere_transport(p, q, w)

    check("transport preserves the norm",
          (tw.norm(dim=-1) - w.norm(dim=-1)).abs().max() < 1e-10,
          f"max dev {(tw.norm(dim=-1) - w.norm(dim=-1)).abs().max():.2e}")
    check("transport lands in T_q S^2",
          (tw * q).sum(-1).abs().max() < 1e-9,
          f"max <T(w), q> {(tw * q).sum(-1).abs().max():.2e}")
    check("transporting there and back is the identity",
          (geo.sphere_transport(q, p, tw) - w).abs().max() < 1e-9)
    check("transport to the same point is the identity",
          (geo.sphere_transport(p, p, w) - w).abs().max() < 1e-12)

    v = geo.sphere_log(p, q)
    rho = v.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    vhat = v / rho
    ref = w + (vhat * w).sum(-1, keepdim=True) * (
        (torch.cos(rho) - 1.0) * vhat - torch.sin(rho) * p)
    check("closed form agrees with the trigonometric form",
          (tw - ref).abs().max() < 1e-8,
          f"max dev {(tw - ref).abs().max():.2e}")

    anti = geo.sphere_transport(p, -p, w)
    check("antipodal transport is finite and tangent",
          torch.isfinite(anti).all() and (anti * p).sum(-1).abs().max() < 1e-9)

    state_p = torch.cat([torch.randn(64, 1, 3, dtype=torch.float64),
                         random_sphere(64, 4)], dim=-2)
    state_q = torch.cat([torch.randn(64, 1, 3, dtype=torch.float64),
                         random_sphere(64, 4)], dim=-2)
    tan = geo.product_project_tangent(
        state_p, torch.randn(64, 5, 3, dtype=torch.float64))
    out = geo.product_transport(state_p, state_q, tan)
    check("product transport is the identity on the root row",
          (out[:, 0] - tan[:, 0]).abs().max() < 1e-12)
    check("product transport lands in the tangent space of the target",
          (out[:, 1:] * state_q[:, 1:]).sum(-1).abs().max() < 1e-9)


def test_solver_order():
    torch.manual_seed(0)
    j, tf, n, ang = 5, 3, 4, 1.0
    omega = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    module = GraphFlowMatching(_RotationField(omega, ang), FlowMatchingConfig(),
                               Skeleton.from_parents([-1, 0, 1, 2, 3]))

    x0 = torch.cat([torch.randn(n, tf, 1, 3, dtype=torch.float64),
                    random_sphere(n, tf, j - 1)], dim=-2)
    exact = torch.cat([x0[..., :1, :],
                       _rodrigues(x0[..., 1:, :], omega, ang)], dim=-2)

    def global_error(solver, steps):
        out, _ = module._integrate(x0, None, None, steps, solver, False)
        return float((out[..., 1:, :] - exact[..., 1:, :]).norm(dim=-1).max())

    for solver, floor in (("euler", 0.8), ("midpoint", 1.8), ("heun", 1.8)):
        e_coarse, e_fine = global_error(solver, 8), global_error(solver, 16)
        order = math.log2(e_coarse / e_fine)
        check(f"{solver}: observed convergence order",
              order > floor,
              f"error {e_coarse:.3e} -> {e_fine:.3e}, order {order:.2f} "
              f"(need > {floor})")

    for solver in ("euler", "midpoint", "heun"):
        out, _ = module._integrate(x0, None, None, 8, solver, False)
        check(f"{solver}: bones stay exactly on S^2",
              (out[..., 1:, :].norm(dim=-1) - 1.0).abs().max() < 1e-12)


if __name__ == "__main__":
    print("--- sphere maps ---");            test_sphere_maps()
    print("--- geodesic closed form ---");   test_geodesic_closed_form()
    print("--- forward kinematics ---");     test_forward_kinematics_bijection()
    print("--- source distribution ---");    test_source_distribution()
    print("--- network and flow ---");       test_network_and_flow()
    print("--- weight tying ---");           test_weight_tying()
    print("--- spatial modes ---");          test_spatial_modes()
    print("--- new configurations ---");     test_new_configurations()
    print("--- cross-dataset support ---");  test_cross_dataset_support()
    print("--- parallel transport ---");     test_parallel_transport()
    print("--- solver order ---");           test_solver_order()
    print()
    if _failures:
        print(f"{len(_failures)} FAILED: {', '.join(_failures)}")
        sys.exit(1)
    print("all checks passed")
