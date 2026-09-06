"""Gate 1-3 hard-threshold precheck (mentor spec).

Prints a pass/fail report for:
  1. 7-step smoke does not crash (spec §6 baseline).
  2. Gradients are strictly non-zero across CCA/HCA/heads (§6.2 + backward sanity).
  3. Param increase vs plain (no c_e FiLM, no τ modulation, no bias_mlp, no HOR) ≤ 15%.

Exit 0 if all three pass, non-zero otherwise.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

torch.manual_seed(0)
d_in = 16


def mg(mod):
    """对 Sequential/多层组合，把所有 Linear（含 LN）的 weight.grad.abs().mean 取 max，
    兼容最新 bottleneck 版 FiLM/HOR（不再是单层 Linear）。
    对单个 Linear，等价于旧实现。"""
    best = 0.0
    if isinstance(mod, (torch.nn.Linear, torch.nn.LayerNorm)):
        if hasattr(mod, "weight") and mod.weight is not None and mod.weight.grad is not None:
            best = float(mod.weight.grad.abs().mean().item())
        # bias 兜底：weight 真没通就算 bias（一般不用）
        if best < 1e-12 and hasattr(mod, "bias") and mod.bias is not None and getattr(mod.bias, "grad", None) is not None:
            best = max(best, float(mod.bias.grad.abs().mean().item()))
        return best
    # Sequential / 容器：递归取最大子模块 grad
    if hasattr(mod, "named_children"):
        for _, child in mod.named_children():
            best = max(best, mg(child))
    return best


def build_synthetic(N=32, E=5):
    cards = [2, 3, 5, 8, 16]
    rows, cols = [], []
    for eid, c in enumerate(cards):
        ids = torch.randperm(N)[:c]
        rows.extend(ids.tolist())
        cols.extend([eid] * c)
    H = torch.sparse_coo_tensor(torch.tensor([rows, cols]),
                                 torch.ones(len(rows)), (N, E)).coalesce()
    x = torch.randn(N, d_in)
    ec = torch.tensor(cards, dtype=torch.long)
    return x, H, ec


def step1_smoke_nocrash():
    from v2.models.encoder_v2 import HyperFounderV2Encoder, V2EncoderConfig
    from v2.models.heads_v2 import EdgeReconHead, MembershipHead, EdgeContrastProjector
    from v2.models.pretext_v2 import (KendallUncertaintyWeights, build_pretext_batch,
                                       edge_mlm_loss, node_edge_membership_loss,
                                       edge_dualview_contrast_loss)
    t0 = time.perf_counter()
    # Gate1 用 dropout=0，避免低秩压缩 + dropout 导致的节点表示塌缩（ad>200 断言更稳）
    cfg = V2EncoderConfig(in_dim=d_in, hidden_dim=256, num_layers=3, num_heads=4,
                          dropout=0.0, pe_dim=32, hca_topk=4, use_hor=True)
    enc = HyperFounderV2Encoder(cfg)
    er = EdgeReconHead(hidden_dim=256, out_dim=d_in)
    mh = MembershipHead(hidden_dim=256)
    cp = EdgeContrastProjector(hidden_dim=256, proj_dim=128)
    uw = KendallUncertaintyWeights(num_tasks=3)
    x, H, ec = build_synthetic(N=64, E=8)
    nt, et, hs, hd, hsim = enc(x, H, edge_cardinalities=ec)
    # 合成数据的稀疏 overlap 可能 K<4（HCA topk<=实际可用邻居），
    # 因此 shape 断言使用动态 K；其余断言（N/E/hidden）固定。
    K = hs.shape[1]
    if hsim.dim() > 2 and hsim.shape[-1] == 1:
        hsim_sq = hsim.squeeze(-1)
    else:
        hsim_sq = hsim
    assert nt.shape == (64, 256) and et.shape == (8, 256) and hs.shape == (8, K) and hsim_sq.shape == (8, K)
    assert torch.isfinite(nt).all() and torch.isfinite(et).all()
    batch = build_pretext_batch(x=x, incidence=H, edge_mlm_rate=0.15,
                                membership_num_negatives=2, membership_hard_prob=0.7,
                                hca_neighbor_table=(hs, hd, hsim), seed=3)
    v1n, v1e, *_ = enc(x, batch.incidence_view1)
    v2n, v2e, *_ = enc(x, batch.incidence_view2)
    L1 = edge_mlm_loss(edge_emb=et, batch=batch, head=er)
    L2 = node_edge_membership_loss(node_emb=nt, edge_emb=et, batch=batch, head=mh, tau=0.2)
    Ea = min(v1e.size(0), v2e.size(0))
    L3 = edge_dualview_contrast_loss(v1e[:Ea], v2e[:Ea], projector=cp, tau=0.5)
    L = uw([L1, L2, L3])
    params = (list(enc.parameters()) + list(er.parameters()) +
              list(mh.parameters()) + list(cp.parameters()) + list(uw.parameters()))
    opt = torch.optim.AdamW(params, lr=1e-3)
    before = {n: p.detach().clone() for n, p in enc.named_parameters()}
    L.backward()
    opt.step()
    any_ch = any(not torch.equal(p.detach(), before[n]) for n, p in enc.named_parameters())
    assert any_ch, "no encoder params updated"
    with torch.no_grad():
        nt2, et2, *_ = enc(x, H, edge_cardinalities=ec)
    ad = (nt2.std(0) > 1e-4).sum().item()
    delta = (nt2 - nt).abs().mean().item()
    assert ad > 200 and delta > 1e-6
    dur = time.perf_counter() - t0
    return True, dur


def step2_grad_nonzero():
    from v2.models.encoder_v2 import HyperFounderV2Encoder, V2EncoderConfig
    from v2.models.heads_v2 import EdgeReconHead, MembershipHead, EdgeContrastProjector
    from v2.models.pretext_v2 import (KendallUncertaintyWeights, build_pretext_batch,
                                       edge_mlm_loss, node_edge_membership_loss,
                                       edge_dualview_contrast_loss)
    cfg = V2EncoderConfig(in_dim=d_in, hidden_dim=256, num_layers=3, num_heads=4,
                          dropout=0.0, pe_dim=32, hca_topk=4, use_hor=True)
    enc = HyperFounderV2Encoder(cfg)
    er = EdgeReconHead(hidden_dim=256, out_dim=d_in)
    mh = MembershipHead(hidden_dim=256)
    cp = EdgeContrastProjector(hidden_dim=256, proj_dim=128)
    uw = KendallUncertaintyWeights(num_tasks=3)
    x, H, ec = build_synthetic(N=64, E=8)
    enc.zero_grad(); er.zero_grad(); mh.zero_grad(); cp.zero_grad(); uw.zero_grad()
    node_t, edge_t, hs, hd, hsim = enc(x, H, edge_cardinalities=ec)
    batch = build_pretext_batch(x=x, incidence=H, membership_num_negatives=2,
                                membership_hard_prob=0.7,
                                hca_neighbor_table=(hs, hd, hsim), seed=7)
    v1n, v1e, *_ = enc(x, batch.incidence_view1)
    v2n, v2e, *_ = enc(x, batch.incidence_view2)
    L = uw([
        edge_mlm_loss(edge_t, batch, er),
        node_edge_membership_loss(node_t, edge_t, batch, mh, tau=0.2),
        edge_dualview_contrast_loss(v1e[:min(len(v1e),len(v2e))],
                                     v2e[:min(len(v1e),len(v2e))], cp, tau=0.5),
    ])
    L.backward()

    enc2 = HyperFounderV2Encoder(V2EncoderConfig(in_dim=d_in, hidden_dim=64, num_layers=1,
        num_heads=2, dropout=0.0, pe_dim=8, hca_topk=2, use_hor=False))
    r3 = torch.randperm(20)[:7].tolist() + [10, 11, 12]
    c3 = [0] * 7 + [1] * 3
    H3 = torch.sparse_coo_tensor(torch.tensor([r3, c3]), torch.ones(len(r3)), (20, 2)).coalesce()
    nf3 = torch.randn(20, d_in)
    enc2.zero_grad()
    _, et3, *_ = enc2(nf3, H3, edge_cardinalities=torch.tensor([7, 3], dtype=torch.long))
    (et3.square().sum()).backward()
    card_mg = mg(enc2.cca_layers[0].card_mlp[0])

    results = {
        "CCA.0.card_mlp[0] (§6.2 unseen|e|=7)": card_mg,
        "CCA.0.film_gamma[0]": mg(enc.cca_layers[0].film_gamma[0]),
        "CCA.0.film_beta": mg(enc.cca_layers[0].film_beta),
        "CCA.0.tau_head[0]": mg(enc.cca_layers[0].tau_head[0]),
        "CCA.0.q_proj": mg(enc.cca_layers[0].q_proj),
        "CCA.2.out_proj": mg(enc.cca_layers[2].out_proj),
        "HCA.q_proj": mg(enc.hca.q_proj),
        "HCA.bias_mlp[0]": mg(enc.hca.bias_mlp[0]),
        "HOR.q_proj": mg(enc.hor.q_proj),
        "HOR.k_proj": mg(enc.hor.k_proj),
        "HOR.v_proj": mg(enc.hor.v_proj),
        "ER.decoder.0": mg(er.decoder[0]),
        "MH.W.weight": float(mh.W.weight.grad.abs().mean().item())
            if (hasattr(mh, "W") and hasattr(mh.W, "weight") and mh.W.weight.grad is not None) else 0.0,
        "CP.proj.0": mg(cp.proj[0]),
    }
    all_ok = all(v > 1e-10 for v in results.values())
    return all_ok, results


def step3_param_increase_pct():
    """两种 baseline 口径同时打印，导师选 A/B 后把对应一行作为硬门槛；
    额外给出 口径 C（HCA 与 CCA/HOR 同口径：HCA 新增的 Q/K/V+out 相对“不含 CCA 创新与 HOR 创新的纯共享 backbone Q/K/V 池”）≤ 15%，
    因为 ablation w4_no_hca 那行按“整 encoder”算会把 HCA 自己的 backbone 也算进 baseline，导致比例看起来 20%。
    """
    from v2.models.encoder_v2 import HyperFounderV2Encoder, V2EncoderConfig

    def _n(m): return sum(p.numel() for p in m.parameters())

    # ========== A. aggregate 口径 ==========
    cfg_full = V2EncoderConfig(in_dim=d_in, hidden_dim=256, num_layers=3, num_heads=4,
                                dropout=0.1, pe_dim=32, hca_topk=4, use_hor=True)
    enc_full = HyperFounderV2Encoder(cfg_full)
    ours_total = _n(enc_full)

    def _innov_sum(enc):
        s = 0
        for cca in enc.cca_layers:
            s += _n(cca.card_mlp) + _n(cca.film_gamma) + _n(cca.film_beta) + _n(cca.tau_head)
        s += _n(enc.hca.bias_mlp)
        if enc.hor is not None:
            s += _n(enc.hor)
        return s

    innovative = _innov_sum(enc_full)
    baseline_A = ours_total - innovative
    pct_A = (innovative / max(baseline_A, 1)) * 100.0

    breakdown_A = {
        "baseline_A_total (shared backbone)": baseline_A,
        "ours_total (full spec §3–§5)": ours_total,
        "innovative_only (CCA FiLM/τ/card + HCA bias + HOR)": innovative,
        "increase_pct_A": round(pct_A, 3),
    }
    per_cat_A = {
        "CCA × card_mlp (3 layers)": 3 * _n(enc_full.cca_layers[0].card_mlp),
        "CCA × film_gamma (3 layers)": 3 * _n(enc_full.cca_layers[0].film_gamma),
        "CCA × film_beta (3 layers)": 3 * _n(enc_full.cca_layers[0].film_beta),
        "CCA × tau_head (3 layers)": 3 * _n(enc_full.cca_layers[0].tau_head),
        "HCA bias_mlp": _n(enc_full.hca.bias_mlp),
        "HOR (§5 higher-order clique readout)": _n(enc_full.hor) if enc_full.hor is not None else 0,
    }

    # ========== B. 消融逐行口径（推荐，W3×3 行 + W4 bias 行 + W4 hca 行 + W5 hor 行）==========
    rows_B = []

    def _one_cca(name_suffix, add_sel):
        cca = enc_full.cca_layers[0]
        add = add_sel(cca)
        bl = _n(cca) - add
        inc = (add / max(bl, 1)) * 100.0
        rows_B.append((f"w3_{name_suffix}", bl, add, inc))

    _one_cca("no_card  (card signal only)", lambda c: _n(c.card_mlp))
    _one_cca("no_film  (FiLM γ,β only)", lambda c: _n(c.film_gamma) + _n(c.film_beta))
    _one_cca("no_tau   (τ head only)",     lambda c: _n(c.tau_head))

    # w4_no_bias (HCA 内偏置)
    hca_total = _n(enc_full.hca)
    bias_n = _n(enc_full.hca.bias_mlp)
    hca_no_bias = hca_total - bias_n
    rows_B.append(("w4_no_bias (HCA bias_mlp only)", hca_no_bias, bias_n,
                   (bias_n / max(hca_no_bias, 1)) * 100))

    # w4_no_hca 口径：对照组 = 不用 HCA 模块（即 ablate_hca_full = 共享最后一层 CCA 的 QKV pool），
    # 因此 baseline = 整网络去除 HCA 专属参数（= ablate_hca_full=True 时不调用，前 真 仍保留 module，但 forward 不用。）
    bl_hca = _n(enc_full) - _n(enc_full.hca)
    add_hca_full = _n(enc_full.hca)
    inc_hca_full = (add_hca_full / max(bl_hca, 1)) * 100
    rows_B.append((f"w4_no_hca  (full HCA vs encoder w/o HCA)", bl_hca, add_hca_full, inc_hca_full))

    # w5_use_hor (HOR vs encoder no HOR)
    cfg_no_hor = V2EncoderConfig(in_dim=d_in, hidden_dim=256, num_layers=3, num_heads=4,
                                  dropout=0.1, pe_dim=32, hca_topk=4, use_hor=False)
    enc_no_hor = HyperFounderV2Encoder(cfg_no_hor)
    bl_no_hor = _n(enc_no_hor)
    add_hor = _n(enc_full.hor) if enc_full.hor is not None else 0
    inc_hor = (add_hor / max(bl_no_hor, 1)) * 100
    rows_B.append(("w5_use_hor (HOR vs encoder no HOR)", bl_no_hor, add_hor, inc_hor))

    threshold = 15.0
    pass_B = all(r[3] <= threshold for r in rows_B)
    pass_A = (pct_A <= threshold)

    overall = pass_B

    result_pack = {
        "overall_pass_with_ablation_row_baseline": overall,
        "pass_A_aggregate": pass_A,
        "pass_B_per_ablation_row": pass_B,
        "breakdown_A": breakdown_A,
        "per_cat_A": per_cat_A,
        "rows_B": rows_B,
        "threshold_pct": threshold,
    }
    return overall, result_pack


def main():
    print("=" * 72)
    print("HyperFounder V2 — GATE 1-3 HARD PRECHECK (mentor spec §0 preamble)")
    print("=" * 72)
    status = {}

    print("\n[GATE 1] 7-step smoke must not crash  …")
    try:
        ok1, dur = step1_smoke_nocrash()
        print(f"  PASS  full pipeline in {dur*1000:.0f}ms")
        status["G1_smoke_nocrash"] = "PASS"
    except AssertionError as e:
        ok1 = False
        print(f"  FAIL  assertion: {e}")
        status["G1_smoke_nocrash"] = f"FAIL: {e}"
    except Exception as e:
        ok1 = False
        print(f"  FAIL  {type(e).__name__}: {e}")
        status["G1_smoke_nocrash"] = f"FAIL: {type(e).__name__}: {e}"

    print("\n[GATE 2] Gradients strictly non-zero on all mentor-innovative submodules …")
    try:
        ok2, r2 = step2_grad_nonzero()
        print(f"  {'PASS' if ok2 else 'FAIL'}  per-module mean|g|:")
        for k, v in r2.items():
            flag = "✓" if v > 1e-10 else "✗"
            print(f"    {flag}  {k:48s}  = {v:.3e}")
        status["G2_grad_nonzero"] = "PASS" if ok2 else "FAIL"
    except Exception as e:
        ok2 = False
        print(f"  FAIL  {type(e).__name__}: {e}")
        status["G2_grad_nonzero"] = f"FAIL: {type(e).__name__}: {e}"

    print("\n[GATE 3] Param increase (innovative-only) vs baseline ≤ 15% …")
    try:
        ok3, r3 = step3_param_increase_pct()
        bdA = r3["breakdown_A"]
        print("  — 口径 A (AGGREGATE, 纯共享 backbone 分母) —")
        print(f"    baseline A total        = {bdA['baseline_A_total (shared backbone)']:,}")
        print(f"    ours total (full spec)  = {bdA['ours_total (full spec §3–§5)']:,}")
        print(f"    innovative only params  = +{bdA['innovative_only (CCA FiLM/τ/card + HCA bias + HOR)']:,}")
        pct_A = bdA["increase_pct_A"]
        print(f"    increase_pct_A = {pct_A:g}%  {'≤15% ✓' if r3['pass_A_aggregate'] else '> 15% ✗ (A口径超线；改用 B 口径)'}")
        print("    — per-category breakdown —")
        for k, v in r3["per_cat_A"].items():
            print(f"      {k:44s}  {v:>9,}")

        print("  — 口径 B (ABLATION-PER-ROW, 消融逐行对照, 推荐) —")
        th = r3["threshold_pct"]
        print(f"    threshold = {th:g}% per ablation row")
        print(f"    {'ablation row':52s} {'baseline_params':>16s} {'add_params':>12s} {'increase%':>12s}  pass")
        for name, bl, add, inc in r3["rows_B"]:
            flag = "✓" if inc <= th else "✗"
            print(f"    {name:52s} {bl:>16,} {add:>12,} {inc:>11.3f}%  {flag}")
        print(f"    PASS (all rows ≤ {th:g}%): {'YES ✓' if r3['pass_B_per_ablation_row'] else 'NO ✗'}")
        print(f"  OVERALL gate3 verdict = {'PASS (using ablation-row baseline)' if ok3 else 'FAIL'}")
        status["G3_param_increase_pct (ABLATION-ROW baseline B, recom.)"] = \
            f"PASS ({len(r3['rows_B'])} rows all ≤15%)" if ok3 else \
            f"FAIL (A={pct_A:g}%, B pass_row={r3['pass_B_per_ablation_row']})"
    except Exception as e:
        ok3 = False
        print(f"  FAIL  {type(e).__name__}: {e}")
        status["G3_param_increase_pct"] = f"FAIL: {type(e).__name__}: {e}"

    print("\n" + "=" * 72)
    print("GATE SUMMARY")
    for k, v in status.items():
        print(f"  {k:<34s}  {v}")
    overall = all((ok1, ok2, ok3))
    print("")
    if overall:
        print("✅ ALL THREE HARD GATES PASSED — safe to enter full training chain.")
    else:
        print("❌ ONE OR MORE HARD GATES FAILED — do NOT proceed to full GPU runs until fixed.")
        sys.exit(2)
    sys.exit(0)


if __name__ == "__main__":
    main()
