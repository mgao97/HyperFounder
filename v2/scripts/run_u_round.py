import csv, sys, torch, yaml
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from v2.models.encoder_v2 import HyperFounderV2Encoder, V2EncoderConfig
from v2.models.heads_v2 import EdgeReconHead, MembershipHead, EdgeContrastProjector
from v2.models.pretext_v2 import _edge_mean_features
from v2.utils.dhg_datasets import load_domain_graphs
from v2.scripts.run_w1w2_lodo_linearprobe import _linear_probe, _split_masks


def _edge_cardinalities(sp: torch.Tensor) -> torch.Tensor:
    sp = sp.coalesce()
    e_idx = sp.indices()[1]
    num_edges = sp.size(1)
    cnt = torch.zeros(num_edges, device=sp.device, dtype=torch.long)
    cnt.index_add_(0, e_idx, torch.ones_like(e_idx))
    return cnt

def compute_u1_metrics(Z, n_classes):
    Z_c = Z - Z.mean(axis=0, keepdims=True)
    try:
        _, S, _ = np.linalg.svd(Z_c, full_matrices=False)
        S = S + 1e-8
        eff_rank = (S.sum() ** 2) / (S ** 2).sum()
        log_S = np.log(S)
        log_idx = np.log(np.arange(1, len(S) + 1))
        slope, _ = np.polyfit(log_idx, log_S, 1)
    except:
        eff_rank, slope = 0.0, 0.0

    try:
        km = KMeans(n_clusters=n_classes, random_state=0, n_init=10)
        sil = silhouette_score(Z, km.fit_predict(Z))
    except:
        sil = 0.0
    return float(eff_rank), float(slope), float(sil)

def get_subspace_projection(G, k):
    G_c = G - G.mean(axis=0, keepdims=True)
    _, _, Vh = np.linalg.svd(G_c, full_matrices=False)
    Vk = Vh[:k].T
    return Vk @ Vk.T

def get_random_projection(d, k):
    R = np.random.randn(d, k)
    Q, _ = np.linalg.qr(R)
    return Q @ Q.T


def _cuda_mem(prefix: str) -> None:
    if not torch.cuda.is_available():
        print(f"[mem] {prefix} cpu-only")
        return
    dev = torch.cuda.current_device()
    alloc = torch.cuda.memory_allocated(dev) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(dev) / (1024 ** 3)
    peak = torch.cuda.max_memory_allocated(dev) / (1024 ** 3)
    print(f"[mem] {prefix} alloc={alloc:.2f}G reserved={reserved:.2f}G peak={peak:.2f}G")


def _edge_grad_to_node_grad(edge_grad: torch.Tensor, incidence: torch.Tensor, num_nodes: int) -> torch.Tensor:
    sp = incidence.coalesce()
    n_idx, e_idx = sp.indices()
    out = torch.zeros(num_nodes, edge_grad.size(1), device=edge_grad.device, dtype=edge_grad.dtype)
    cnt = torch.zeros(num_nodes, 1, device=edge_grad.device, dtype=edge_grad.dtype)
    if n_idx.numel() == 0:
        return out
    out.index_add_(0, n_idx, edge_grad[e_idx])
    cnt.index_add_(0, n_idx, torch.ones(n_idx.numel(), 1, device=edge_grad.device, dtype=edge_grad.dtype))
    return out / cnt.clamp_min(1.0)


def _compute_frozen_tokens_and_overfit_proxy(
    enc,
    er,
    mh,
    x: torch.Tensor,
    H: torch.Tensor,
    H_e: torch.Tensor,
):
    # U2 uses a memory-bounded proxy: compute frozen tokens once, then estimate
    # overfit directions with sampled/chunked MLM+membership gradients on detached
    # token tensors. This preserves the training-free nature while keeping a single
    # 24G GPU within budget.
    _cuda_mem("u2-enter")
    with torch.no_grad():
        node_t, edge_t, _, _, _ = enc(x, H, edge_cardinalities=H_e)
    print(
        f"[u2-shape] nodes={node_t.size(0)} edges={edge_t.size(0)} dim={node_t.size(1)} "
        f"inc_nnz={H._nnz()}"
    )
    _cuda_mem("after-encode")

    node_t_det = node_t.detach().requires_grad_(True)
    edge_t_det = edge_t.detach().requires_grad_(True)

    if node_t_det.grad is not None:
        node_t_det.grad.zero_()
    if edge_t_det.grad is not None:
        edge_t_det.grad.zero_()

    # ---- sampled MLM proxy -------------------------------------------------
    g_cpu = torch.Generator(device="cpu").manual_seed(0)
    E = edge_t_det.size(0)
    max_mask_edges = min(E, 4096)
    edge_mask_idx = torch.randperm(E, generator=g_cpu, device="cpu")[:max_mask_edges].to(edge_t_det.device)
    edge_mask_target = _edge_mean_features(x, H)[edge_mask_idx].to(edge_t_det.dtype)
    mlm_chunk = 1024
    mlm_loss_total = 0.0
    for lo in range(0, edge_mask_idx.numel(), mlm_chunk):
        hi = min(lo + mlm_chunk, edge_mask_idx.numel())
        idx = edge_mask_idx[lo:hi]
        pred = er(edge_t_det[idx])
        chunk_loss = torch.nn.functional.mse_loss(pred, edge_mask_target[lo:hi], reduction="mean")
        scaled = chunk_loss * (idx.numel() / max(edge_mask_idx.numel(), 1))
        scaled.backward()
        mlm_loss_total += float(scaled.detach().cpu())

    # ---- sampled membership proxy -----------------------------------------
    sp = H.coalesce()
    n_idx, e_idx = sp.indices()
    total_pos = n_idx.numel()
    max_pos = min(total_pos, 32768)
    pos_pick = torch.randperm(total_pos, generator=g_cpu, device="cpu")[:max_pos].to(node_t_det.device)
    pos_nodes = n_idx[pos_pick]
    pos_edges = e_idx[pos_pick]

    neg_per_pos = 2
    neg_chunk_pos = 4096
    mem_loss_total = 0.0
    num_edges = edge_t_det.size(0)
    for lo in range(0, max_pos, neg_chunk_pos):
        hi = min(lo + neg_chunk_pos, max_pos)
        cur_nodes = pos_nodes[lo:hi]
        cur_pos_edges = pos_edges[lo:hi]
        cur_m = cur_nodes.numel()
        if cur_m == 0:
            continue
        pos_s = mh(node_t_det[cur_nodes], edge_t_det[cur_pos_edges])  # [m]
        neg_edges = torch.randint(
            0, num_edges, (cur_m * neg_per_pos,), generator=g_cpu, device="cpu"
        ).to(node_t_det.device)
        same = neg_edges.view(cur_m, neg_per_pos) == cur_pos_edges.unsqueeze(1)
        if bool(same.any()):
            repl = torch.randint(
                0, num_edges, (cur_m * neg_per_pos,), generator=g_cpu, device="cpu"
            ).to(node_t_det.device).view(cur_m, neg_per_pos)
            neg_edges = torch.where(same, repl, neg_edges.view(cur_m, neg_per_pos)).view(-1)
        neg_nodes = cur_nodes.repeat_interleave(neg_per_pos)
        neg_s = mh(node_t_det[neg_nodes], edge_t_det[neg_edges]).view(cur_m, neg_per_pos)
        logits = torch.cat([pos_s.unsqueeze(-1), neg_s], dim=-1) / 0.2
        labels = torch.zeros(cur_m, dtype=torch.long, device=logits.device)
        chunk_loss = torch.nn.functional.cross_entropy(logits, labels)
        scaled = chunk_loss * (cur_m / max(max_pos, 1))
        scaled.backward()
        mem_loss_total += float(scaled.detach().cpu())

    print(
        f"[u2-batch] mlm_mask={edge_mask_idx.numel()} sampled_pos={max_pos} sampled_neg={max_pos * neg_per_pos}"
    )
    print(f"[u2-loss] mem={mem_loss_total:.6f} mlm={mlm_loss_total:.6f}")
    _cuda_mem("after-backward")

    node_grad = node_t_det.grad if node_t_det.grad is not None else torch.zeros_like(node_t_det)
    edge_grad = edge_t_det.grad if edge_t_det.grad is not None else torch.zeros_like(edge_t_det)
    G = node_grad + _edge_grad_to_node_grad(edge_grad, H, node_t_det.size(0))
    return node_t.detach().cpu().numpy(), G.detach().cpu().numpy()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(ROOT / "v2/configs/pretrain_v2.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    graphs_by_domain = load_domain_graphs(cfg, seed=42, require_node_splits=True)
    nodecls = [(dom, g.name or g.dataset_name or dom, g) for dom, gs in graphs_by_domain.items() for g in gs if g.node_labels is not None]
    
    out_csv = ROOT / "outputs_v2/u_round/u_round_results.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "group", "seed", "fraction", "dataset", "baseline_acc", "Z_acc", "delta_pp",
        "u1_eff_rank", "u1_slope", "u1_sil",
        "ov1_acc", "rand1_acc", "pca1_acc",
        "ov2_acc", "rand2_acc", "pca2_acc",
        "ov3_acc", "rand3_acc", "pca3_acc"
    ]
    
    with open(out_csv, "w", newline="") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=fields)
        writer.writeheader()

        num_rows = 0
        for group in ["none", "cca_hor"]:
            for seed in [1, 7, 13]:
                for frac in [12, 25, 38, 50, 62, 75, 88, 100]:
                    ckpt_path = ROOT / f"outputs_v2/u_checkpoints/{group}/seed_{seed}/checkpoints/pretrain_frac_{frac:02d}_v2.pt"
                    if not ckpt_path.exists():
                        print(f"[skip] missing ckpt: group={group} seed={seed} frac={frac}")
                        continue
                    
                    enc_cfg = V2EncoderConfig(
                        in_dim=cfg["model"]["input_dim"], hidden_dim=cfg["model"]["hidden_dim"], num_layers=3, num_heads=4,
                        use_hor=(group=="cca_hor"), ablate_hca_full=True,
                        ablate_cca_card=(group=="none"), ablate_cca_film=(group=="none"), ablate_cca_tau=(group=="none"), ablate_hca_bias=(group=="none")
                    )
                    enc = HyperFounderV2Encoder(enc_cfg).to(device)
                    er = EdgeReconHead(enc_cfg.hidden_dim, enc_cfg.in_dim).to(device)
                    mh = MembershipHead(enc_cfg.hidden_dim).to(device)
                    cp = EdgeContrastProjector(enc_cfg.hidden_dim, 128).to(device)
                    
                    sd = torch.load(ckpt_path, map_location=device)
                    enc.load_state_dict(sd["encoder"], strict=False)
                    if "edge_recon_head" in sd: er.load_state_dict(sd["edge_recon_head"])
                    if "membership_head" in sd: mh.load_state_dict(sd["membership_head"])
                    if "edge_contrast_proj" in sd: cp.load_state_dict(sd["edge_contrast_proj"])
                    enc.eval(); er.eval(); mh.eval(); cp.eval()
                    
                    print(f"[ckpt] group={group} seed={seed} frac={frac} start")
                    for dom, ds_name, g in nodecls:
                        try:
                            tr, va, te = _split_masks(g, 42)
                            train_use = tr | va
                            labels = g.node_labels.cpu().numpy()
                            n_classes = int(labels.max() + 1)
                            x = g.x.to(device).float()
                            H = g.incidence_matrix().to_sparse_coo().coalesce().to(device)
                            H_e = _edge_cardinalities(H)

                            Z, G = _compute_frozen_tokens_and_overfit_proxy(enc, er, mh, x, H, H_e)
                            X_raw = x.detach().cpu().numpy()

                            def _acc(features):
                                return _linear_probe(features[train_use], labels[train_use], features[te], labels[te], seed=42, device=device)

                            base_acc = _acc(X_raw)
                            Z_acc = _acc(Z)
                            u1_eff, u1_slope, u1_sil = compute_u1_metrics(Z, n_classes)

                            row = {"group": group, "seed": seed, "fraction": frac, "dataset": ds_name,
                                   "baseline_acc": base_acc, "Z_acc": Z_acc, "delta_pp": (Z_acc - base_acc)*100,
                                   "u1_eff_rank": u1_eff, "u1_slope": u1_slope, "u1_sil": u1_sil}

                            # U2 子空间切除变体
                            for k in [1, 2, 3]:
                                row[f"ov{k}_acc"] = _acc(Z - Z @ get_subspace_projection(G, k))
                                row[f"pca{k}_acc"] = _acc(Z - Z @ get_subspace_projection(Z, k))
                                row[f"rand{k}_acc"] = _acc(Z - Z @ get_random_projection(Z.shape[1], k))

                            writer.writerow(row)
                            f_csv.flush()
                            num_rows += 1
                            print(f"[row] {group} seed={seed} frac={frac} dataset={ds_name} delta={row['delta_pp']:.2f}")
                        finally:
                            if "x" in locals():
                                del x
                            if "H" in locals():
                                del H
                            if "H_e" in locals():
                                del H_e
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

        print(f"[done] wrote {num_rows} rows to {out_csv}")

if __name__ == "__main__":
    main()
