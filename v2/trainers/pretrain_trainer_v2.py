"""v2 pretraining trainer — strict match encoder-design-spec.md (2026-09-01).

Pipeline per spec:

  1. Load hypergraphs from config (flat, domain tag only for log diversity).
  2. Build HyperFounderV2Encoder (CCA×3 + HCA + optional HOR) + 3 heads +
     Kendall 3-task homoscedastic uncertainty (§5).
  3. Each step:
       a. Sample subhypergraph → get node features x, incidence H, edge cards.
       b. Encoder forward → returns (node_t, edge_t, hca_nbr_src, hca_nbr_dst,
          hca_nbr_sim).  HCA overlap Top-K table is *computed once* and then
          re-used two ways: (i) inside HCA forward itself, (ii) passed into
          `build_pretext_batch` for 70% hard negative membership sampling.
       c. Build pretext batch = edge-MLM 15% mask + (n,e+,e−) triples + dual
          node/edge-drop views.
       d. Two-view forward through encoder → v1e/v2e for dual contrast.
       e. Compute 3 losses:
            • L1 = edge_mlm_loss       (MSE 2-layer decoder)
            • L2 = node_edge_membership_loss   (InfoNCE, τ=0.2)
            • L3 = edge_dualview_contrast_loss (symmetric InfoNCE, τ=0.5)
       f. Kendall UW: L = Σ_i [ exp(−s_i)·L_i + 0.5·s_i ]  (s_i = log σ_i²)
       g. Backward + AdamW, with grad-clip.
  4. Save best ckpt by total loss, persist epoch/step CSV + JSON summary.

AMP note: encoder body forces float32 (scatter + sparse.mm + overlap-CSR do not
play well with bf16) so AMP is off by default in pretrain_v2.yaml.
"""
from __future__ import annotations

import csv as _csv
import json
import time
from pathlib import Path
from typing import Dict, List

import torch
from torch import nn

from v2.models.encoder_v2 import HyperFounderV2Encoder, V2EncoderConfig, build_encoder_v2
from v2.models.heads_v2 import (
    EdgeContrastProjector,
    EdgeReconHead,
    MembershipHead,
)
from v2.models.pretext_v2 import (
    KendallUncertaintyWeights,
    PretextBatchV2,
    ResidualUncertaintyWeights,
    VariationalBottleneck,
    build_pretext_batch,
    edge_dualview_contrast_loss,
    edge_mlm_loss,
    node_edge_membership_loss,
)
from v2.trainers.trainer_base import TrainerBase
from v2.utils.dhg_datasets import load_domain_graphs
from v2.utils.hypergraph import SimpleHypergraph, iter_graphs
from v2.utils.minibatch_sampling import sample_online_subhypergraph


def _to_sparse_incidence(hg: SimpleHypergraph) -> torch.Tensor:
    inc = hg.incidence_matrix()
    if not inc.is_sparse:
        inc = inc.to_sparse_coo()
    return inc.coalesce()


def _edge_cardinalities(sp: torch.Tensor) -> torch.Tensor:
    sp = sp.coalesce()
    e_idx = sp.indices()[1]
    E = sp.size(1)
    cnt = torch.zeros(E, device=sp.device, dtype=torch.long)
    cnt.index_add_(0, e_idx, torch.ones_like(e_idx))
    return cnt


class V2PretrainTrainer(TrainerBase):
    def __init__(self, config: Dict):
        super().__init__(config, ensure_subdirs=("checkpoints", "logs", "results"))
        self._log(f"Device: {self.device}")

        # -------------------------------------------------------------- data
        self._log("Loading hypergraphs…")
        self.domains = load_domain_graphs(config, seed=int(config["training"]["seed"]))
        self.graphs: List[SimpleHypergraph] = list(iter_graphs(self.domains))
        if not self.graphs:
            raise RuntimeError("No graphs loaded from config.")
        in_dim = self.graphs[0].x.size(-1)
        self._log(f"Loaded {len(self.graphs)} hypergraphs; in_dim={in_dim}")

        # ------------------------------------------------------------ model
        m = config["model"]
        hidden_dim = int(m.get("hidden_dim", 256))
        t_cfg = config.get("training", {})
        self.encoder: HyperFounderV2Encoder = build_encoder_v2(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_layers=int(m.get("num_layers", 3)),
            num_heads=int(m.get("num_heads", 4)),
            dropout=float(m.get("dropout", 0.1)),
            pe_dim=int(m.get("pe_dim", 32)),
            hca_topk=int(m.get("hca_topk", 16)),
            use_hor=bool(m.get("use_hor", True)),
            ablate_cca_card=bool(t_cfg.get("ablate_cca_card", m.get("ablate_cca_card", False))),
            ablate_cca_film=bool(t_cfg.get("ablate_cca_film", m.get("ablate_cca_film", False))),
            ablate_cca_tau=bool(t_cfg.get("ablate_cca_tau", m.get("ablate_cca_tau", False))),
            ablate_hca_bias=bool(t_cfg.get("ablate_hca_bias", m.get("ablate_hca_bias", False))),
            ablate_hca_full=bool(t_cfg.get("ablate_hca_full", m.get("ablate_hca_full", False))),
        ).to(self.device)

        # Three heads (§5)
        self.edge_recon_head = EdgeReconHead(hidden_dim=hidden_dim, out_dim=in_dim).to(self.device)
        self.membership_head = MembershipHead(hidden_dim=hidden_dim).to(self.device)
        proj_dim = int(config["training"].get("edge_dualview_proj_dim", 128))
        self.edge_contrast_proj = EdgeContrastProjector(hidden_dim=hidden_dim, proj_dim=proj_dim).to(self.device)

        # Task weighting (baseline Kendall / T4 residual uncertainty / fixed weights)
        self.uncertainty_mode = str(config["training"].get("uncertainty_mode", "kendall")).lower()
        self.use_kendall_uw = bool(config["training"].get("use_kendall_uw", True))
        if self.uncertainty_mode == "kendall" and self.use_kendall_uw:
            self.uw = KendallUncertaintyWeights(num_tasks=3).to(self.device)
        elif self.uncertainty_mode == "residual":
            self.uw = ResidualUncertaintyWeights(num_tasks=3).to(self.device)
        else:
            self.uw = None

        # T1 information bottleneck regularization
        self.use_ib = bool(config["training"].get("use_ib", False))
        self.ib_beta = float(config["training"].get("ib_beta", 0.0))
        self.ib_latent_dim = int(config["training"].get("ib_latent_dim", hidden_dim))
        self.ib_mlm = VariationalBottleneck(hidden_dim, self.ib_latent_dim).to(self.device) if self.use_ib else None
        self.ib_mem = VariationalBottleneck(hidden_dim, self.ib_latent_dim).to(self.device) if self.use_ib else None
        self.ib_dual = VariationalBottleneck(hidden_dim, self.ib_latent_dim).to(self.device) if self.use_ib else None

        # ----------------------------------------------------------- params
        parameters = (
            list(self.encoder.parameters())
            + list(self.edge_recon_head.parameters())
            + list(self.membership_head.parameters())
            + list(self.edge_contrast_proj.parameters())
        )
        if self.uw is not None:
            parameters += list(self.uw.parameters())
        if self.use_ib:
            parameters += (
                list(self.ib_mlm.parameters())
                + list(self.ib_mem.parameters())
                + list(self.ib_dual.parameters())
            )
        self.optimizer = torch.optim.AdamW(
            parameters,
            lr=float(config["training"].get("lr", 1e-3)),
            weight_decay=float(config["training"].get("weight_decay", 1e-4)),
        )

        amp = config.get("training", {}).get("amp", {})
        self.use_amp = bool(amp.get("enabled", False)) and self.device.type == "cuda"
        self.amp_dtype = torch.bfloat16 if str(amp.get("dtype", "float32")).lower() == "bf16" else torch.float16
        self.grad_clip_norm = float(config.get("training", {}).get("grad_clip_norm", 1.0))

        # --------------------------------------------------------- minibatch
        mb = config.get("training", {}).get("minibatch", {})
        self.mb_max_nodes = int(mb.get("max_nodes", 512))
        self.mb_max_edges = int(mb.get("max_edges", 256))
        self.mb_expansion_hops = int(mb.get("expansion_hops", 2))
        self.mb_seed_edges_per_sub = int(mb.get("seed_edges_per_subhypergraph", 3))
        self.mb_min_nodes = int(mb.get("min_subgraph_nodes", 64))

        # --------------------------------------------------------- pretext
        t = config["training"]
        self.pretext_edge_mlm_rate = float(t.get("edge_mlm_mask_rate", 0.15))
        self.pretext_num_neg = int(t.get("membership_num_negatives", 4))
        self.pretext_hard_prob = float(t.get("membership_hard_prob", 0.7))
        self.pretext_ndrop = float(t.get("dualview_node_drop", 0.15))
        self.pretext_edrop = float(t.get("dualview_edge_drop", 0.10))
        self.pretext_tau_mem = float(t.get("tau_membership", 0.2))
        self.pretext_tau_dual = float(t.get("tau_edge_dualview", 0.5))
        if self.uw is None:
            lw = t.get("loss_weights", {})
            self.fixed_w = (
                float(lw.get("edge_mlm", 1.0)),
                float(lw.get("membership_contrast", 1.0)),
                float(lw.get("edge_dualview_contrast", 1.0)),
            )
        else:
            self.fixed_w = None

        # ----------------------------------------------------------- loop
        self.epochs = int(t.get("epochs", 60))
        self.steps_per_epoch = int(t.get("steps_per_epoch", 32))
        es = t.get("early_stopping", {})
        self.patience = int(es.get("patience", 12))

        self.best_loss = float("inf")
        self.best_epoch = -1
        self.bad_epochs = 0
        self.step_losses: List[Dict[str, float]] = []
        self.epoch_losses: List[Dict[str, float]] = []
        fracs = t.get("checkpoint_fractions", []) or []
        self.checkpoint_fractions = sorted({float(v) for v in fracs if 0.0 < float(v) < 1.0})
        self.checkpoint_fraction_epochs = {
            frac: max(1, min(self.epochs, int(round(self.epochs * frac))))
            for frac in self.checkpoint_fractions
        }
        self.saved_fraction_ckpts: set[float] = set()

    # ----------------------------------------------------------------- I/O
    def _save_best(self, epoch: int, loss: float, pretext_loss: float | None = None):
        p = Path(self.config["training"]["output_dir"]) / "checkpoints" / "pretrain_best_v2.pt"
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "encoder": self.encoder.state_dict(),
            "edge_recon_head": self.edge_recon_head.state_dict(),
            "membership_head": self.membership_head.state_dict(),
            "edge_contrast_proj": self.edge_contrast_proj.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": epoch, "best_loss": loss, "best_pretext_loss": pretext_loss, "config": self.config,
        }
        if self.uw is not None:
            payload["uncertainty_weights"] = self.uw.state_dict()
            payload["uncertainty_mode"] = self.uncertainty_mode
        if self.use_ib:
            payload["ib_modules"] = {
                "mlm": self.ib_mlm.state_dict(),
                "mem": self.ib_mem.state_dict(),
                "dual": self.ib_dual.state_dict(),
                "beta": self.ib_beta,
                "latent_dim": self.ib_latent_dim,
            }
        torch.save(payload, p)
        self._log(f"✓ best ckpt epoch={epoch+1} loss={loss:.4f} → {p}")

    def _save_last(self, epoch: int):
        p = Path(self.config["training"]["output_dir"]) / "checkpoints" / "pretrain_last_v2.pt"
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "encoder": self.encoder.state_dict(),
            "edge_recon_head": self.edge_recon_head.state_dict(),
            "membership_head": self.membership_head.state_dict(),
            "edge_contrast_proj": self.edge_contrast_proj.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": epoch, "config": self.config,
        }
        if self.uw is not None:
            payload["uncertainty_weights"] = self.uw.state_dict()
            payload["uncertainty_mode"] = self.uncertainty_mode
        if self.use_ib:
            payload["ib_modules"] = {
                "mlm": self.ib_mlm.state_dict(),
                "mem": self.ib_mem.state_dict(),
                "dual": self.ib_dual.state_dict(),
                "beta": self.ib_beta,
                "latent_dim": self.ib_latent_dim,
            }
        torch.save(payload, p)

    def _save_fraction_ckpt(self, epoch: int, loss: float, frac: float, pretext_loss: float | None = None):
        tag = int(round(frac * 100))
        p = Path(self.config["training"]["output_dir"]) / "checkpoints" / f"pretrain_frac_{tag:02d}_v2.pt"
        payload = {
            "encoder": self.encoder.state_dict(),
            "edge_recon_head": self.edge_recon_head.state_dict(),
            "membership_head": self.membership_head.state_dict(),
            "edge_contrast_proj": self.edge_contrast_proj.state_dict(),
            "epoch": epoch,
            "loss": loss,
            "pretext_loss": pretext_loss,
            "fraction": frac,
            "config": self.config,
        }
        if self.uw is not None:
            payload["uncertainty_weights"] = self.uw.state_dict()
            payload["uncertainty_mode"] = self.uncertainty_mode
        if self.use_ib:
            payload["ib_modules"] = {
                "mlm": self.ib_mlm.state_dict(),
                "mem": self.ib_mem.state_dict(),
                "dual": self.ib_dual.state_dict(),
                "beta": self.ib_beta,
                "latent_dim": self.ib_latent_dim,
            }
        torch.save(payload, p)
        self._log(f"✓ fraction ckpt {tag}% epoch={epoch+1} loss={loss:.4f} → {p}")

    # ------------------------------------------------------------ sampling
    def _sample_sub(self, hg: SimpleHypergraph, seed: int) -> SimpleHypergraph:
        return sample_online_subhypergraph(
            hg,
            minibatch_config={
                "max_nodes": self.mb_max_nodes,
                "max_edges": self.mb_max_edges,
                "expansion_hops": self.mb_expansion_hops,
                "seed_edges_per_subhypergraph": self.mb_seed_edges_per_sub,
            },
            seed=seed,
        )

    # --------------------------------------------------------------- train
    def train(self):
        rng = torch.Generator(device="cpu").manual_seed(int(self.config["training"]["seed"]))
        for epoch in range(self.epochs):
            ep = {"L_mlm": 0.0, "L_mem": 0.0, "L_dual": 0.0, "L_pretext_total": 0.0, "total": 0.0}
            if self.uw is not None:
                ep["uw_mlm_w"] = 0.0; ep["uw_mem_w"] = 0.0; ep["uw_dual_w"] = 0.0
                ep["uw_logv_mlm"] = 0.0; ep["uw_logv_mem"] = 0.0; ep["uw_logv_dual"] = 0.0
            if self.use_ib:
                ep["L_kl_total"] = 0.0
                ep["L_kl_mlm"] = 0.0
                ep["L_kl_mem"] = 0.0
                ep["L_kl_dual"] = 0.0
            steps = 0
            t0 = time.perf_counter()

            for step in range(self.steps_per_epoch):
                g_idx = int(torch.randint(0, len(self.graphs), (1,), generator=rng).item())
                hg = self.graphs[g_idx]
                att = 0
                sub: SimpleHypergraph | None = None
                while True:
                    sub = self._sample_sub(hg, seed=epoch * 100000 + step * 17 + att)
                    if sub.num_nodes >= self.mb_min_nodes or att >= 4:
                        break
                    att += 1

                x = sub.x.to(self.device)
                if x.dim() == 1:
                    x = x.unsqueeze(-1)
                H = _to_sparse_incidence(sub).to(self.device)
                H_e = _edge_cardinalities(H)

                step_seed = epoch * 100000 + step
                self.optimizer.zero_grad(set_to_none=True)

                # AMP on for the outer scope, encoder internally disables when sparse
                ctx = (
                    torch.autocast(device_type=self.device.type, dtype=self.amp_dtype)
                    if self.use_amp else _NullCtx()
                )

                try:
                    with ctx:
                        # Anchor forward — share HCA overlap table w/ membership hard negatives
                        node_t, edge_t, hca_src, hca_dst, hca_sim = self.encoder(
                            x, H, edge_cardinalities=H_e
                        )
                        node_t = torch.nan_to_num(node_t, nan=0.0, posinf=1e4, neginf=-1e4)
                        edge_t = torch.nan_to_num(edge_t, nan=0.0, posinf=1e4, neginf=-1e4)

                        batch: PretextBatchV2 = build_pretext_batch(
                            x=x,
                            incidence=H,
                            edge_mlm_rate=self.pretext_edge_mlm_rate,
                            node_drop_p=self.pretext_ndrop,
                            edge_drop_p=self.pretext_edrop,
                            membership_num_negatives=self.pretext_num_neg,
                            membership_hard_prob=self.pretext_hard_prob,
                            hca_neighbor_table=(hca_src, hca_dst, hca_sim),
                            seed=step_seed + 1,
                        )

                        # Dual-view forward (re-use encoder, incidence augmented)
                        v1n, v1e, *_ = self.encoder(x, batch.incidence_view1)
                        v2n, v2e, *_ = self.encoder(x, batch.incidence_view2)
                        v1e = torch.nan_to_num(v1e, nan=0.0, posinf=1e4, neginf=-1e4)
                        v2e = torch.nan_to_num(v2e, nan=0.0, posinf=1e4, neginf=-1e4)

                        ib_kl_mlm = edge_t.new_tensor(0.0)
                        ib_kl_mem = edge_t.new_tensor(0.0)
                        ib_kl_dual = edge_t.new_tensor(0.0)
                        edge_t_mlm = edge_t
                        node_t_mem = node_t
                        edge_t_mem = edge_t
                        v1e_dual = v1e
                        v2e_dual = v2e
                        if self.use_ib:
                            edge_t_mlm, ib_kl_mlm = self.ib_mlm(edge_t)
                            node_t_mem, kl_mem_node = self.ib_mem(node_t)
                            edge_t_mem, kl_mem_edge = self.ib_mem(edge_t)
                            ib_kl_mem = kl_mem_node + kl_mem_edge
                            v1e_dual, kl_dual_1 = self.ib_dual(v1e)
                            v2e_dual, kl_dual_2 = self.ib_dual(v2e)
                            ib_kl_dual = 0.5 * (kl_dual_1 + kl_dual_2)

                        L1 = edge_mlm_loss(edge_emb=edge_t_mlm, batch=batch, head=self.edge_recon_head)
                        L2 = node_edge_membership_loss(
                            node_emb=node_t_mem,
                            edge_emb=edge_t_mem,
                            batch=batch,
                            head=self.membership_head,
                            tau=self.pretext_tau_mem,
                        )
                        Ea = min(v1e_dual.size(0), v2e_dual.size(0))
                        L3 = edge_dualview_contrast_loss(
                            edge_emb_view1=v1e_dual[:Ea],
                            edge_emb_view2=v2e_dual[:Ea],
                            projector=self.edge_contrast_proj,
                            tau=self.pretext_tau_dual,
                        )
                        if self.uw is not None:
                            pretext_total = self.uw([L1, L2, L3])
                        else:
                            a, b, c = self.fixed_w
                            pretext_total = a * L1 + b * L2 + c * L3
                        total = pretext_total
                        if self.use_ib:
                            ib_kl_total = ib_kl_mlm + ib_kl_mem + ib_kl_dual
                            total = total + self.ib_beta * ib_kl_total
                        else:
                            ib_kl_total = edge_t.new_tensor(0.0)
                    total.backward()
                except RuntimeError as exc:
                    msg = str(exc)
                    if any(k in msg for k in ("out of bounds", "CUBLAS", "index", "size mismatch", "sparse", "SparseCsr")):
                        self._log(f"[skip] ep{epoch+1}/s{step} {hg.dataset_name} n{sub.num_nodes} e{len(sub.hyperedges)} — {msg[:120]}")
                        self.optimizer.zero_grad(set_to_none=True)
                        continue
                    raise
                except Exception as exc:
                    self._log(f"[skip] ep{epoch+1}/s{step} {hg.dataset_name} — {type(exc).__name__}: {str(exc)[:120]}")
                    self.optimizer.zero_grad(set_to_none=True)
                    continue

                if self.grad_clip_norm > 0:
                    all_p = (
                        list(self.encoder.parameters())
                        + list(self.edge_recon_head.parameters())
                        + list(self.membership_head.parameters())
                        + list(self.edge_contrast_proj.parameters())
                    )
                    if self.uw is not None:
                        all_p += list(self.uw.parameters())
                    if self.use_ib:
                        all_p += (
                            list(self.ib_mlm.parameters())
                            + list(self.ib_mem.parameters())
                            + list(self.ib_dual.parameters())
                        )
                    nn.utils.clip_grad_norm_(all_p, self.grad_clip_norm)
                self.optimizer.step()

                ep["L_mlm"] += float(L1.item())
                ep["L_mem"] += float(L2.item())
                ep["L_dual"] += float(L3.item())
                ep["L_pretext_total"] += float(pretext_total.item())
                ep["total"] += float(total.item())
                if self.uw is not None:
                    ws = self.uw.task_weights()
                    ep["uw_mlm_w"] += ws[0]; ep["uw_mem_w"] += ws[1]; ep["uw_dual_w"] += ws[2]
                    logv = self.uw.log_sigma if hasattr(self.uw, "log_sigma") else self.uw.log_var
                    ep["uw_logv_mlm"] += float(logv[0].item())
                    ep["uw_logv_mem"] += float(logv[1].item())
                    ep["uw_logv_dual"] += float(logv[2].item())
                if self.use_ib:
                    ep["L_kl_total"] += float(ib_kl_total.item())
                    ep["L_kl_mlm"] += float(ib_kl_mlm.item())
                    ep["L_kl_mem"] += float(ib_kl_mem.item())
                    ep["L_kl_dual"] += float(ib_kl_dual.item())
                steps += 1
                step_row = {
                    "epoch": epoch, "step": step,
                    "L_mlm": float(L1.item()), "L_mem": float(L2.item()), "L_dual": float(L3.item()),
                    "L_pretext_total": float(pretext_total.item()),
                    "L_total": float(total.item()),
                }
                if self.uw is not None:
                    step_row.update({
                        "uw_mlm_w": float(ws[0]),
                        "uw_mem_w": float(ws[1]),
                        "uw_dual_w": float(ws[2]),
                        "uw_logv_mlm": float(logv[0].item()),
                        "uw_logv_mem": float(logv[1].item()),
                        "uw_logv_dual": float(logv[2].item()),
                    })
                if self.use_ib:
                    step_row.update({
                        "L_kl_total": float(ib_kl_total.item()),
                        "L_kl_mlm": float(ib_kl_mlm.item()),
                        "L_kl_mem": float(ib_kl_mem.item()),
                        "L_kl_dual": float(ib_kl_dual.item()),
                        "ib_beta": float(self.ib_beta),
                    })
                self.step_losses.append(step_row)

            if steps == 0:
                self._log(f"epoch {epoch+1}/{self.epochs}: 0 valid steps, skipping.")
                self.bad_epochs += 1
                if self.bad_epochs >= self.patience:
                    self._log(f"Early stop (all steps skipped for {self.patience} epochs)")
                    break
                continue
            for k in ep:
                ep[k] /= steps
            self.epoch_losses.append({"epoch": epoch, **ep})
            dt = time.perf_counter() - t0
            uw_msg = ""
            if self.uw is not None:
                logv = self.uw.log_sigma if hasattr(self.uw, "log_sigma") else self.uw.log_var
                uw_msg = (f" uw=[{ep['uw_mlm_w']:.3f},{ep['uw_mem_w']:.3f},{ep['uw_dual_w']:.3f}]"
                          f" logv={logv.detach().cpu().tolist()}")
            ib_msg = ""
            if self.use_ib:
                ib_msg = (f"  kl={ep['L_kl_total']:.4f}"
                          f" [mlm={ep['L_kl_mlm']:.4f},mem={ep['L_kl_mem']:.4f},dual={ep['L_kl_dual']:.4f}]"
                          f" beta={self.ib_beta:g}")
            self._log(
                f"ep {epoch+1}/{self.epochs}  total={ep['total']:.4f}"
                f"  pretext={ep['L_pretext_total']:.4f}"
                f"  mlm={ep['L_mlm']:.4f}  mem={ep['L_mem']:.4f}  dual={ep['L_dual']:.4f}"
                f"{uw_msg}{ib_msg}  ({dt:.1f}s, {steps} valid steps)"
            )

            if ep["total"] < self.best_loss:
                self.best_loss = ep["total"]
                self.best_epoch = epoch
                self.bad_epochs = 0
                self._save_best(epoch, ep["total"], ep["L_pretext_total"])
            else:
                self.bad_epochs += 1
            self._save_last(epoch)
            for frac, target_epoch in self.checkpoint_fraction_epochs.items():
                if frac not in self.saved_fraction_ckpts and (epoch + 1) >= target_epoch:
                    self._save_fraction_ckpt(epoch, ep["total"], frac, ep["L_pretext_total"])
                    self.saved_fraction_ckpts.add(frac)
            if self.bad_epochs >= self.patience:
                self._log(f"Early stop at ep {epoch+1}; best={self.best_epoch+1} total={self.best_loss:.4f}")
                break

        # -------------------------- summaries -------------------------------
        out = Path(self.config["training"]["output_dir"]) / "results"
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "pretrain_summary_v2.json", "w") as f:
            logv = None
            if self.uw is not None:
                logv_tensor = self.uw.log_sigma if hasattr(self.uw, "log_sigma") else self.uw.log_var
                logv = logv_tensor.detach().cpu().tolist()
            json.dump({
                "best_epoch": self.best_epoch,
                "best_total_loss": self.best_loss,
                "best_pretext_loss": min((row["L_pretext_total"] for row in self.epoch_losses), default=None),
                "epochs_run": len(self.epoch_losses),
                "final_losses": self.epoch_losses[-1] if self.epoch_losses else None,
                "uncertainty_mode": self.uncertainty_mode if self.uw is not None else "fixed",
                "uncertainty_final_logv": logv,
                "use_ib": self.use_ib,
                "ib_beta": self.ib_beta if self.use_ib else 0.0,
                "ib_latent_dim": self.ib_latent_dim if self.use_ib else None,
                "checkpoint_fractions": self.checkpoint_fractions,
            }, f, indent=2)
        if self.epoch_losses:
            p = Path(self.config["training"]["output_dir"]) / "logs" / "pretrain_epochs_v2.csv"
            with open(p, "w") as f:
                w = _csv.DictWriter(f, fieldnames=list(self.epoch_losses[0].keys()))
                w.writeheader(); w.writerows(self.epoch_losses)
        if self.step_losses:
            p = Path(self.config["training"]["output_dir"]) / "logs" / "pretrain_steps_v2.csv"
            with open(p, "w") as f:
                w = _csv.DictWriter(f, fieldnames=list(self.step_losses[0].keys()))
                w.writeheader(); w.writerows(self.step_losses)


class _NullCtx:
    def __enter__(self): return self
    def __exit__(self, *a): return False
