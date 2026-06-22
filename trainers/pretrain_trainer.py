from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Set
import time

import torch

from models.encoder import UnifiedHypergraphEncoder
from models.heads import TaskHeads
from models.pretext_tasks import compute_pretraining_losses
from utils.clustering import (
    build_community_pseudo_labels,
    build_motif_pseudo_labels,
    build_node_pseudo_labels,
    refresh_cross_domain_prototypes,
)
from utils.common import ensure_dir
from utils.dhg_datasets import load_domain_graphs
from utils.eval import write_loss_history
from utils.hypergraph import iter_graphs
from utils.minibatch_sampling import build_subhypergraph_pool, sample_subhypergraph_batch, should_use_subhypergraph_pool


class PretrainTrainer:
    def __init__(self, config: Dict, drop_tasks: Set[str] | None = None):
        self.config = config
        self.drop_tasks = drop_tasks or set()
        self.device = torch.device(config["training"].get("device", "cpu"))
        self.domains = load_domain_graphs(config, seed=int(config["training"]["seed"]))
        self.graphs = iter_graphs(self.domains)
        hidden_dim = int(config["model"]["hidden_dim"])
        self.encoder = UnifiedHypergraphEncoder(
            in_dim=int(config["model"]["input_dim"]),
            hidden_dim=hidden_dim,
            dropout=float(config["model"]["dropout"]),
            num_layers=int(config["model"]["num_layers"]),
            num_heads=int(config["model"]["num_heads"]),
            spectral_dim=int(config["model"]["spectral_dim"]),
        ).to(self.device)
        self.heads = TaskHeads(
            hidden_dim=hidden_dim,
            node_classes=int(config["tasks"]["node_clusters"]),
            motif_classes=int(config["tasks"]["motif_clusters"]),
            community_classes=int(config["tasks"]["community_clusters"]),
            prototype_classes=int(config["tasks"]["prototype_clusters"]),
        ).to(self.device)
        parameters = list(self.encoder.parameters()) + list(self.heads.parameters())
        self.optimizer = torch.optim.AdamW(
            parameters,
            lr=float(config["training"]["lr"]),
            weight_decay=float(config["training"]["weight_decay"]),
        )
        self.output_dir = Path(config["training"]["output_dir"])
        ensure_dir(self.output_dir / "checkpoints")
        ensure_dir(self.output_dir / "logs")
        ensure_dir(self.output_dir / "results")
        self.minibatch_config = dict(config["training"].get("minibatch", {}))
        self.pool_cache = self._build_pool_cache()
        self.domain_sample_counts = {name: 0 for name in self.domains}
        self.training_domains = sorted(self.domains)
        self.training_datasets = sorted({graph.dataset_name for graph in self.graphs})

    def _build_pool_cache(self) -> Dict[str, List]:
        pool_cache: Dict[str, List] = {}
        base_seed = int(self.config["training"]["seed"])
        for graph_index, hg in enumerate(self.graphs):
            if not should_use_subhypergraph_pool(hg, self.minibatch_config):
                continue
            pool_cache[hg.name] = build_subhypergraph_pool(
                hg,
                minibatch_config=self.minibatch_config,
                seed=base_seed + graph_index * 1009,
            )
        return pool_cache

    def _build_domain_schedule(self, epoch: int, steps_per_epoch: int) -> List[List[str]]:
        if not self.training_domains:
            return [[] for _ in range(steps_per_epoch)]
        generator = torch.Generator().manual_seed(int(self.config["training"]["seed"]) + epoch * 313)
        domains_per_step = max(1, min(int(self.minibatch_config.get("domains_per_step", 2)), len(self.training_domains)))
        permutation = torch.randperm(len(self.training_domains), generator=generator).tolist()
        ordered_domains = [self.training_domains[index] for index in permutation]
        schedule: List[List[str]] = []
        cursor = 0
        for step in range(steps_per_epoch):
            if cursor + domains_per_step > len(ordered_domains):
                permutation = torch.randperm(len(self.training_domains), generator=generator).tolist()
                ordered_domains = [self.training_domains[index] for index in permutation]
                cursor = 0
            schedule.append(ordered_domains[cursor : cursor + domains_per_step])
            cursor += domains_per_step
        return schedule

    def _build_subhypergraph_task_cache(self, hg, seed: int) -> Dict:
        motif_budget = int(self.config["training"]["motif_budget"])
        node_clusters = int(self.config["tasks"]["node_clusters"])
        motif_clusters = int(self.config["tasks"]["motif_clusters"])
        community_clusters = int(self.config["tasks"]["community_clusters"])
        motifs, _, motif_labels = build_motif_pseudo_labels(
            hg,
            motif_budget=motif_budget,
            num_clusters=motif_clusters,
            seed=seed,
        )
        communities, _, _, community_node_labels = build_community_pseudo_labels(
            hg,
            num_clusters=community_clusters,
            seed=seed + 1000,
        )
        return {
            "node_labels": build_node_pseudo_labels(
                hg,
                num_clusters=node_clusters,
                seed=seed,
            ),
            "motifs": motifs,
            "communities": communities,
            "motif_labels": motif_labels,
            "community_node_labels": community_node_labels,
            "prototype_labels": torch.zeros((0,), dtype=torch.long),
        }

    @torch.no_grad()
    def _assign_batch_prototypes(self, batch_graphs: List, task_caches: List[Dict], epoch: int, step: int) -> None:
        self.encoder.eval()
        cross_embeddings: List[torch.Tensor] = []
        cross_counts: List[int] = []
        for hg, task_cache in zip(batch_graphs, task_caches):
            _, _, _, aux = self.encoder(
                hg,
                hg.x.to(self.device),
                motif_budget=int(self.config["training"]["motif_budget"]),
                motifs=task_cache["motifs"],
                communities=task_cache["communities"],
                motif_seed=epoch + step,
            )
            cross_emb = aux["cross_emb"]
            cross_counts.append(cross_emb.size(0))
            if cross_emb.numel():
                cross_embeddings.append(cross_emb)
        _, label_tensor = refresh_cross_domain_prototypes(
            cross_embeddings,
            num_clusters=int(self.config["tasks"]["prototype_clusters"]),
            seed=int(self.config["training"]["seed"]) + epoch * 997 + step,
        )
        offset = 0
        for task_cache, cross_count in zip(task_caches, cross_counts):
            if cross_count == 0:
                task_cache["prototype_labels"] = torch.zeros((0,), dtype=torch.long)
                continue
            task_cache["prototype_labels"] = label_tensor[offset : offset + cross_count].cpu()
            offset += cross_count

    def _save_checkpoint(self, filename: str) -> str:
        checkpoint_path = self.output_dir / "checkpoints" / filename
        torch.save(
            {
                "encoder": self.encoder.state_dict(),
                "heads": self.heads.state_dict(),
                "config": self.config,
            },
            checkpoint_path,
        )
        return str(checkpoint_path)

    def train(self) -> Dict:
        history: List[Dict[str, float]] = []
        epochs = int(self.config["training"]["epochs"])
        steps_per_epoch = int(self.config["training"].get("steps_per_epoch", max(len(self.graphs), 1)))
        base_seed = int(self.config["training"]["seed"])
        patience = int(self.config["training"].get("early_stopping", {}).get("patience", 50))
        best_loss = float("inf")
        best_epoch = 0
        bad_epochs = 0
        train_start = time.perf_counter()
        for epoch in range(1, epochs + 1):
            self.encoder.train()
            self.heads.train()
            epoch_domain_counts = {name: 0 for name in self.training_domains}
            epoch_losses = {
                "struct": 0.0,
                "node": 0.0,
                "edge": 0.0,
                "motif": 0.0,
                "community": 0.0,
                "global": 0.0,
                "cross": 0.0,
                "total": 0.0,
            }
            domain_schedule = self._build_domain_schedule(epoch, steps_per_epoch)
            for step, step_domains in enumerate(domain_schedule):
                batch_graphs = sample_subhypergraph_batch(
                    self.domains,
                    minibatch_config=self.minibatch_config,
                    pool_cache=self.pool_cache,
                    seed=base_seed + epoch * 10000 + step,
                    preferred_domains=step_domains,
                )
                if not batch_graphs:
                    continue
                for batch_index, hg in enumerate(batch_graphs):
                    hg.name = f"{hg.name}_e{epoch}_s{step}_b{batch_index}"
                    self.domain_sample_counts[hg.domain] += 1
                    epoch_domain_counts[hg.domain] += 1
                task_caches = [
                    self._build_subhypergraph_task_cache(hg, seed=base_seed + epoch * 10000 + step * 97 + batch_index)
                    for batch_index, hg in enumerate(batch_graphs)
                ]
                self._assign_batch_prototypes(batch_graphs, task_caches, epoch=epoch, step=step)
                self.encoder.train()
                self.heads.train()
                self.optimizer.zero_grad()
                batch_loss_dicts = []
                for hg, cache in zip(batch_graphs, task_caches):
                    batch_loss_dicts.append(
                        compute_pretraining_losses(
                            encoder=self.encoder,
                            heads=self.heads,
                            hg=hg,
                            task_cache=cache,
                            config=self.config,
                            device=self.device,
                            epoch=epoch,
                            drop_tasks=self.drop_tasks,
                        )
                    )
                total_loss = torch.stack([loss_dict["total"] for loss_dict in batch_loss_dicts]).mean()
                total_loss.backward()
                self.optimizer.step()
                averaged_losses = {
                    key: sum(float(loss_dict[key].detach().cpu().item()) for loss_dict in batch_loss_dicts)
                    / max(len(batch_loss_dicts), 1)
                    for key in epoch_losses
                }
                for key in epoch_losses:
                    epoch_losses[key] += averaged_losses[key]
            history.append(
                {
                    "epoch": float(epoch),
                    **{key: value / max(steps_per_epoch, 1) for key, value in epoch_losses.items()},
                    **{f"domain_{name}": float(epoch_domain_counts[name]) for name in self.training_domains},
                }
            )

            epoch_total = float(history[-1]["total"])
            if epoch_total < best_loss:
                best_loss = epoch_total
                best_epoch = epoch
                bad_epochs = 0
                self._save_checkpoint("pretrain_best.pt")
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    break

        train_time_sec = time.perf_counter() - train_start
        checkpoint_path = self._save_checkpoint("pretrain_last.pt")
        write_loss_history(str(self.output_dir / "logs" / "pretrain_losses.csv"), history)
        return {
            "checkpoint_path": checkpoint_path,
            "loss_history_path": str(self.output_dir / "logs" / "pretrain_losses.csv"),
            "cross_domain_pretraining": len(self.training_domains) > 1,
            "training_domains": self.training_domains,
            "training_datasets": self.training_datasets,
            "num_domains": len(self.training_domains),
            "domain_sample_counts": self.domain_sample_counts,
            "sampling_mode": "hyperedge_centered_subhypergraph_minibatch",
            "domain_batch_policy": "balanced_round_robin",
            "pooled_graphs": sorted(self.pool_cache),
            "early_stopping_patience": int(patience),
            "best_epoch": int(best_epoch),
            "best_total_loss": float(best_loss) if best_epoch else None,
            "pretrain_train_time_sec": float(train_time_sec),
        }
