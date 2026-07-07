from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Set
import time

import torch

from models.encoder import UnifiedHypergraphEncoder
from models.heads import TaskHeads
from models.pretext_tasks import compute_pretraining_losses
from trainers.trainer_base import TrainerBase
from utils.dhg_datasets import load_domain_graphs
from utils.eval import write_loss_history
from utils.hypergraph import iter_graphs
from utils.minibatch_sampling import build_subhypergraph_pool, sample_subhypergraph_batch, should_use_subhypergraph_pool


class PretrainTrainer(TrainerBase):
    def __init__(self, config: Dict, drop_tasks: Set[str] | None = None):
        super().__init__(config, ensure_subdirs=("checkpoints", "logs", "results"))
        all_tasks = {"masked_node", "hyperedge_recon", "contrastive", "size_pred", "domain_align"}
        enabled = config.get("training", {}).get("enabled_tasks")
        configured_drop = config.get("training", {}).get("drop_tasks")
        if enabled is not None:
            enabled_set = {str(name) for name in enabled}
            self.drop_tasks = {name for name in all_tasks if name not in enabled_set}
        elif configured_drop is not None:
            self.drop_tasks = {str(name) for name in configured_drop}
        else:
            self.drop_tasks = drop_tasks or set()
        self.domains = load_domain_graphs(config, seed=int(config["training"]["seed"]))
        self.graphs = iter_graphs(self.domains)
        hidden_dim = int(config["model"]["hidden_dim"])
        self.encoder = UnifiedHypergraphEncoder(
            in_dim=int(config["model"]["input_dim"]),
            hidden_dim=hidden_dim,
            dropout=float(config["model"]["dropout"]),
            num_layers=int(config["model"]["num_layers"]),
            num_heads=int(config["model"]["num_heads"]),
            structure_pe_dim=int(config["model"].get("structure_pe_dim", config["model"].get("spectral_dim", 0))),
            num_domains=len(sorted(self.domains)) if self.domains else 1,
            domain_names=sorted(self.domains),
        ).to(self.device)
        self.heads = TaskHeads(
            hidden_dim=hidden_dim,
            input_dim=int(config["model"]["input_dim"]),
            num_domains=max(len(sorted(self.domains)), 1),
        ).to(self.device)
        parameters = list(self.encoder.parameters()) + list(self.heads.parameters())
        self.optimizer = torch.optim.AdamW(
            parameters,
            lr=float(config["training"]["lr"]),
            weight_decay=float(config["training"]["weight_decay"]),
        )
        self.minibatch_config = dict(config["training"].get("minibatch", {}))
        self.pool_cache = self._build_pool_cache()
        self.domain_sample_counts = {name: 0 for name in self.domains}
        self.training_domains = sorted(self.domains)
        self.training_datasets = sorted({graph.dataset_name for graph in self.graphs})
        self._log_startup_info()

    def _log(self, message: str) -> None:
        super()._log(f"[HyperFounder][Pretrain] {message}")

    def _format_domain_counts(self, counts: Dict[str, int]) -> str:
        if not counts:
            return "-"
        return ", ".join(f"{name}:{counts[name]}" for name in sorted(counts))

    def _describe_batch_graphs(self, batch_graphs: List) -> str:
        parts = []
        for hg in batch_graphs:
            parts.append(
                f"{hg.dataset_name}/{hg.domain}"
                f"(nodes={hg.num_nodes}, edges={len(hg.hyperedges)})"
            )
        return "; ".join(parts)

    def _log_startup_info(self) -> None:
        dataset_names = ", ".join(self.training_datasets) if self.training_datasets else "-"
        domain_names = ", ".join(self.training_domains) if self.training_domains else "-"
        pooled_graphs = ", ".join(sorted(self.pool_cache)) if self.pool_cache else "-"
        self._log(
            "Loaded datasets="
            f"[{dataset_names}] domains=[{domain_names}] "
            f"device={self.device} output_dir={self.output_dir}"
        )
        self._log(
            f"Prepared {len(self.graphs)} graphs across {len(self.training_domains)} domains; "
            f"subhypergraph_pool_graphs=[{pooled_graphs}]"
        )

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
        step_history: List[Dict[str, float]] = []
        epochs = int(self.config["training"]["epochs"])
        steps_per_epoch = int(self.config["training"].get("steps_per_epoch", max(len(self.graphs), 1)))
        base_seed = int(self.config["training"]["seed"])
        patience = int(self.config["training"].get("early_stopping", {}).get("patience", 50))
        log_interval_steps = int(
            self.config["training"].get("log_interval_steps", max(1, steps_per_epoch // 4))
        )
        log_step_history = bool(self.config["training"].get("log_step_history", False))
        best_loss = float("inf")
        best_epoch = 0
        bad_epochs = 0
        train_start = time.perf_counter()
        self._log(
            f"Training start: epochs={epochs}, steps_per_epoch={steps_per_epoch}, "
            f"log_interval_steps={log_interval_steps}, patience={patience}"
        )
        for epoch in range(1, epochs + 1):
            self.encoder.train()
            self.heads.train()
            epoch_start = time.perf_counter()
            epoch_domain_counts = {name: 0 for name in self.training_domains}
            epoch_losses = {
                "masked_node": 0.0,
                "hyperedge_recon": 0.0,
                "contrastive": 0.0,
                "size_pred": 0.0,
                "domain_align": 0.0,
                "total": 0.0,
            }
            domain_schedule = self._build_domain_schedule(epoch, steps_per_epoch)
            schedule_preview = " | ".join(
                ",".join(step_domains) if step_domains else "-"
                for step_domains in domain_schedule[: min(3, len(domain_schedule))]
            )
            self._log(
                f"Epoch {epoch}/{epochs} start: schedule_preview=[{schedule_preview}]"
            )
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
                self.encoder.train()
                self.heads.train()
                self.optimizer.zero_grad()
                batch_loss_dicts = []
                for hg in batch_graphs:
                    hg.metadata["domain_id"] = self.training_domains.index(hg.domain) if hg.domain in self.training_domains else 0
                    batch_loss_dicts.append(
                        compute_pretraining_losses(
                            encoder=self.encoder,
                            heads=self.heads,
                            hg=hg,
                            task_cache={},
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
                if log_step_history:
                    step_domain_counts = {name: 0 for name in self.training_domains}
                    for hg in batch_graphs:
                        if hg.domain in step_domain_counts:
                            step_domain_counts[hg.domain] += 1
                    step_history.append(
                        {
                            "epoch": float(epoch),
                            "step": float(step + 1),
                            "batch_size": float(len(batch_graphs)),
                            "avg_nodes": float(sum(hg.num_nodes for hg in batch_graphs) / max(len(batch_graphs), 1)),
                            "avg_edges": float(sum(len(hg.hyperedges) for hg in batch_graphs) / max(len(batch_graphs), 1)),
                            **{f"domain_{name}": float(step_domain_counts[name]) for name in self.training_domains},
                            **{f"loss_{key}": float(averaged_losses[key]) for key in epoch_losses},
                        }
                    )
                for key in epoch_losses:
                    epoch_losses[key] += averaged_losses[key]
                if (
                    step == 0
                    or (step + 1) % log_interval_steps == 0
                    or step + 1 == steps_per_epoch
                ):
                    self._log(
                        f"Epoch {epoch}/{epochs} step {step + 1}/{steps_per_epoch}: "
                        f"preferred_domains={','.join(step_domains) if step_domains else '-'} "
                        f"batch_size={len(batch_graphs)} total={averaged_losses['total']:.4f} "
                        f"masked_node={averaged_losses['masked_node']:.4f} "
                        f"hyperedge_recon={averaged_losses['hyperedge_recon']:.4f} "
                        f"contrastive={averaged_losses['contrastive']:.4f} "
                        f"size_pred={averaged_losses['size_pred']:.4f} "
                        f"domain_align={averaged_losses['domain_align']:.4f} "
                        f"batch_graphs=[{self._describe_batch_graphs(batch_graphs)}]"
                    )
            history.append(
                {
                    "epoch": float(epoch),
                    **{key: value / max(steps_per_epoch, 1) for key, value in epoch_losses.items()},
                    **{f"domain_{name}": float(epoch_domain_counts[name]) for name in self.training_domains},
                }
            )

            epoch_total = float(history[-1]["total"])
            epoch_time_sec = time.perf_counter() - epoch_start
            self._log(
                f"Epoch {epoch}/{epochs} done: total={epoch_total:.4f} "
                f"domain_samples=[{self._format_domain_counts(epoch_domain_counts)}] "
                f"best_total={min(best_loss, epoch_total):.4f} epoch_time_sec={epoch_time_sec:.2f}"
            )
            if epoch_total < best_loss:
                best_loss = epoch_total
                best_epoch = epoch
                bad_epochs = 0
                best_path = self._save_checkpoint("pretrain_best.pt")
                self._log(
                    f"New best checkpoint at epoch {epoch}: total={best_loss:.4f} path={best_path}"
                )
            else:
                bad_epochs += 1
                self._log(
                    f"No improvement at epoch {epoch}: bad_epochs={bad_epochs}/{patience}"
                )
                if bad_epochs >= patience:
                    self._log(f"Early stopping triggered at epoch {epoch}.")
                    break

        train_time_sec = time.perf_counter() - train_start
        checkpoint_path = self._save_checkpoint("pretrain_last.pt")
        write_loss_history(str(self.output_dir / "logs" / "pretrain_losses.csv"), history)
        step_history_path = None
        if log_step_history and step_history:
            step_history_path = str(self.output_dir / "logs" / "pretrain_step_losses.csv")
            write_loss_history(step_history_path, step_history)
        self._log(
            f"Training finished: last_checkpoint={checkpoint_path} "
            f"history_csv={self.output_dir / 'logs' / 'pretrain_losses.csv'} "
            f"best_epoch={best_epoch} best_total={best_loss:.4f} "
            f"train_time_sec={train_time_sec:.2f}"
        )
        return {
            "checkpoint_path": checkpoint_path,
            "loss_history_path": str(self.output_dir / "logs" / "pretrain_losses.csv"),
            "step_loss_history_path": step_history_path,
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
