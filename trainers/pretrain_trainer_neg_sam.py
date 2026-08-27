from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Set
import time

import torch
try:
    from torch.amp import GradScaler
except ImportError:
    from torch.cuda.amp import GradScaler
from tqdm.auto import tqdm

from models.encoder import UnifiedHypergraphEncoder
from models.heads_neg_sam import TaskHeadsNegSam
from models.pretext_tasks_neg_sam import compute_pretraining_losses
from trainers.trainer_base import TrainerBase
from utils.dhg_datasets import load_domain_graphs
from utils.eval import write_loss_history
from utils.hypergraph import iter_graphs
from utils.minibatch_sampling import build_subhypergraph_pool, sample_subhypergraph_batch, sample_subhypergraph_batch_with_quality, should_use_subhypergraph_pool
from utils.negative_bank import HardNegativeBank, QualityAwareNegativeSampler


class PretrainTrainerNegSam(TrainerBase):
    def __init__(self, config: Dict, drop_tasks: Set[str] | None = None):
        super().__init__(config, ensure_subdirs=("checkpoints", "logs", "results"))
        all_tasks = {
            "masked_node", "hyperedge_recon", "contrastive", "size_pred",
            "domain_align", "membership_contrast", "motif", "community",
            "structure_align", "structure_discrimination",
            "orth_node", "orth_edge", "private_domain_node", "private_domain_edge",
        }
        enabled = config.get("training", {}).get("enabled_tasks")
        configured_drop = config.get("training", {}).get("drop_tasks")
        if enabled is not None:
            enabled_set = {str(name) for name in enabled}
            self.drop_tasks = {name for name in all_tasks if name not in enabled_set}
        elif configured_drop is not None:
            self.drop_tasks = {str(name) for name in configured_drop}
        else:
            self.drop_tasks = drop_tasks or set()
        self.enabled_tasks = sorted(all_tasks.difference(self.drop_tasks))
        progress_config = dict(config.get("training", {}).get("progress", {}))
        self.show_epoch_bar = bool(progress_config.get("show_epoch_bar", True))
        self.show_step_bar = bool(progress_config.get("show_step_bar", True))
        self.show_pool_bar = bool(progress_config.get("show_pool_bar", True))
        self.leave_progress_bar = bool(progress_config.get("leave_progress_bar", False))
        self.progress_mininterval_sec = float(progress_config.get("mininterval_sec", 2.0))
        self._log("Stage 1/4: loading domain graphs")
        self.domains = load_domain_graphs(config, seed=int(config["training"]["seed"]))
        self.graphs = iter_graphs(self.domains)
        hidden_dim = int(config["model"]["hidden_dim"])
        
        # Domain adapter settings
        use_domain_adapter = bool(config["model"].get("use_domain_adapter", True))
        adapter_type = str(config["model"].get("adapter_type", "adapter"))
        adapter_dim = int(config["model"].get("adapter_dim", 32))
        num_experts = int(config["model"].get("num_experts", 4))
        
        self._log("Stage 2/4: building encoder and negative-sampling task heads")
        self.encoder = UnifiedHypergraphEncoder(
            in_dim=int(config["model"]["input_dim"]),
            hidden_dim=hidden_dim,
            dropout=float(config["model"]["dropout"]),
            num_layers=int(config["model"]["num_layers"]),
            num_heads=int(config["model"]["num_heads"]),
            structure_pe_dim=int(config["model"].get("structure_pe_dim", config["model"].get("spectral_dim", 0))),
            num_domains=len(sorted(self.domains)) if self.domains else 1,
            domain_names=sorted(self.domains),
            max_k=int(config["model"].get("max_k", 512)),
            use_domain_adapter=use_domain_adapter,
            adapter_type=adapter_type,
            adapter_dim=adapter_dim,
            num_experts=num_experts,
        ).to(self.device)
        self.training_domains = sorted(self.domains)
        self._pre_register_domain_projectors()
        
        # Task head settings
        num_motif_types = int(config["training"].get("num_motif_types", 8))
        num_prototypes = int(config["training"].get("num_prototypes", 8))
        
        # Challenge 1: Disentanglement & Alignment settings
        shared_dim = int(config["training"].get("shared_dim", hidden_dim))
        private_dim = int(config["training"].get("private_dim", hidden_dim))
        lambda_orth = float(config["training"].get("lambda_orth", 0.02))
        lambda_private_domain = float(config["training"].get("lambda_private_domain", 0.05))
        lambda_align = float(config["training"].get("lambda_align", 0.1))
        use_confidence_routing = bool(config["training"].get("use_confidence_routing", True))
        use_node_alignment = bool(config["training"].get("use_node_alignment", True))
        use_edge_alignment = bool(config["training"].get("use_edge_alignment", True))
        tau_node_align = float(config["training"].get("tau_node_align", 0.6))
        tau_edge_align = float(config["training"].get("tau_edge_align", 0.65))
        
        self.heads = TaskHeadsNegSam(
            hidden_dim=hidden_dim,
            input_dim=int(config["model"]["input_dim"]),
            num_domains=max(len(self.training_domains), 1),
            num_motif_types=num_motif_types,
            num_prototypes=num_prototypes,
            # Challenge 1 settings
            shared_dim=shared_dim,
            private_dim=private_dim,
            lambda_orth=lambda_orth,
            lambda_private_domain=lambda_private_domain,
            lambda_align=lambda_align,
            use_confidence_routing=use_confidence_routing,
            use_node_alignment=use_node_alignment,
            use_edge_alignment=use_edge_alignment,
            tau_node_align=tau_node_align,
            tau_edge_align=tau_edge_align,
        ).to(self.device)
        
        # Log actual device info
        encoder_device = next(self.encoder.parameters()).device
        heads_device = next(self.heads.parameters()).device
        self._log(f"Model devices: encoder={encoder_device}, heads={heads_device}")
        if torch.cuda.is_available():
            self._log(f"CUDA info: device_count={torch.cuda.device_count()}, current_device={torch.cuda.current_device()}, device_name={torch.cuda.get_device_name()}")
        
        parameters = list(self.encoder.parameters()) + list(self.heads.parameters())
        self.optimizer = torch.optim.AdamW(
            parameters,
            lr=float(config["training"]["lr"]),
            weight_decay=float(config["training"]["weight_decay"]),
        )
        # Mixed precision: prefer bf16 (stable, no GradScaler needed) over fp16.
        # When CUDA is unavailable we disable AMP entirely (CPU autocast is slow).
        amp_cfg = config.get("training", {}).get("amp", {})
        cuda_available = torch.cuda.is_available()
        self.use_amp = bool(amp_cfg.get("enabled", True)) and cuda_available
        amp_dtype = str(amp_cfg.get("dtype", "bf16")).lower()
        self.amp_dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
        # GradScaler is only meaningful for fp16; bf16 has sufficient
        # dynamic range that scaling is unnecessary (Kendall et al.).
        self.scaler = GradScaler(
            "cuda",
            enabled=self.use_amp and self.amp_dtype is torch.float16,
        )
        if self.use_amp:
            self._log(
                f"Mixed precision enabled: dtype={'bf16' if self.amp_dtype is torch.bfloat16 else 'fp16'} "
                f"(GradScaler enabled={self.scaler.is_enabled()})"
            )
        self.grad_clip_norm = float(config["training"].get("grad_clip_norm", 1.0))
        self.minibatch_config = dict(config.get("training", {}).get("minibatch", {}))
        self._log("Stage 3/4: building subhypergraph pools")
        self.pool_cache = self._build_pool_cache()
        self.domain_sample_counts = {name: 0 for name in self.domains}
        self.training_datasets = sorted({graph.dataset_name for graph in self.graphs})
        
        # Challenge 2: Hard Negative Bank (now with HEDG-similarity-aware
        # entry scoring + optional replay of historically hard subgraphs.)
        training_cfg = config.get("training", {})
        self.use_hard_negative_bank = bool(training_cfg.get("use_hard_negative_bank", True))
        self.log_quality_stats = bool(training_cfg.get("log_quality_stats", True))
        if self.use_hard_negative_bank:
            self.hard_negative_bank = HardNegativeBank(
                max_size=int(training_cfg.get("hard_negative_bank_size", 1000)),
                num_tiers=int(training_cfg.get("hard_negative_num_tiers", 3)),
                num_domains=len(self.training_domains),
                sampling_strategy=str(training_cfg.get("hard_negative_sampling_strategy", "quality_weighted")),
            )
            self.hard_negative_replay_enabled = bool(
                training_cfg.get("hard_negative_replay_enabled", True)
            )
            self.hard_negative_replay_interval = int(
                training_cfg.get("hard_negative_replay_interval", 5)
            )
            self.hard_negative_replay_fraction = float(
                training_cfg.get("hard_negative_replay_fraction", 0.2)
            )
            self._log(
                f"Initialized HardNegativeBank: size={self.hard_negative_bank.max_size} "
                f"replay={'on' if self.hard_negative_replay_enabled else 'off'} "
                f"(every {self.hard_negative_replay_interval} steps, "
                f"{int(self.hard_negative_replay_fraction * 100)}% batch)"
            )
        else:
            self.hard_negative_bank = None
            self.hard_negative_replay_enabled = False
            self.hard_negative_replay_interval = 0
            self.hard_negative_replay_fraction = 0.0
        
        self._log_startup_info()

    def _log(self, message: str) -> None:
        super()._log(f"[HyperFounder][PretrainNegSam] {message}")

    def _format_domain_counts(self, counts: Dict[str, int]) -> str:
        if not counts:
            return "-"
        return ", ".join(f"{name}:{counts[name]}" for name in sorted(counts))

    def _pre_register_domain_projectors(self) -> None:
        for domain_name in self.training_domains:
            domain_graphs = self.domains.get(domain_name, [])
            if not domain_graphs:
                continue
            sample_graph = domain_graphs[0]
            domain_id = self.training_domains.index(domain_name)
            feature_type = str(sample_graph.metadata.get("feature_type", "numerical"))
            feature_dim = int(sample_graph.x.size(-1))
            self.encoder.projector.register_domain(
                domain_id=domain_id,
                node_dim=feature_dim,
                edge_dim=feature_dim,
                feature_type=feature_type,
            )

    def _describe_batch_graphs(self, batch_graphs: List) -> str:
        parts = []
        for hg in batch_graphs:
            parts.append(f"{hg.dataset_name}/{hg.domain}(nodes={hg.num_nodes}, edges={len(hg.hyperedges)})")
        return "; ".join(parts)

    def _aggregate_losses(self, batch_loss_dicts: List[Dict]) -> Dict[str, float]:
        """Aggregate losses across a batch of graphs."""
        if not batch_loss_dicts:
            return {"total": 0.0}
        # Sum all losses
        aggregated = {}
        for loss_dict in batch_loss_dicts:
            for key, value in loss_dict.items():
                if key == "stats":
                    continue  # Skip stats, handle separately
                if isinstance(value, torch.Tensor):
                    if value.numel() == 1:
                        value = value.item()
                    else:
                        value = value.mean().item()
                if isinstance(value, (int, float)):
                    aggregated[key] = aggregated.get(key, 0.0) + float(value)
        # Average by batch size
        n = len(batch_loss_dicts)
        for key in aggregated:
            aggregated[key] /= n
        # Compute total loss
        loss_keys = [k for k in aggregated.keys() if k != "total"]
        aggregated["total"] = sum(aggregated.get(k, 0.0) for k in loss_keys)
        return aggregated

    def _aggregate_stats(
        self,
        batch_stats: List[Dict],
        batch_quality_stats: List[Dict],
    ) -> Dict[str, float]:
        """Aggregate stats across a batch of graphs."""
        aggregated = {}
        # Aggregate from batch_stats
        for stats in batch_stats:
            for key, value in stats.items():
                if isinstance(value, (int, float)):
                    aggregated[key] = aggregated.get(key, 0.0) + value
        # Aggregate from batch_quality_stats
        for stats in batch_quality_stats:
            for key, value in stats.items():
                if isinstance(value, (int, float)):
                    aggregated[key] = aggregated.get(key, 0.0) + value
        # Average by batch size
        n = len(batch_stats) if batch_stats else 1
        for key in aggregated:
            aggregated[key] /= n
        return aggregated

    def _log_startup_info(self) -> None:
        dataset_names = ", ".join(self.training_datasets) if self.training_datasets else "-"
        domain_names = ", ".join(self.training_domains) if self.training_domains else "-"
        pooled_graphs = ", ".join(sorted(self.pool_cache)) if self.pool_cache else "-"
        loss_weights = self.config.get("training", {}).get("loss_weights", {})
        loss_weight_text = ", ".join(
            f"{task}:{float(loss_weights.get(task, 1.0)):.3f}" for task in self.enabled_tasks
        ) if self.enabled_tasks else "-"
        self._log(
            "Loaded datasets="
            f"[{dataset_names}] domains=[{domain_names}] "
            f"device={self.device} output_dir={self.output_dir}"
        )
        self._log(
            f"Prepared {len(self.graphs)} graphs across {len(self.training_domains)} domains; "
            f"subhypergraph_pool_graphs=[{pooled_graphs}]"
        )
        self._log(
            f"Stage 4/4: ready for training; enabled_tasks=[{', '.join(self.enabled_tasks)}] "
            f"loss_weights=[{loss_weight_text}] log_step_history={bool(self.config['training'].get('log_step_history', False))}"
        )

    def _build_pool_cache(self) -> Dict[str, List]:
        pool_cache: Dict[str, List] = {}
        base_seed = int(self.config["training"]["seed"])
        pool_targets = [
            (graph_index, hg)
            for graph_index, hg in enumerate(self.graphs)
            if should_use_subhypergraph_pool(hg, self.minibatch_config)
        ]
        if not pool_targets:
            self._log("No large graphs require subhypergraph pool precomputation.")
            return pool_cache
        iterator = pool_targets
        pool_bar = None
        if self.show_pool_bar:
            pool_bar = tqdm(
                pool_targets,
                total=len(pool_targets),
                desc="Pool build",
                ascii=True,
                leave=self.leave_progress_bar,
                mininterval=self.progress_mininterval_sec,
            )
            iterator = pool_bar
        for graph_index, hg in iterator:
            if not should_use_subhypergraph_pool(hg, self.minibatch_config):
                continue
            pool_cache[hg.name] = build_subhypergraph_pool(
                hg,
                minibatch_config=self.minibatch_config,
                seed=base_seed + graph_index * 1009,
            )
            if pool_bar is not None:
                pool_bar.set_postfix(graph=hg.dataset_name, pool=len(pool_cache[hg.name]), refresh=False)
        if pool_bar is not None:
            pool_bar.close()
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
        for _ in range(steps_per_epoch):
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
        log_interval_steps = int(self.config["training"].get("log_interval_steps", max(1, steps_per_epoch // 4)))
        log_step_history = bool(self.config["training"].get("log_step_history", False))
        best_loss = float("inf")
        best_epoch = 0
        bad_epochs = 0
        train_start = time.perf_counter()
        self._log(
            f"Training start: epochs={epochs}, steps_per_epoch={steps_per_epoch}, "
            f"log_interval_steps={log_interval_steps}, patience={patience}"
        )
        epoch_iterator = range(1, epochs + 1)
        epoch_bar = None
        if self.show_epoch_bar:
            epoch_bar = tqdm(
                epoch_iterator,
                total=epochs,
                desc="Pretrain epochs",
                ascii=True,
                leave=self.leave_progress_bar,
                mininterval=self.progress_mininterval_sec,
            )
            epoch_iterator = epoch_bar
        for epoch in epoch_iterator:
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
                "membership_contrast": 0.0,
                "motif": 0.0,
                "community": 0.0,
                "structure_align": 0.0,
                "structure_discrimination": 0.0,
                "orth_node": 0.0,
                "orth_edge": 0.0,
                "private_domain_node": 0.0,
                "private_domain_edge": 0.0,
                "total": 0.0,
            }
            epoch_stats = {
                # Negative sampling stats
                "num_hyperedge_negatives": 0.0,
                "num_membership_negatives": 0.0,
                "num_subgraph_negatives": 0.0,
                "hyperedge_negative_rejects": 0.0,
                "membership_false_negative_rejects": 0.0,
                "avg_negative_overlap": 0.0,
                "avg_membership_hop": 0.0,
                "avg_subgraph_strength_pos": 0.0,
                "avg_subgraph_strength_neg": 0.0,
                # Challenge 2: Quality stats
                "num_valid_samples": 0.0,
                "num_invalid_samples": 0.0,
                "num_hard_negatives": 0.0,
                "avg_quality_score": 0.0,
                "num_membership_routed": 0.0,
                "num_hyperedge_routed": 0.0,
                "num_motif_routed": 0.0,
                "num_hard_negative_bank": 0.0,
            }
            domain_schedule = self._build_domain_schedule(epoch, steps_per_epoch)
            schedule_preview = " | ".join(
                ",".join(step_domains) if step_domains else "-"
                for step_domains in domain_schedule[: min(3, len(domain_schedule))]
            )
            self._log(f"Epoch {epoch}/{epochs} start: schedule_preview=[{schedule_preview}]")
            step_bar = None
            if self.show_step_bar:
                step_bar = tqdm(
                    total=steps_per_epoch,
                    desc=f"Epoch {epoch}/{epochs}",
                    ascii=True,
                    leave=self.leave_progress_bar,
                    mininterval=self.progress_mininterval_sec,
                )
            for step, step_domains in enumerate(domain_schedule):
                # Use quality-aware batch sampling
                batch_items = sample_subhypergraph_batch_with_quality(
                    domains=self.domains,
                    minibatch_config=self.minibatch_config,
                    pool_cache=self.pool_cache,
                    seed=base_seed + epoch * 10000 + step,
                    preferred_domains=step_domains,
                )
                if not batch_items:
                    continue
                
                # Separate graphs and quality metadata
                batch_graphs = [item["subhypergraph"] for item in batch_items]
                batch_qualities = [item["quality_meta"] for item in batch_items]
                
                if not batch_graphs:
                    continue
                for batch_index, hg in enumerate(batch_graphs):
                    hg.name = f"{hg.name}_e{epoch}_s{step}_b{batch_index}"
                    self.domain_sample_counts[hg.domain] += 1
                    epoch_domain_counts[hg.domain] += 1
                self.optimizer.zero_grad()
                batch_loss_dicts = []
                batch_stats = []
                batch_quality_stats = []
                
                # Compute per-graph losses; KEEP computation graph for backward.
                per_graph_total = []
                for batch_index, hg in enumerate(batch_graphs):
                    hg.metadata["domain_id"] = self.training_domains.index(hg.domain) if hg.domain in self.training_domains else 0
                    loss_dict = compute_pretraining_losses(
                        encoder=self.encoder,
                        heads=self.heads,
                        hg=hg,
                        task_cache={},
                        config=self.config,
                        device=self.device,
                        epoch=epoch,
                        drop_tasks=self.drop_tasks,
                        amp_enabled=self.use_amp,
                        amp_dtype=self.amp_dtype,
                    )
                    # Stats are floats/dicts - safe to detach.
                    batch_stats.append({k: (float(v) if hasattr(v, "item") else v) for k, v in loss_dict.get("stats", {}).items()})
                    # Loss tensors must NOT be detached yet - keep them for backward().
                    batch_loss_dicts.append({key: value for key, value in loss_dict.items() if key != "stats"})
                    per_graph_total.append(loss_dict["total"])
                    
                    # Collect quality stats
                    quality_meta = batch_qualities[batch_index]
                    routing = quality_meta.routing
                    quality_stats = {
                        "num_valid_samples": 1.0 if routing["valid"] else 0.0,
                        "num_invalid_samples": 1.0 if not routing["valid"] else 0.0,
                        "num_hard_negatives": 1.0 if routing["hard_negative"] else 0.0,
                        "avg_quality_score": quality_meta.quality_score,
                        "num_membership_routed": 1.0 if routing["membership"] else 0.0,
                        "num_hyperedge_routed": 1.0 if routing["hyperedge_recon"] else 0.0,
                        "num_motif_routed": 1.0 if routing["motif"] else 0.0,
                        "num_hard_negative_bank": 1.0 if routing["hard_negative"] else 0.0,
                    }
                    batch_quality_stats.append(quality_stats)
                    
                    # Add to hard negative bank if applicable.
                    # HEDG-level metadata (avg similarity, fallback rate, ...)
                    # is bundled together with the structural quality score
                    # so the quality_weighted sampler naturally favours
                    # subgraphs that historically produced the hardest, most
                    # structurally similar negatives.
                    if self.hard_negative_bank is not None and routing["hard_negative"]:
                        per_graph_stats = batch_stats[-1] if batch_stats else {}
                        n_hard = float(per_graph_stats.get("n_hard_used", 0.0))
                        n_fb = float(per_graph_stats.get("n_random_fallback", 0.0))
                        fallback_rate = (n_fb / max(n_hard + n_fb, 1e-6)) if (n_hard + n_fb) > 0 else 0.0
                        hedg_meta = {
                            "avg_similarity": float(per_graph_stats.get("avg_neg_hedg_similarity", 0.0)),
                            "fallback_rate": fallback_rate,
                            "n_hard_used": n_hard,
                            "n_random_fallback": n_fb,
                        }
                        self.hard_negative_bank.add(
                            subhypergraph_data={"name": hg.name, "domain": hg.domain},
                            quality_score=quality_meta.quality_score,
                            domain_id=quality_meta.domain_id,
                            hedg_meta=hedg_meta,
                        )

                # === Optional replay: re-forward on historically-hard subgraphs. ===
                # Pull replay_fraction * batch_size entries from the bank, find
                # the matching pooled subhypergraphs by name in pool_cache, and
                # append their loss to step_loss. Missing entries are skipped.
                replay_loss_acc = None
                replay_count = 0
                if (
                    self.hard_negative_bank is not None
                    and self.hard_negative_replay_enabled
                    and (step + 1) % self.hard_negative_replay_interval == 0
                ):
                    replay_target = max(1, int(self.hard_negative_replay_fraction * len(batch_graphs)))
                    bank_entries = self.hard_negative_bank.sample(replay_target)
                    replay_hgs: List = []
                    for entry in bank_entries:
                        data = entry.get("data", {})
                        name = data.get("name")
                        domain = data.get("domain")
                        if not name or not domain:
                            continue
                        pool_list = self.pool_cache.get(name)
                        if pool_list:
                            replay_hgs.append(pool_list[0])
                    if replay_hgs:
                        replay_totals = []
                        for r_hg in replay_hgs:
                            r_hg.metadata["domain_id"] = (
                                self.training_domains.index(r_hg.domain)
                                if r_hg.domain in self.training_domains else 0
                            )
                            r_loss = compute_pretraining_losses(
                                encoder=self.encoder,
                                heads=self.heads,
                                hg=r_hg,
                                task_cache={},
                                config=self.config,
                                device=self.device,
                                epoch=epoch,
                                drop_tasks=self.drop_tasks,
                                amp_enabled=self.use_amp,
                                amp_dtype=self.amp_dtype,
                            )
                            r_t = r_loss.get("total")
                            if isinstance(r_t, torch.Tensor) and r_t.numel() == 1:
                                replay_totals.append(r_t)
                        if replay_totals:
                            replay_loss_acc = torch.stack(replay_totals).mean()
                            replay_count = len(replay_totals)

                # === CRITICAL FIX: actually backprop and step the optimizer. ===
                # Without this the model is never updated and losses never decrease.
                if per_graph_total:
                    step_loss = torch.stack(per_graph_total).mean()
                    if replay_loss_acc is not None:
                        step_loss = step_loss + self.hard_negative_replay_fraction * replay_loss_acc
                    if self.use_amp and self.scaler is not None:
                        self.scaler.scale(step_loss).backward()
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            list(self.encoder.parameters()) + list(self.heads.parameters()),
                            max_norm=self.grad_clip_norm,
                        )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        step_loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            list(self.encoder.parameters()) + list(self.heads.parameters()),
                            max_norm=self.grad_clip_norm,
                        )
                        self.optimizer.step()
                # Detach per-graph losses for logging (do NOT touch in-graph tensors).
                # Only scalar tensors (numel == 1) are loss values; multi-element
                # entries would be metadata and must be skipped.
                detached_batch_losses = []
                for loss_dict in batch_loss_dicts:
                    detached = {}
                    for k, v in loss_dict.items():
                        if hasattr(v, "detach"):
                            t = v.detach()
                            if t.numel() != 1:
                                continue
                            detached[k] = float(t.item())
                        else:
                            detached[k] = float(v)
                    detached_batch_losses.append(detached)
                # Compute averaged losses and stats after processing all graphs in batch
                averaged_losses = self._aggregate_losses(detached_batch_losses)
                averaged_stats = self._aggregate_stats(batch_stats, batch_quality_stats)

                if step_bar is not None:
                    step_bar.update(1)
                    step_bar.set_postfix(
                        loss=f"{averaged_losses.get('total', 0.0):.4f}",
                        neg=f"{averaged_stats.get('num_hyperedge_negatives', 0.0):.1f}/{averaged_stats.get('num_membership_negatives', 0.0):.1f}",
                        refresh=False,
                    )
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
                            **{f"loss_{key}": float(averaged_losses.get(key, 0.0)) for key in epoch_losses},
                            **{f"stat_{key}": float(averaged_stats.get(key, 0.0)) for key in epoch_stats},
                        }
                    )
                for key in epoch_losses:
                    epoch_losses[key] += averaged_losses.get(key, 0.0)
                for key in epoch_stats:
                    epoch_stats[key] += averaged_stats.get(key, 0.0)
                if step == 0 or (step + 1) % log_interval_steps == 0 or step + 1 == steps_per_epoch:
                    extra_log_parts: List[str] = []
                    if replay_count > 0:
                        extra_log_parts.append(f"replay_n={replay_count}")
                    if self.log_quality_stats and self.hard_negative_bank is not None:
                        bstats = self.hard_negative_bank.get_stats()
                        extra_log_parts.append(
                            f"bank_size={int(bstats['total_size'])}/{bstats['max_size']:.0f} "
                            f"bank_q={bstats['avg_quality']:.3f} "
                            f"bank_hardness={bstats.get('avg_hardness_proxy', 0.0):.3f} "
                            f"bank_hedg_sim={bstats.get('avg_hedg_similarity', 0.0):.2f}"
                        )
                    extra = (" | " + " | ".join(extra_log_parts)) if extra_log_parts else ""
                    self._log(
                        f"Epoch {epoch}/{epochs} step {step + 1}/{steps_per_epoch}: "
                        f"preferred_domains={','.join(step_domains) if step_domains else '-'} "
                        f"batch_size={len(batch_graphs)} total={averaged_losses.get('total', 0.0):.4f} "
                        f"masked_node={averaged_losses.get('masked_node', 0.0):.4f} "
                        f"hyperedge_recon={averaged_losses.get('hyperedge_recon', 0.0):.4f} "
                        f"contrastive={averaged_losses.get('contrastive', 0.0):.4f} "
                        f"size_pred={averaged_losses.get('size_pred', 0.0):.4f} "
                        f"domain_align={averaged_losses.get('domain_align', 0.0):.4f} "
                        f"membership_contrast={averaged_losses.get('membership_contrast', 0.0):.4f} "
                        f"motif={averaged_losses.get('motif', 0.0):.4f} "
                        f"community={averaged_losses.get('community', 0.0):.4f} "
                        f"structure_align={averaged_losses.get('structure_align', 0.0):.4f} "
                        f"neg_hyperedges={averaged_stats.get('num_hyperedge_negatives', 0.0):.1f} "
                        f"neg_memberships={averaged_stats.get('num_membership_negatives', 0.0):.1f} "
                        f"avg_neg_overlap={averaged_stats.get('avg_negative_overlap', 0.0):.2f} "
                        f"avg_membership_hop={averaged_stats.get('avg_membership_hop', 0.0):.2f}"
                        f"{extra} "
                        f"batch_graphs=[{self._describe_batch_graphs(batch_graphs)}]"
                    )
            if step_bar is not None:
                step_bar.close()
            history.append(
                {
                    "epoch": float(epoch),
                    **{key: value / max(steps_per_epoch, 1) for key, value in epoch_losses.items()},
                    **{f"domain_{name}": float(epoch_domain_counts[name]) for name in self.training_domains},
                    **{f"stat_{key}": float(epoch_stats[key] / max(steps_per_epoch, 1)) for key in epoch_stats},
                }
            )

            epoch_total = float(history[-1]["total"])
            epoch_time_sec = time.perf_counter() - epoch_start
            self._log(
                f"Epoch {epoch}/{epochs} done: total={epoch_total:.4f} "
                f"domain_samples=[{self._format_domain_counts(epoch_domain_counts)}] "
                f"best_total={min(best_loss, epoch_total):.4f} epoch_time_sec={epoch_time_sec:.2f}"
            )
            if epoch_bar is not None:
                epoch_bar.set_postfix(total=f"{epoch_total:.4f}", best=f"{min(best_loss, epoch_total):.4f}", refresh=False)
            if epoch_total < best_loss:
                best_loss = epoch_total
                best_epoch = epoch
                bad_epochs = 0
                best_path = self._save_checkpoint("pretrain_best_neg_sam.pt")
                self._log(f"New best checkpoint at epoch {epoch}: total={best_loss:.4f} path={best_path}")
            else:
                bad_epochs += 1
                self._log(f"No improvement at epoch {epoch}: bad_epochs={bad_epochs}/{patience}")
                if bad_epochs >= patience:
                    self._log(f"Early stopping triggered at epoch {epoch}.")
                    break
        if epoch_bar is not None:
            epoch_bar.close()

        train_time_sec = time.perf_counter() - train_start
        checkpoint_path = self._save_checkpoint("pretrain_last_neg_sam.pt")
        loss_history_path = str(self.output_dir / "logs" / "pretrain_losses_neg_sam.csv")
        write_loss_history(loss_history_path, history)
        step_history_path = None
        if log_step_history and step_history:
            step_history_path = str(self.output_dir / "logs" / "pretrain_step_losses_neg_sam.csv")
            write_loss_history(step_history_path, step_history)
        self._log(
            f"Training finished: last_checkpoint={checkpoint_path} "
            f"history_csv={loss_history_path} "
            f"best_epoch={best_epoch} best_total={best_loss:.4f} "
            f"train_time_sec={train_time_sec:.2f}"
        )
        return {
            "checkpoint_path": checkpoint_path,
            "loss_history_path": loss_history_path,
            "step_loss_history_path": step_history_path,
            "cross_domain_pretraining": len(self.training_domains) > 1,
            "training_domains": self.training_domains,
            "training_datasets": self.training_datasets,
            "num_domains": len(self.training_domains),
            "domain_sample_counts": self.domain_sample_counts,
            "sampling_mode": "hyperedge_centered_subhypergraph_minibatch_neg_sam",
            "domain_batch_policy": "balanced_round_robin",
            "pooled_graphs": sorted(self.pool_cache),
            "early_stopping_patience": int(patience),
            "best_epoch": int(best_epoch),
            "best_total_loss": float(best_loss) if best_epoch else None,
            "pretrain_train_time_sec": float(train_time_sec),
        }
