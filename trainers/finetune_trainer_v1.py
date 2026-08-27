from __future__ import annotations

from typing import Dict, List
import contextlib
import time

import torch
import torch.nn.functional as F

from models.encoder_v1 import UnifiedHypergraphEncoderV1
from trainers.downstream_base import DownstreamTrainerBase
from utils.eval import summarize_seed_runs
from utils.metrics import multiclass_accuracy, multiclass_macro_f1


class FinetuneTrainerV1(DownstreamTrainerBase):
    """
    v1 finetune trainer: fixes the train/eval representation mismatch.

    Previous version fed the raw `node_emb` (pre-disentanglement) to the
    downstream classifier, while pre-training optimized the *shared* branch
    `z_shared` via the disentanglement + alignment objectives. That made the
    learned transferable structure unused at test time.

    v1: when `use_shared_for_downstream: true` (default), the encoder is called
    with `return_shared=True` and the trained disentanglers, so the downstream
    head receives node_emb that has been fused with the shared (cross-domain)
    branch -- exactly the representation the pre-training objective aligned.
    """

    def _run_node_task(self, encoder, graph, heads=None) -> Dict[str, float]:
        if graph.node_train_mask is None or graph.node_test_mask is None:
            raise ValueError(f"Dataset '{graph.dataset_name}' does not provide node splits.")
        if graph.node_val_mask is None:
            raise ValueError(f"Dataset '{graph.dataset_name}' does not provide validation splits for early stopping.")
        num_classes = int(graph.metadata["num_node_classes"])
        classifier = torch.nn.Linear(int(self.config["model"]["hidden_dim"]), num_classes).to(self.device)
        params = list(encoder.parameters()) + list(classifier.parameters())
        optimizer = torch.optim.AdamW(
            params,
            lr=float(self.config["training"]["lr"]),
            weight_decay=float(self.config["training"].get("weight_decay", 0.0)),
        )

        best_val = -1.0
        best_epoch = -1
        patience = int(self.config["training"].get("early_stopping", {}).get("patience", 50))
        bad_epochs = 0
        best_encoder_state = None
        best_classifier_state = None

        max_epochs = int(self.config["training"]["finetune_epochs"])
        log_interval = int(self.config["training"].get("log_interval_epochs", 10))
        train_start = time.perf_counter()
        print(
            "[HyperFounder][Transfer][Node][v1] Train start:"
            f" dataset={graph.dataset_name}"
            f" epochs={max_epochs}"
            f" patience={patience}"
            f" train_nodes={int(graph.node_train_mask.sum().item())}"
            f" val_nodes={int(graph.node_val_mask.sum().item())}"
            f" test_nodes={int(graph.node_test_mask.sum().item())}"
        )

        x = graph.x.to(self.device)
        labels = graph.node_labels.to(self.device)
        train_mask = graph.node_train_mask.to(self.device)
        val_mask = graph.node_val_mask.to(self.device)
        test_mask = graph.node_test_mask.to(self.device)

        val_check_interval = max(int(self.config["training"].get("val_check_interval", 5)), 1)
        use_amp = self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda") if use_amp else None
        last_val_score = -1.0

        use_shared = bool(self.config["model"].get("use_shared_for_downstream", True))
        node_dis, edge_dis = None, None
        if use_shared and isinstance(encoder, UnifiedHypergraphEncoderV1) and heads is not None:
            node_dis = heads.node_disentangler
            edge_dis = heads.edge_disentangler

        def _encode():
            return encoder(
                graph, x, motif_budget=0, motifs=[], motif_seed=0,
                return_shared=use_shared,
                node_disentangler=node_dis,
                edge_disentangler=edge_dis,
            )[0]

        for epoch in range(max_epochs):
            encoder.train()
            classifier.train()
            optimizer.zero_grad(set_to_none=True)
            amp_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_amp else contextlib.nullcontext()
            with amp_ctx:
                node_emb = _encode()
                logits = classifier(node_emb)
                loss = F.cross_entropy(logits[train_mask], labels[train_mask])
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            do_val = (epoch == 0) or ((epoch + 1) % val_check_interval == 0) or (epoch == max_epochs - 1)
            if do_val:
                encoder.eval()
                classifier.eval()
                with torch.no_grad():
                    node_emb = _encode()
                    val_logits = classifier(node_emb)[val_mask]
                    val_score = multiclass_accuracy(val_logits, labels[val_mask])
                last_val_score = val_score
            else:
                val_score = last_val_score

            should_log = (
                epoch == 0
                or epoch == max_epochs - 1
                or (epoch + 1) % max(log_interval, 1) == 0
            )

            if val_score > best_val:
                best_val = val_score
                best_epoch = epoch
                bad_epochs = 0
                best_encoder_state = {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()}
                best_classifier_state = {k: v.detach().cpu().clone() for k, v in classifier.state_dict().items()}
                if should_log:
                    print(
                        "[HyperFounder][Transfer][Node][v1] Epoch"
                        f" {epoch + 1}/{max_epochs}:"
                        f" loss={float(loss.item()):.4f}"
                        f" val_acc={float(val_score):.4f}"
                        f" best_val_acc={float(best_val):.4f}"
                        " status=best"
                    )
            else:
                bad_epochs += 1
                if should_log:
                    print(
                        "[HyperFounder][Transfer][Node][v1] Epoch"
                        f" {epoch + 1}/{max_epochs}:"
                        f" loss={float(loss.item()):.4f}"
                        f" val_acc={float(val_score):.4f}"
                        f" best_val_acc={float(best_val):.4f}"
                    )
                if bad_epochs >= patience:
                    print(
                        "[HyperFounder][Transfer][Node][v1] Early stop:"
                        f" dataset={graph.dataset_name}"
                        f" epoch={epoch + 1}"
                        f" best_epoch={best_epoch + 1 if best_epoch >= 0 else -1}"
                        f" best_val_acc={float(best_val):.4f}"
                    )
                    break
        train_time_sec = time.perf_counter() - train_start

        if best_encoder_state is not None:
            encoder.load_state_dict(best_encoder_state, strict=False)
        if best_classifier_state is not None:
            classifier.load_state_dict(best_classifier_state, strict=False)

        encoder.eval()
        classifier.eval()
        eval_start = time.perf_counter()
        with torch.no_grad():
            node_emb = _encode()
            test_logits = classifier(node_emb)[test_mask]
            test_labels = labels[test_mask]
            metrics = {
                "accuracy": multiclass_accuracy(test_logits, test_labels),
                "macro_f1": multiclass_macro_f1(test_logits, test_labels, num_classes=num_classes),
                "best_val_accuracy": float(best_val),
                "best_epoch": float(best_epoch),
                "finetune_train_time_sec": float(train_time_sec),
            }
        metrics["finetune_eval_time_sec"] = float(time.perf_counter() - eval_start)
        print(
            "[HyperFounder][Transfer][Node][v1] Eval done:"
            f" dataset={graph.dataset_name}"
            f" acc={float(metrics['accuracy']):.4f}"
            f" macro_f1={float(metrics['macro_f1']):.4f}"
        )
        return metrics

    def load_pretrained_heads(self):
        """Load the TaskHeadsNegSamV1 (with disentanglers) saved in the pretrain checkpoint.

        Returns None if unavailable so the caller falls back to raw embeddings.
        """
        from models.heads_neg_sam_v1 import TaskHeadsNegSamV1
        from pathlib import Path

        checkpoint_path = self.config["training"].get("pretrained_checkpoint")
        if not checkpoint_path or not Path(checkpoint_path).exists():
            return None
        try:
            state = torch.load(checkpoint_path, map_location=self.device)
        except Exception as e:
            print(f"[HyperFounder][v1] failed to load checkpoint: {e}")
            return None
        heads_state = state.get("heads")
        if heads_state is None:
            print("[HyperFounder][v1] checkpoint has no 'heads'; falling back to raw emb.")
            return None
        # Build a heads instance with the same hyper-parameters used during pretraining.
        pcfg = state.get("config", {})
        mcfg = pcfg.get("model", {})
        hcfg = pcfg.get("heads", {})
        heads = TaskHeadsNegSamV1(
            hidden_dim=int(self.config["model"]["hidden_dim"]),
            input_dim=int(self.config["model"]["input_dim"]),
            num_domains=len(pcfg.get("data", {}).get("domain_map", {})) or int(mcfg.get("num_domains", 4)),
            projection_dim=int(hcfg.get("projection_dim", 64)),
            num_motif_types=int(hcfg.get("num_motif_types", 8)),
            num_prototypes=int(hcfg.get("num_prototypes", 8)),
            lambda_orth=float(hcfg.get("lambda_orth", 0.05)),
            lambda_private_domain=float(hcfg.get("lambda_private_domain", 0.05)),
            lambda_align=float(hcfg.get("lambda_align", 0.1)),
            use_confidence_routing=bool(hcfg.get("use_confidence_routing", True)),
            use_node_alignment=bool(hcfg.get("use_node_alignment", True)),
            use_edge_alignment=bool(hcfg.get("use_edge_alignment", True)),
        ).to(self.device)
        cur = heads.state_dict()
        compat = {k: v for k, v in heads_state.items() if k in cur and cur[k].shape == v.shape}
        heads.load_state_dict(compat, strict=False)
        heads.eval()
        return heads

    def run(self, task_name: str, heldout_domain: str) -> Dict[str, float | str]:
        resolved_domain = self.resolve_heldout(heldout_domain)
        target_graphs = self.load_target_graphs(self.select_dataset_names(resolved_domain), require_node_splits=True)
        if task_name != "node":
            raise ValueError(f"Task '{task_name}' is not supported by the current pipeline.")
        graph_scores: List[float] = []
        graph_f1_scores: List[float] = []
        graph_train_times: List[float] = []
        graph_eval_times: List[float] = []
        dataset_results: List[Dict[str, float | str]] = []
        base_seed = int(self.config["training"]["seed"])
        num_seeds = int(self.config["training"].get("num_seeds", 3))
        # (re)load the pre-trained heads so disentanglers are available for shared-branch fusion
        heads = None
        if bool(self.config["model"].get("use_shared_for_downstream", True)):
            try:
                heads = self.load_pretrained_heads()
            except Exception as e:
                print(f"[HyperFounder][v1] could not load pretrained heads: {e}; falling back to raw emb.")
                heads = None
        for graph in target_graphs:
            seed_scores: List[float] = []
            seed_f1_scores: List[float] = []
            seed_train_times: List[float] = []
            seed_eval_times: List[float] = []
            for seed_offset in range(num_seeds):
                run_seed = base_seed + seed_offset
                torch.manual_seed(run_seed)
                encoder = self.build_encoder()
                metrics = self._run_node_task(encoder, graph, heads=heads)
                seed_scores.append(float(metrics["accuracy"]))
                seed_f1_scores.append(float(metrics["macro_f1"]))
                seed_train_times.append(float(metrics["finetune_train_time_sec"]))
                seed_eval_times.append(float(metrics["finetune_eval_time_sec"]))
            graph_summary = summarize_seed_runs(seed_scores, metric_name=f"{task_name}_accuracy")
            graph_f1_summary = summarize_seed_runs(seed_f1_scores, metric_name=f"{task_name}_macro_f1")
            graph_train_summary = summarize_seed_runs(seed_train_times, metric_name="finetune_train_time_sec")
            graph_eval_summary = summarize_seed_runs(seed_eval_times, metric_name="finetune_eval_time_sec")
            graph_scores.append(float(graph_summary[f"{task_name}_accuracy"]))
            graph_f1_scores.append(float(graph_f1_summary[f"{task_name}_macro_f1"]))
            dataset_results.append({
                "dataset_name": graph.dataset_name,
                **graph_summary,
                **graph_f1_summary,
                **graph_train_summary,
                **graph_eval_summary,
            })
        summary = summarize_seed_runs(graph_scores, metric_name=f"{task_name}_accuracy")
        summary.update(summarize_seed_runs(graph_f1_scores, metric_name=f"{task_name}_macro_f1"))
        summary.update(summarize_seed_runs(graph_train_times, metric_name="finetune_train_time_sec"))
        summary.update(summarize_seed_runs(graph_eval_times, metric_name="finetune_eval_time_sec"))
        summary["heldout_domain"] = resolved_domain
        summary["task"] = task_name
        summary["num_graphs"] = len(target_graphs)
        summary["evaluated_datasets"] = [graph.dataset_name for graph in target_graphs]
        summary["dataset_results"] = dataset_results
        return self.attach_pretrain_domains(summary)
