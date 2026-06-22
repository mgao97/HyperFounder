from __future__ import annotations

from typing import Dict, List
import time

import torch
import torch.nn.functional as F

from models.encoder import UnifiedHypergraphEncoder
from trainers.downstream_base import DownstreamTrainerBase
from utils.eval import summarize_seed_runs
from utils.metrics import multiclass_accuracy, multiclass_macro_f1


class FinetuneTrainer(DownstreamTrainerBase):

    def _run_node_task(self, encoder: UnifiedHypergraphEncoder, graph) -> Dict[str, float]:
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
        train_start = time.perf_counter()
        for epoch in range(max_epochs):
            encoder.train()
            classifier.train()
            optimizer.zero_grad()
            node_emb, _, _, _ = encoder(graph, graph.x.to(self.device), motif_budget=0, motifs=[], motif_seed=0)
            logits = classifier(node_emb)
            labels = graph.node_labels.to(self.device)
            loss = F.cross_entropy(logits[graph.node_train_mask], labels[graph.node_train_mask])
            loss.backward()
            optimizer.step()

            encoder.eval()
            classifier.eval()
            with torch.no_grad():
                node_emb, _, _, _ = encoder(graph, graph.x.to(self.device), motif_budget=0, motifs=[], motif_seed=0)
                val_logits = classifier(node_emb)[graph.node_val_mask]
                val_labels = graph.node_labels.to(self.device)[graph.node_val_mask]
                val_score = multiclass_accuracy(val_logits, val_labels)

            if val_score > best_val:
                best_val = val_score
                best_epoch = epoch
                bad_epochs = 0
                best_encoder_state = {k: v.detach().clone() for k, v in encoder.state_dict().items()}
                best_classifier_state = {k: v.detach().clone() for k, v in classifier.state_dict().items()}
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
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
            node_emb, _, _, _ = encoder(graph, graph.x.to(self.device), motif_budget=0, motifs=[], motif_seed=0)
            test_logits = classifier(node_emb)[graph.node_test_mask]
            test_labels = graph.node_labels.to(self.device)[graph.node_test_mask]
            metrics = {
                "accuracy": multiclass_accuracy(test_logits, test_labels),
                "macro_f1": multiclass_macro_f1(test_logits, test_labels, num_classes=num_classes),
                "best_val_accuracy": float(best_val),
                "best_epoch": float(best_epoch),
                "finetune_train_time_sec": float(train_time_sec),
            }
        metrics["finetune_eval_time_sec"] = float(time.perf_counter() - eval_start)
        return metrics

    def run(self, task_name: str, heldout_domain: str) -> Dict[str, float | str]:
        resolved_domain = self.resolve_heldout(heldout_domain)
        target_graphs = self.load_target_graphs(self.select_dataset_names(resolved_domain), require_node_splits=True)
        if task_name != "node":
            raise ValueError(
                f"Task '{task_name}' is not supported by the current DHG-Bench-aligned pipeline."
            )
        graph_scores: List[float] = []
        graph_f1_scores: List[float] = []
        graph_train_times: List[float] = []
        graph_eval_times: List[float] = []
        dataset_results: List[Dict[str, float | str]] = []
        base_seed = int(self.config["training"]["seed"])
        num_seeds = int(self.config["training"].get("num_seeds", 3))
        for graph in target_graphs:
            seed_scores: List[float] = []
            seed_f1_scores: List[float] = []
            seed_train_times: List[float] = []
            seed_eval_times: List[float] = []
            for seed_offset in range(num_seeds):
                torch.manual_seed(base_seed + seed_offset)
                encoder = self.build_encoder()
                metrics = self._run_node_task(encoder, graph)
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
            graph_train_times.append(float(graph_train_summary["finetune_train_time_sec"]))
            graph_eval_times.append(float(graph_eval_summary["finetune_eval_time_sec"]))
            dataset_results.append(
                {
                    "dataset_name": graph.dataset_name,
                    **graph_summary,
                    **graph_f1_summary,
                    **graph_train_summary,
                    **graph_eval_summary,
                }
            )
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
