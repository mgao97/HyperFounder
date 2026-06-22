from __future__ import annotations

import time
from dataclasses import replace
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F

from trainers.downstream_base import DownstreamTrainerBase
from utils.eval import summarize_seed_runs
from utils.hypergraph import SimpleHypergraph
from utils.metrics import hit_rate_at_k, ndcg_at_k
from utils.minibatch_sampling import expand_hyperedge_centered_subhypergraph


class RecommendationTrainer(DownstreamTrainerBase):
    def _prepare_graph(self, graph: SimpleHypergraph) -> SimpleHypergraph:
        train_adj = graph.metadata["train_adj_list"]
        test_adj = graph.metadata["test_adj_list"]
        train_core: List[List[int]] = []
        val_targets: List[int] = []
        test_targets: List[int] = []
        for train_items, test_items in zip(train_adj, test_adj):
            unique_train = list(dict.fromkeys(int(item_id) for item_id in train_items))
            unique_test = list(dict.fromkeys(int(item_id) for item_id in test_items))
            if len(unique_train) > 1:
                val_targets.append(int(unique_train[-1]))
                train_core.append(unique_train[:-1])
            else:
                val_targets.append(-1)
                train_core.append(unique_train)
            test_targets.append(int(unique_test[0]) if unique_test else -1)
        metadata = dict(graph.metadata)
        metadata["train_core_adj_list"] = train_core
        metadata["val_targets"] = val_targets
        metadata["test_targets"] = test_targets
        return replace(graph, hyperedges=[sorted(set(items)) for items in train_core], metadata=metadata)

    def _sample_negative_item(self, excluded: set[int], num_items: int, seed: int) -> int:
        allowed = [item_id for item_id in range(num_items) if item_id not in excluded]
        if not allowed:
            return 0
        generator = torch.Generator().manual_seed(seed)
        index = int(torch.randint(0, len(allowed), (1,), generator=generator).item())
        return allowed[index]

    def _augment_subhypergraph(
        self,
        parent_graph: SimpleHypergraph,
        subhypergraph: SimpleHypergraph,
        candidate_item_ids: Sequence[int],
    ) -> SimpleHypergraph:
        global_node_ids = list(subhypergraph.metadata.get("global_node_ids", list(range(subhypergraph.num_nodes))))
        existing = set(global_node_ids)
        extra_item_ids = [int(item_id) for item_id in candidate_item_ids if 0 <= int(item_id) < parent_graph.num_nodes and int(item_id) not in existing]
        if not extra_item_ids:
            return subhypergraph
        x_extra = parent_graph.x[extra_item_ids].clone()
        x = torch.cat([subhypergraph.x, x_extra], dim=0)
        node_labels = torch.cat([subhypergraph.node_labels, parent_graph.node_labels[extra_item_ids].clone()], dim=0)
        metadata = dict(subhypergraph.metadata)
        metadata["global_node_ids"] = global_node_ids + extra_item_ids
        false_mask = torch.zeros(len(extra_item_ids), dtype=torch.bool)
        return SimpleHypergraph(
            num_nodes=subhypergraph.num_nodes + len(extra_item_ids),
            hyperedges=[list(edge) for edge in subhypergraph.hyperedges],
            x=x,
            name=subhypergraph.name,
            domain=subhypergraph.domain,
            dataset_name=subhypergraph.dataset_name,
            node_labels=node_labels,
            edge_labels=subhypergraph.edge_labels.clone() if subhypergraph.edge_labels is not None else None,
            graph_label=subhypergraph.graph_label.clone() if subhypergraph.graph_label is not None else None,
            node_train_mask=torch.cat([subhypergraph.node_train_mask, false_mask], dim=0) if subhypergraph.node_train_mask is not None else None,
            node_val_mask=torch.cat([subhypergraph.node_val_mask, false_mask], dim=0) if subhypergraph.node_val_mask is not None else None,
            node_test_mask=torch.cat([subhypergraph.node_test_mask, false_mask], dim=0) if subhypergraph.node_test_mask is not None else None,
            metadata=metadata,
        )

    def _build_user_subhypergraph(
        self,
        graph: SimpleHypergraph,
        user_id: int,
        candidate_item_ids: Sequence[int],
        seed: int,
    ) -> tuple[SimpleHypergraph, int, Dict[int, int]]:
        minibatch_config = self.config["training"].get("minibatch", {})
        subhypergraph = expand_hyperedge_centered_subhypergraph(
            graph,
            seed_edge_ids=[user_id],
            max_nodes=int(minibatch_config.get("max_nodes", 256)),
            max_edges=int(minibatch_config.get("max_edges", 128)),
            expansion_hops=int(minibatch_config.get("expansion_hops", 2)),
            seed=seed,
        )
        subhypergraph = self._augment_subhypergraph(graph, subhypergraph, candidate_item_ids)
        global_edge_ids = list(subhypergraph.metadata.get("global_edge_ids", []))
        if user_id not in global_edge_ids:
            raise ValueError(f"User edge {user_id} is missing from sampled subhypergraph.")
        local_user_edge_id = global_edge_ids.index(user_id)
        global_node_ids = list(subhypergraph.metadata.get("global_node_ids", list(range(subhypergraph.num_nodes))))
        node_mapping = {global_id: local_id for local_id, global_id in enumerate(global_node_ids)}
        return subhypergraph, local_user_edge_id, node_mapping

    def _ranking_metrics(self, ranked_items: Sequence[int], positive_item: int) -> Dict[str, float]:
        return {
            "hr@5": hit_rate_at_k(ranked_items, positive_item, 5),
            "hr@10": hit_rate_at_k(ranked_items, positive_item, 10),
            "ndcg@5": ndcg_at_k(ranked_items, positive_item, 5),
            "ndcg@10": ndcg_at_k(ranked_items, positive_item, 10),
        }

    def _evaluate_split(
        self,
        encoder,
        graph: SimpleHypergraph,
        targets: Sequence[int],
        eval_users: Sequence[int],
        seed: int,
    ) -> Dict[str, float]:
        all_metrics: List[Dict[str, float]] = []
        eval_negative_samples = int(self.config["training"].get("eval_negative_samples", 99))
        train_core = graph.metadata["train_core_adj_list"]
        raw_train = graph.metadata["train_adj_list"]
        raw_test = graph.metadata["test_targets"]
        eval_start = time.perf_counter()
        encoder.eval()
        with torch.no_grad():
            for offset, user_id in enumerate(eval_users):
                positive_item = int(targets[user_id])
                if positive_item < 0:
                    continue
                excluded = set(int(item_id) for item_id in raw_train[user_id])
                test_item = int(raw_test[user_id])
                if test_item >= 0:
                    excluded.add(test_item)
                negative_items = []
                for negative_index in range(eval_negative_samples):
                    negative_items.append(
                        self._sample_negative_item(
                            excluded=set(excluded).union(negative_items).union({positive_item}),
                            num_items=int(graph.metadata["num_items"]),
                            seed=seed + user_id * 1009 + negative_index,
                        )
                    )
                subhypergraph, local_user_edge_id, node_mapping = self._build_user_subhypergraph(
                    graph,
                    user_id=user_id,
                    candidate_item_ids=[positive_item] + negative_items,
                    seed=seed + offset * 37,
                )
                node_emb, edge_emb, _, _ = encoder(
                    subhypergraph,
                    subhypergraph.x.to(self.device),
                    motif_budget=0,
                    motifs=[],
                    communities=[],
                    motif_seed=0,
                )
                user_emb = edge_emb[local_user_edge_id]
                candidate_ids = [positive_item] + negative_items
                candidate_scores = []
                for item_id in candidate_ids:
                    item_emb = node_emb[node_mapping[item_id]]
                    candidate_scores.append(float(torch.dot(user_emb, item_emb).item()))
                ranking = [item_id for _, item_id in sorted(zip(candidate_scores, candidate_ids), key=lambda pair: pair[0], reverse=True)]
                all_metrics.append(self._ranking_metrics(ranking, positive_item))
        eval_time = time.perf_counter() - eval_start
        if not all_metrics:
            return {
                "hr@5": 0.0,
                "hr@10": 0.0,
                "ndcg@5": 0.0,
                "ndcg@10": 0.0,
                "finetune_eval_time_sec": float(eval_time),
            }
        summary = {
            key: float(sum(metric[key] for metric in all_metrics) / len(all_metrics))
            for key in all_metrics[0]
        }
        summary["finetune_eval_time_sec"] = float(eval_time)
        return summary

    def _run_recommendation_task(self, encoder, graph: SimpleHypergraph) -> Dict[str, float]:
        graph = self._prepare_graph(graph)
        optimizer = torch.optim.AdamW(
            encoder.parameters(),
            lr=float(self.config["training"]["lr"]),
            weight_decay=float(self.config["training"].get("weight_decay", 0.0)),
        )
        train_core = graph.metadata["train_core_adj_list"]
        val_targets = graph.metadata["val_targets"]
        train_users = [user_id for user_id, items in enumerate(train_core) if items]
        val_users = [user_id for user_id, item_id in enumerate(val_targets) if item_id >= 0 and train_core[user_id]]
        if not train_users:
            raise ValueError(f"Recommendation dataset '{graph.dataset_name}' has no trainable users.")
        users_per_epoch = min(int(self.config["training"].get("users_per_epoch", 256)), len(train_users))
        negatives_per_positive = int(self.config["training"].get("negatives_per_positive", 1))
        patience = int(self.config["training"].get("early_stopping", {}).get("patience", 50))
        best_val = -1.0
        best_epoch = -1
        bad_epochs = 0
        best_encoder_state = None

        train_start = time.perf_counter()
        for epoch in range(int(self.config["training"]["finetune_epochs"])):
            generator = torch.Generator().manual_seed(int(self.config["training"]["seed"]) + epoch * 97)
            permutation = torch.randperm(len(train_users), generator=generator).tolist()[:users_per_epoch]
            sampled_users = [train_users[index] for index in permutation]
            encoder.train()
            optimizer.zero_grad()
            batch_losses = []
            for user_offset, user_id in enumerate(sampled_users):
                positive_items = train_core[user_id]
                positive_index = int(torch.randint(0, len(positive_items), (1,), generator=generator).item())
                positive_item = int(positive_items[positive_index])
                observed = set(int(item_id) for item_id in graph.metadata["train_adj_list"][user_id])
                negative_items = [
                    self._sample_negative_item(
                        excluded=observed.union({positive_item}),
                        num_items=int(graph.metadata["num_items"]),
                        seed=int(self.config["training"]["seed"]) + epoch * 10000 + user_id * 97 + negative_offset,
                    )
                    for negative_offset in range(negatives_per_positive)
                ]
                subhypergraph, local_user_edge_id, node_mapping = self._build_user_subhypergraph(
                    graph,
                    user_id=user_id,
                    candidate_item_ids=[positive_item] + negative_items,
                    seed=int(self.config["training"]["seed"]) + epoch * 10000 + user_offset,
                )
                node_emb, edge_emb, _, _ = encoder(
                    subhypergraph,
                    subhypergraph.x.to(self.device),
                    motif_budget=0,
                    motifs=[],
                    communities=[],
                    motif_seed=0,
                )
                user_emb = edge_emb[local_user_edge_id]
                pos_emb = node_emb[node_mapping[positive_item]]
                pos_score = torch.dot(user_emb, pos_emb)
                for negative_item in negative_items:
                    neg_emb = node_emb[node_mapping[negative_item]]
                    neg_score = torch.dot(user_emb, neg_emb)
                    batch_losses.append(-F.logsigmoid(pos_score - neg_score))
            if batch_losses:
                loss = torch.stack(batch_losses).mean()
                loss.backward()
                optimizer.step()

            if val_users:
                val_metrics = self._evaluate_split(
                    encoder,
                    graph,
                    targets=val_targets,
                    eval_users=val_users,
                    seed=int(self.config["training"]["seed"]) + epoch * 1000,
                )
                val_score = float(val_metrics["hr@10"])
            else:
                val_score = 0.0
            if val_score > best_val:
                best_val = val_score
                best_epoch = epoch
                bad_epochs = 0
                best_encoder_state = {k: v.detach().clone() for k, v in encoder.state_dict().items()}
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    break
        train_time_sec = time.perf_counter() - train_start
        if best_encoder_state is not None:
            encoder.load_state_dict(best_encoder_state, strict=False)
        test_users = [user_id for user_id, item_id in enumerate(graph.metadata["test_targets"]) if item_id >= 0 and train_core[user_id]]
        test_metrics = self._evaluate_split(
            encoder,
            graph,
            targets=graph.metadata["test_targets"],
            eval_users=test_users,
            seed=int(self.config["training"]["seed"]) + 999999,
        )
        test_metrics.update(
            {
                "best_val_hr@10": float(best_val),
                "best_epoch": float(best_epoch),
                "finetune_train_time_sec": float(train_time_sec),
            }
        )
        return test_metrics

    def run(self, task_name: str, heldout_domain: str) -> Dict[str, float | str]:
        if task_name not in {"rec", "recommendation"}:
            raise ValueError(f"Unsupported recommendation task '{task_name}'.")
        resolved_domain = self.resolve_heldout(heldout_domain)
        target_graphs = self.load_target_graphs(self.select_dataset_names(resolved_domain), require_node_splits=False)
        metric_names = ["hr@5", "hr@10", "ndcg@5", "ndcg@10", "finetune_train_time_sec", "finetune_eval_time_sec"]
        aggregate: Dict[str, List[float]] = {name: [] for name in metric_names}
        dataset_results: List[Dict[str, float | str]] = []
        base_seed = int(self.config["training"]["seed"])
        num_seeds = int(self.config["training"].get("num_seeds", 3))
        for graph in target_graphs:
            seed_metrics: Dict[str, List[float]] = {name: [] for name in metric_names}
            for seed_offset in range(num_seeds):
                torch.manual_seed(base_seed + seed_offset)
                encoder = self.build_encoder()
                metrics = self._run_recommendation_task(encoder, graph)
                for metric_name in metric_names:
                    seed_metrics[metric_name].append(float(metrics[metric_name]))
            dataset_summary: Dict[str, float | str] = {"dataset_name": graph.dataset_name}
            for metric_name in metric_names:
                metric_summary = summarize_seed_runs(seed_metrics[metric_name], metric_name=metric_name)
                dataset_summary.update(metric_summary)
                aggregate[metric_name].append(float(metric_summary[metric_name]))
            dataset_results.append(dataset_summary)

        summary: Dict[str, float | str] = {
            "heldout_domain": resolved_domain,
            "task": "rec",
            "num_graphs": len(target_graphs),
            "evaluated_datasets": [graph.dataset_name for graph in target_graphs],
            "dataset_results": dataset_results,
        }
        for metric_name in metric_names:
            summary.update(summarize_seed_runs(aggregate[metric_name], metric_name=metric_name))
        return self.attach_pretrain_domains(summary)

