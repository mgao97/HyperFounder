from __future__ import annotations

from typing import Dict

from trainers.downstream_base import DownstreamTrainerBase


class GraphFinetuneTrainer(DownstreamTrainerBase):
    def run(self, task_name: str, heldout_domain: str) -> Dict[str, float | str]:
        if task_name not in {"graph", "graph_cls"}:
            raise ValueError(f"Unsupported graph-level task '{task_name}'.")
        resolved_domain = self.resolve_heldout(heldout_domain)
        dataset_names = self.select_dataset_names(resolved_domain)
        target_graphs = self.load_target_graphs(dataset_names, require_node_splits=False)
        incompatible = []
        for graph in target_graphs:
            has_graph_label = graph.graph_label is not None
            has_graph_split = all(
                key in graph.metadata for key in ("graph_train_indices", "graph_val_indices", "graph_test_indices")
            )
            if not (has_graph_label and has_graph_split):
                incompatible.append(graph.dataset_name)
        if incompatible:
            raise ValueError(
                "The current DHG environment does not expose compatible graph-level datasets with per-graph labels and "
                f"train/val/test splits. Incompatible datasets: {', '.join(incompatible)}"
            )
        raise NotImplementedError("Graph-level fine-tuning is reserved for DHG datasets that satisfy the graph contract.")

