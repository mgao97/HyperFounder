from __future__ import annotations

from dataclasses import dataclass
from inspect import signature
from typing import Any, Callable, Dict, Optional


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    domain: str
    task_type: str
    loader: Callable[[Optional[str]], Any]


def _build_dhg_loader(class_name: str) -> Callable[[Optional[str]], Any]:
    def _loader(data_root: Optional[str] = None):
        try:
            import dhg
        except Exception as e:
            raise RuntimeError("DHG is required. Install it via `pip install dhg`.") from e

        dataset_cls = getattr(dhg.data, class_name, None)
        if dataset_cls is None:
            available = [name for name in dir(dhg.data) if not name.startswith("_")]
            raise ValueError(f"Dataset class '{class_name}' is not available in dhg.data. Available: {', '.join(sorted(available))}")

        try:
            params = signature(dataset_cls).parameters
        except Exception:
            params = {}

        if data_root is not None and "data_root" in params:
            return dataset_cls(data_root=data_root)
        if data_root is not None and len(params) == 1:
            return dataset_cls(data_root)
        return dataset_cls()

    return _loader


DATASET_REGISTRY: Dict[str, DatasetSpec] = {
    "cora": DatasetSpec("cora", "citation", "node_cls", _build_dhg_loader("Cora")),
    "citeseer": DatasetSpec("citeseer", "citation", "node_cls", _build_dhg_loader("Citeseer")),
    "pubmed": DatasetSpec("pubmed", "citation", "node_cls", _build_dhg_loader("Pubmed")),
    "cora_cc": DatasetSpec("cora_cc", "citation", "node_cls", _build_dhg_loader("CocitationCora")),
    "citeseer_cc": DatasetSpec("citeseer_cc", "citation", "node_cls", _build_dhg_loader("CocitationCiteseer")),
    "pubmed_cc": DatasetSpec("pubmed_cc", "citation", "node_cls", _build_dhg_loader("CocitationPubmed")),
    "coauthorship_cora": DatasetSpec("coauthorship_cora", "academic", "node_cls", _build_dhg_loader("CoauthorshipCora")),
    "coauthorship_dblp": DatasetSpec("coauthorship_dblp", "academic", "node_cls", _build_dhg_loader("CoauthorshipDBLP")),
    "dblp_8k": DatasetSpec("dblp_8k", "academic", "node_cls", _build_dhg_loader("DBLP8k")),
    "imdb_4k": DatasetSpec("imdb_4k", "academic", "node_cls", _build_dhg_loader("IMDB4k")),
    "cooking_200": DatasetSpec("cooking_200", "document", "node_cls", _build_dhg_loader("Cooking200")),
    "news20": DatasetSpec("news20", "document", "node_cls", _build_dhg_loader("News20")),
    "tencent_2k": DatasetSpec("tencent_2k", "recommendation", "node_cls", _build_dhg_loader("Tencent2k")),
    "gowalla": DatasetSpec("gowalla", "recommendation", "rec", _build_dhg_loader("Gowalla")),
    "yelp_2018": DatasetSpec("yelp_2018", "recommendation", "rec", _build_dhg_loader("Yelp2018")),
    "movielens_1m": DatasetSpec("movielens_1m", "recommendation", "rec", _build_dhg_loader("MovieLens1M")),
    "house_committees": DatasetSpec("house_committees", "political", "graph_candidate", _build_dhg_loader("HouseCommittees")),
    "walmart_trips": DatasetSpec("walmart_trips", "commerce", "graph_candidate", _build_dhg_loader("WalmartTrips")),
}


def get_dataset_spec(dataset_name: str) -> DatasetSpec:
    if dataset_name not in DATASET_REGISTRY:
        supported = ", ".join(sorted(DATASET_REGISTRY))
        raise ValueError(f"Unknown dataset '{dataset_name}'. Supported datasets: {supported}")
    return DATASET_REGISTRY[dataset_name]


def build_domain_aliases() -> Dict[str, str]:
    return {
        "c": "citation",
        "a": "academic",
        "r": "recommendation",
        "d": "document",
        "p": "political",
        "m": "commerce",
    }
