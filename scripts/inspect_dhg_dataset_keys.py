import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _content_keys(dataset) -> List[str]:
    try:
        return sorted(list(dataset.content.keys()))
    except Exception:
        pass
    try:
        return sorted(list(dataset._content.keys()))
    except Exception:
        return []


def _summarize_value(value) -> str:
    if hasattr(value, "shape"):
        try:
            return f"{type(value).__name__}(shape={tuple(value.shape)})"
        except Exception:
            return f"{type(value).__name__}(shape=?)"
    if isinstance(value, list):
        elem_t = type(value[0]).__name__ if value else None
        return f"list(len={len(value)}, elem_type={elem_t})"
    return type(value).__name__


def main() -> None:
    import dhg

    datasets = {
        "Gowalla": getattr(dhg.data, "Gowalla", None),
        "Yelp3k": getattr(dhg.data, "Yelp3k", None),
        "Yelp2018": getattr(dhg.data, "Yelp2018", None),
        "News20": getattr(dhg.data, "News20", None),
        "Cooking200": getattr(dhg.data, "Cooking200", None),
        "MovieLens1M": getattr(dhg.data, "MovieLens1M", None),
        "IMDB4k": getattr(dhg.data, "IMDB4k", None),
        "DBLP8k": getattr(dhg.data, "DBLP8k", None),
        "CocitationCora": getattr(dhg.data, "CocitationCora", None),
    }

    interesting = {
        "edge_list",
        "labels",
        "features",
        "train_mask",
        "val_mask",
        "test_mask",
        "graph_labels",
        "train_adj_list",
        "test_adj_list",
        "num_users",
        "num_items",
        "num_interactions",
        "num_vertices",
        "num_edges",
        "num_classes",
    }

    for name, cls in datasets.items():
        print("=" * 80)
        print("DATASET", name)
        if cls is None:
            print("NOT_AVAILABLE")
            continue
        try:
            ds = cls()
        except Exception as e:
            print("LOAD_ERROR", repr(e))
            continue

        keys = _content_keys(ds)
        print("KEYS", keys)
        for k in keys:
            if k not in interesting:
                continue
            try:
                v = ds[k]
            except Exception as e:
                print(" ", k, "ERROR", repr(e))
                continue
            print(" ", k, "->", _summarize_value(v))


if __name__ == "__main__":
    main()
