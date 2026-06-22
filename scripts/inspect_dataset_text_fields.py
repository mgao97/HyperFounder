import re
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.dataset_registry import DATASET_REGISTRY, get_dataset_spec


def _keys_of(dataset) -> List[str]:
    for attr in ("content", "_content"):
        content = getattr(dataset, attr, None)
        if content is None:
            continue
        try:
            keys = list(content.keys())
        except Exception:
            continue
        if keys:
            return sorted(set(keys))
    return []


def _looks_like_raw_text(value) -> bool:
    if isinstance(value, str):
        return True
    if isinstance(value, (list, tuple)) and len(value) > 0 and isinstance(value[0], str):
        return True
    return False


def main() -> None:
    datasets = sorted(DATASET_REGISTRY)

    rx = re.compile(r"(text|title|abstract|review|sentence|doc|document|content|body)", re.I)

    for name in datasets:
        print("=" * 80)
        print("DATASET", name)

        try:
            dataset = get_dataset_spec(name).loader()
        except Exception as e:
            print("LOAD_ERROR", repr(e))
            continue

        keys = _keys_of(dataset)
        print("KEYS", keys)

        candidates = [k for k in keys if rx.search(k)]
        print("TEXT_LIKE_KEYS", candidates)

        for k in candidates:
            value = None
            try:
                value = dataset[k]
            except Exception:
                try:
                    value = getattr(dataset, "_content", {}).get(k)
                except Exception:
                    value = None

            if value is None:
                print(" ", k, "-> None")
                continue

            summary = type(value).__name__
            if isinstance(value, (list, tuple)):
                elem_t = type(value[0]).__name__ if len(value) > 0 else None
                summary += f"(len={len(value)}, elem_type={elem_t})"
            elif isinstance(value, str):
                summary += f"(len={len(value)})"

            print(" ", k, "->", summary, "raw_text=", _looks_like_raw_text(value))

    print("=" * 80)
    print("DONE")


if __name__ == "__main__":
    main()
