"""Aggregate per-sample JSONL into per-question-type breakdown."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List

from experiments.metrics import aggregate_means

KS = (2, 5, 10)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True, help="Path to a runner output directory")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    per_path = run_dir / "per_sample.jsonl"
    if not per_path.is_file():
        raise SystemExit(f"Missing {per_path}")

    # systems -> question_type -> list of metric rows
    buckets: DefaultDict[str, DefaultDict[str, List[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )

    with per_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            qt = row.get("question_type") or "unknown"
            for sys_name, m in row["systems"].items():
                buckets[sys_name][qt].append(m)

    out: Dict[str, Any] = {"run_dir": str(run_dir), "per_system_per_type": {}}

    for sys_name, qt_map in buckets.items():
        out["per_system_per_type"][sys_name] = {}
        for qt, rows in qt_map.items():
            out["per_system_per_type"][sys_name][qt] = aggregate_means(rows, KS)

    out_path = run_dir / "by_question_type.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
