"""Generate thesis-oriented Markdown + CSV tables from a benchmark run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def _plot_both_at_k(run_dir: Path, per_sys: Dict[str, Any]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    systems: List[str] = []
    b2: List[float] = []
    b5: List[float] = []
    b10: List[float] = []
    for sys_name in sorted(per_sys.keys()):
        block = per_sys[sys_name]
        if "both@10" not in block:
            continue
        systems.append(sys_name)
        b2.append(float(block.get("both@2", 0)))
        b5.append(float(block.get("both@5", 0)))
        b10.append(float(block.get("both@10", 0)))

    if len(systems) <= 1:
        return

    x = range(len(systems))
    w = 0.25
    fig, ax = plt.subplots(figsize=(max(8, len(systems) * 0.6), 4))
    ax.bar([i - w for i in x], b2, width=w, label="both@2")
    ax.bar(x, b5, width=w, label="both@5")
    ax.bar([i + w for i in x], b10, width=w, label="both@10")
    ax.set_xticks(list(x))
    ax.set_xticklabels(systems, rotation=35, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean recall")
    ax.set_title("Turkuaz-RAG-style multi-context retrieval (both gold docs in top-k)")
    ax.legend()
    fig.tight_layout()
    out = run_dir / "both_recall_at_k.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)

    summary_path = run_dir / "summary.json"
    by_type_path = run_dir / "by_question_type.json"

    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    by_type: Dict[str, Any] = {}
    if by_type_path.is_file():
        by_type = json.loads(by_type_path.read_text(encoding="utf-8"))

    md_lines = [
        "# Turkuaz-RAG benchmark run",
        "",
        f"- Run directory: `{run_dir}`",
        "",
        "## Overall retrieval metrics",
        "",
        "| System | both@2 | both@5 | both@10 | single@10 | MRR |",
        "|--------|--------|--------|---------|-----------|-----|",
    ]

    per_sys = summary.get("per_system", {})
    for sys_name in sorted(per_sys.keys()):
        block = per_sys[sys_name]
        if "both@10" not in block:
            md_lines.append(f"| {sys_name} | (see manifest) | | | | |")
            continue
        md_lines.append(
            f"| {sys_name} | {block.get('both@2', 0):.3f} | {block.get('both@5', 0):.3f} "
            f"| {block.get('both@10', 0):.3f} | {block.get('single@10', 0):.3f} | {block.get('mrr', 0):.3f} |"
        )

    md_lines.extend(["", "## Notes", "", "- **both@k**: both gold news articles appear in the top-k retrieved documents.", "- **single@k**: at least one gold article appears in top-k.", ""])

    if by_type.get("per_system_per_type"):
        md_lines.append("## Breakdown by question type")
        md_lines.append("")
        md_lines.append("See `by_question_type.json` for full JSON; CSV extracts below.")

    md_path = run_dir / "thesis_summary.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    csv_path = run_dir / "metrics_overall.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as cf:
        w = csv.writer(cf)
        w.writerow(["system", "both@2", "both@5", "both@10", "single@10", "mrr"])
        for sys_name in sorted(per_sys.keys()):
            block = per_sys[sys_name]
            if "both@10" not in block:
                w.writerow([sys_name, "", "", "", "", ""])
                continue
            w.writerow(
                [
                    sys_name,
                    f"{block.get('both@2', 0):.6f}",
                    f"{block.get('both@5', 0):.6f}",
                    f"{block.get('both@10', 0):.6f}",
                    f"{block.get('single@10', 0):.6f}",
                    f"{block.get('mrr', 0):.6f}",
                ]
            )

    _plot_both_at_k(run_dir, per_sys)

    print(f"Wrote {md_path} and {csv_path}")
    plot_path = run_dir / "both_recall_at_k.png"
    if plot_path.is_file():
        print(f"Wrote {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
