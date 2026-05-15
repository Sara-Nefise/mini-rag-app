"""
Post-hoc analysis for live eval runs (eval_turkuaz_live_search).

Reads per_sample.jsonl (+ optional summary.json) and produces:
- Metrics by question_type (extends summary.per_question_type with gaps vs overall)
- Optional stratification by question length (word-count buckets)
- Rankings: which types are hardest on both-hit / NDCG

Usage:
  cd src && PYTHONPATH=. python -m scripts.analyze_live_eval \\
    --run-dir experiments/live_eval_results/20260511T103047Z_62ebd1e7
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _repo_src() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _extract_ks(rows: Sequence[Dict[str, Any]]) -> List[int]:
    ks_set: set[int] = set()
    pat = re.compile(r"^(?:single|both|mrr|ndcg)@(\d+)$")
    for r in rows:
        for key in r:
            m = pat.match(key)
            if m:
                ks_set.add(int(m.group(1)))
    return sorted(ks_set)


def _word_count(q: str) -> int:
    return len(re.findall(r"\S+", q or ""))


def _length_bucket(n_words: int, short_max: int, long_min: int) -> str:
    if n_words <= short_max:
        return f"short_<={short_max}_words"
    if n_words < long_min:
        return f"medium_{short_max + 1}-{long_min - 1}_words"
    return f"long_>={long_min}_words"


def _evaluable_retrieval_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rows that participated in retrieval metrics (have single@k etc.)."""
    out: List[Dict[str, Any]] = []
    for r in rows:
        if r.get("error"):
            continue
        if any(k.startswith("single@") for k in r):
            out.append(r)
    return out


def _aggregate_retrieval(
    rows: Sequence[Dict[str, Any]],
    ks: Sequence[int],
    metrics: Sequence[str],
) -> Dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    rec: Dict[str, Any] = {"n": n}
    rec["single_hit_rate_by_k"] = {
        str(k): round(sum(int(r.get(f"single@{k}", 0)) for r in rows) / n, 4) for k in ks
    }
    rec["both_hit_rate_by_k"] = {
        str(k): round(sum(int(r.get(f"both@{k}", 0)) for r in rows) / n, 4) for k in ks
    }
    if "mrr" in metrics:
        rec["mrr_by_k"] = {
            str(k): round(sum(float(r.get(f"mrr@{k}", 0.0)) for r in rows) / n, 4) for k in ks
        }
    if "ndcg" in metrics:
        rec["ndcg_by_k"] = {
            str(k): round(sum(float(r.get(f"ndcg@{k}", 0.0)) for r in rows) / n, 4) for k in ks
        }
    return rec


def _gap_vs_overall(
    type_metrics: Dict[str, Any],
    overall: Dict[str, Any],
    ks: Sequence[int],
    key_prefixes: Tuple[str, ...] = ("single_hit_rate_by_k", "both_hit_rate_by_k", "mrr_by_k", "ndcg_by_k"),
) -> Dict[str, Dict[str, float]]:
    gaps: Dict[str, Dict[str, float]] = {}
    for pk in key_prefixes:
        tm = type_metrics.get(pk)
        om = overall.get(pk)
        if not isinstance(tm, dict) or not isinstance(om, dict):
            continue
        for k in ks:
            sk = str(k)
            if sk in tm and sk in om:
                gaps.setdefault(pk, {})[sk] = round(float(tm[sk]) - float(om[sk]), 4)
    return gaps


def _markdown_table(
    title: str,
    row_label_header: str,
    row_keys: Sequence[str],
    by_row: Dict[str, Dict[str, Any]],
    ks: Sequence[int],
    ref_k: int,
) -> str:
    rk = str(ref_k)
    lines = [
        f"## {title}",
        "",
        f"| {row_label_header} | n | single@{ref_k} | both@{ref_k} | mrr@{ref_k} | ndcg@{ref_k} |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key in row_keys:
        m = by_row.get(key, {})
        n = m.get("n", 0)
        sh = m.get("single_hit_rate_by_k", {}).get(rk, "-")
        bh = m.get("both_hit_rate_by_k", {}).get(rk, "-")
        mr = m.get("mrr_by_k", {}).get(rk, "-") if "mrr_by_k" in m else "-"
        nd = m.get("ndcg_by_k", {}).get(rk, "-") if "ndcg_by_k" in m else "-"
        lines.append(f"| {key} | {n} | {sh} | {bh} | {mr} | {nd} |")
    lines.append("")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze live eval run by question type (+ length buckets).")
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Folder under src/experiments/live_eval_results/<id> or absolute path",
    )
    parser.add_argument("--ref-k", type=int, default=10, help="Primary k for tables and rankings.")
    parser.add_argument("--short-words", type=int, default=40, help="Upper bound (words) for 'short' bucket.")
    parser.add_argument("--long-words", type=int, default=80, help="Lower bound (words) for 'long' bucket.")
    parser.add_argument(
        "--out-prefix",
        type=str,
        default="analysis",
        help="Write {out-prefix}_by_question_type.json and .md",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = (_repo_src() / run_dir).resolve()
    per_path = run_dir / "per_sample.jsonl"
    if not per_path.is_file():
        print(f"Missing {per_path}", flush=True)
        return 1

    summary_path = run_dir / "summary.json"
    summary: Dict[str, Any] = {}
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

    raw_rows = _parse_jsonl(per_path)
    rows = _evaluable_retrieval_rows(raw_rows)
    if not rows:
        print("No evaluable retrieval rows in per_sample.jsonl (need single@k fields).", flush=True)
        return 2

    ks = _extract_ks(rows)
    if not ks and summary.get("overall", {}).get("ks"):
        ks = [int(x) for x in summary["overall"]["ks"]]
    if not ks:
        print("Could not infer ks from per_sample.jsonl or summary.", flush=True)
        return 3

    ref_metrics: List[str] = []
    if any(f"mrr@{ks[0]}" in r for r in rows):
        ref_metrics.append("mrr")
    if any(f"ndcg@{ks[0]}" in r for r in rows):
        ref_metrics.append("ndcg")

    overall = summary.get("overall", {})
    by_qt: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        qt = r.get("question_type") or "unknown"
        by_qt.setdefault(qt, []).append(r)

    per_type: Dict[str, Dict[str, Any]] = {}
    for qt, rs in by_qt.items():
        per_type[qt] = _aggregate_retrieval(rs, ks, ref_metrics)

    ref_k = args.ref_k
    rk = str(ref_k)
    if rk not in [str(k) for k in ks]:
        ref_k = ks[-1]
        rk = str(ref_k)

    gaps_by_type: Dict[str, Any] = {}
    if overall.get("mode") == "retrieval":
        for qt, metrics in per_type.items():
            gaps_by_type[qt] = _gap_vs_overall(metrics, overall, ks)

    # Rank types by both@ref_k ascending (hardest first)
    types_sorted_hard = sorted(
        per_type.keys(),
        key=lambda t: float(per_type[t].get("both_hit_rate_by_k", {}).get(rk, 0.0)),
    )

    # Length buckets (global, not per type — still useful when many types collapse to one)
    by_bucket: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        nwords = _word_count(str(r.get("question", "")))
        b = _length_bucket(nwords, args.short_words, args.long_words)
        by_bucket.setdefault(b, []).append(r)

    per_length_bucket: Dict[str, Dict[str, Any]] = {}
    for bname, rs in by_bucket.items():
        per_length_bucket[bname] = _aggregate_retrieval(rs, ks, ref_metrics)

    one_type_note: Optional[str] = None
    if len(per_type) == 1:
        only = next(iter(per_type.keys()))
        one_type_note = (
            f"Only one question_type in this run ({only!r}). Stratification by type matches overall. "
            "Use Turkuaz CSV / larger slice / jsonl with mixed types for cross-type comparison."
        )

    report: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "per_sample_rows_total": len(raw_rows),
        "evaluable_retrieval_rows": len(rows),
        "ks": ks,
        "ref_k_for_ranking": ref_k,
        "retrieval_profile": overall.get("retrieval_profile") or rows[0].get("retrieval_profile"),
        "overall_from_summary": overall if overall else None,
        "by_question_type": per_type,
        "gap_vs_overall_by_type": gaps_by_type,
        "question_types_sorted_hard_first": types_sorted_hard,
        "by_question_length_bucket": per_length_bucket,
        "length_bucket_thresholds_words": {
            "short_max": args.short_words,
            "long_min": args.long_words,
        },
        "notes": {"single_question_type": one_type_note},
    }

    out_json = run_dir / f"{args.out_prefix}_by_question_type.json"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    md_parts = [
        f"# Live eval analysis — `{run_dir.name}`",
        "",
        f"- Evaluable samples: **{len(rows)}**",
        f"- Retrieval profile: **{report.get('retrieval_profile', '?')}**",
        "",
    ]
    if one_type_note:
        md_parts.extend(["> " + one_type_note, ""])

    md_parts.append(
        _markdown_table("Metrics by question type", "question_type", types_sorted_hard, per_type, ks, ref_k)
    )

    # Length bucket table (fixed order)
    bucket_order = sorted(per_length_bucket.keys())
    md_parts.append(
        _markdown_table(
            "Metrics by question length (words)",
            "length_bucket",
            bucket_order,
            per_length_bucket,
            ks,
            ref_k,
        )
    )

    md_parts.append("## Hardest types (lowest both@" + rk + ", first = hardest)")
    md_parts.append("")
    for i, t in enumerate(types_sorted_hard, start=1):
        bh = per_type[t].get("both_hit_rate_by_k", {}).get(rk, "-")
        md_parts.append(f"{i}. `{t}` — both@{rk}={bh}, n={per_type[t].get('n')}")
    md_parts.append("")

    out_md = run_dir / f"{args.out_prefix}_by_question_type.md"
    out_md.write_text("\n".join(md_parts), encoding="utf-8")

    print(json.dumps({"wrote_json": str(out_json), "wrote_md": str(out_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
