"""
Sweep RERANKER_FUSION_CE_WEIGHT via per-request JSON (no server restart).

Runs eval_turkuaz_live_search once per weight and prints a compact table.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _src_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_weights(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _latest_run_summary(output_root: Path) -> dict:
    runs = [p for p in output_root.iterdir() if p.is_dir()]
    if not runs:
        raise RuntimeError(f"No run dirs under {output_root}")
    latest = max(runs, key=lambda p: p.stat().st_mtime)
    path = latest / "summary.json"
    if not path.exists():
        raise RuntimeError(f"Missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _run_eval(args: argparse.Namespace, weight: float) -> dict:
    root = _src_root()
    out_root = root / args.output_dir
    out_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "scripts.eval_turkuaz_live_search",
        "--base-url",
        args.base_url,
        "--project-id",
        str(args.project_id),
        "--mode",
        "retrieval",
        "--source",
        args.source,
        "--data-dir",
        args.data_dir,
        "--limit",
        str(args.limit),
        "--ks",
        args.ks,
        "--retrieval-profile",
        args.retrieval_profile,
        "--fusion-ce-weight",
        str(weight),
        "--request-timeout",
        str(args.request_timeout),
        "--retries",
        str(args.retries),
        "--error-policy",
        args.error_policy,
        "--output-dir",
        args.output_dir,
        "--no-progress",
    ]
    if args.score_calibration:
        cmd.append("--score-calibration")
    proc = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"w={weight} failed:\n{proc.stderr or proc.stdout}")
    return _latest_run_summary(out_root)


def main() -> int:
    p = argparse.ArgumentParser(description="Sweep reranker fusion CE weight (live eval).")
    p.add_argument("--base-url", type=str, default="http://127.0.0.1:8000")
    p.add_argument("--project-id", type=int, required=True)
    p.add_argument("--source", choices=("huggingface", "local", "jsonl"), default="local")
    p.add_argument("--data-dir", type=str, default="data/turkuaz-rag")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--ks", type=str, default="1,3,5,10,20")
    p.add_argument("--retrieval-profile", type=str, default="hybrid_xe")
    p.add_argument("--weights", type=str, default="0.3,0.35,0.4,0.45")
    p.add_argument("--request-timeout", type=float, default=300.0)
    p.add_argument("--retries", type=int, default=0)
    p.add_argument("--error-policy", choices=("continue", "stop"), default="stop")
    p.add_argument("--output-dir", type=str, default="experiments/live_eval_results")
    p.add_argument("--score-calibration", action="store_true")
    args = p.parse_args()

    weights = _parse_weights(args.weights)
    rows: list[tuple[float, dict]] = []
    for w in weights:
        print(f"Running fusion_ce_weight={w} ...", flush=True)
        summary = _run_eval(args, w)
        rows.append((w, summary.get("overall", {})))

    print("\n=== fusion_ce_weight | ndcg@10 | both@10 | mrr@10 | single@10 ===")
    for w, ov in rows:
        nd = ov.get("ndcg_by_k", {}).get("10", "-")
        bh = ov.get("both_hit_rate_by_k", {}).get("10", "-")
        mr = ov.get("mrr_by_k", {}).get("10", "-")
        sh = ov.get("single_hit_rate_by_k", {}).get("10", "-")
        print(f"  {w:>4}  |  {nd}  |  {bh}  |  {mr}  |  {sh}")

    out = _src_root() / args.output_dir / "fusion_sweep_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"weights": weights, "rows": [{"w": w, "overall": ov} for w, ov in rows]},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
