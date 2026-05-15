"""
Run retrieval ablations against live /index/search endpoint and aggregate results.

Example (same as daily workflow — env-only rerank settings):
  cd mini-rag-app
  PYTHONPATH=src src/.venv/bin/python -m scripts.run_retrieval_ablations \\
    --base-url http://127.0.0.1:8001 --project-id 100 \\
    --source local --data-dir data/turkuaz-rag --limit 100 \\
    --profiles hybrid_xe,hybrid_calibrated_xe,full_xe \\
    --analyze

Optional ``--analyze`` runs ``analyze_live_eval`` on each profile run folder (tables by question_type).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from json import JSONDecoder
from pathlib import Path
from typing import Any, Dict, List


def _repo_src() -> Path:
    return Path(__file__).resolve().parents[1]


def _latest_run_dir(output_root: Path) -> Path | None:
    runs = [p for p in output_root.iterdir() if p.is_dir()]
    if not runs:
        return None
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0]


def _run_profile(args: argparse.Namespace, profile: str) -> Dict:
    src_root = _repo_src()
    output_root = src_root / args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "scripts.eval_turkuaz_live_search",
        "--mode",
        "retrieval",
        "--base-url",
        args.base_url,
        "--project-id",
        str(args.project_id),
        "--source",
        args.source,
        "--data-dir",
        args.data_dir,
        "--limit",
        str(args.limit),
        "--ks",
        args.ks,
        "--retries",
        str(args.retries),
        "--request-timeout",
        str(args.request_timeout),
        "--error-policy",
        args.error_policy,
        "--retrieval-profile",
        profile,
        "--output-dir",
        args.output_dir,
        "--score-calibration",
    ]
    if args.no_progress:
        cmd.append("--no-progress")
    if args.max_per_question_type is not None:
        cmd += ["--max-per-question-type", str(args.max_per_question_type)]
    if args.strict_id_match:
        cmd.append("--strict-id-match")
    if args.allow_duplicates:
        cmd.append("--allow-duplicates")
    if getattr(args, "fusion_ce_weight", None) is not None:
        cmd += ["--fusion-ce-weight", str(args.fusion_ce_weight)]

    # Let child tqdm show on terminal (otherwise capture hides progress until profile ends).
    live_child = sys.stderr.isatty() and not args.no_progress
    proc = subprocess.run(
        cmd,
        cwd=str(src_root),
        capture_output=not live_child,
        text=True,
    )
    if proc.returncode != 0:
        out = proc.stdout or ""
        err = proc.stderr or ""
        if live_child and not (out or err):
            err = "(output went to terminal; re-run with --no-progress to capture logs)"
        raise RuntimeError(
            f"Profile `{profile}` failed with exit code {proc.returncode}\n"
            f"STDOUT:\n{out}\nSTDERR:\n{err}"
        )

    run_dir = _latest_run_dir(output_root)
    if run_dir is None:
        raise RuntimeError(f"No run dir found under {output_root}")
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise RuntimeError(f"Missing summary for profile `{profile}` at {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {"profile": profile, "run_dir": str(run_dir), "summary": summary}


def _run_analyze_after_eval(run_dir: Path, src_root: Path) -> Dict[str, Any]:
    """Run analyze_live_eval on one run folder; returns paths or error."""
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.analyze_live_eval",
            "--run-dir",
            str(run_dir),
        ],
        cwd=str(src_root),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return {"error": (proc.stderr or proc.stdout or "analyze failed").strip()}
    raw = (proc.stdout or "").strip()
    idx = raw.rfind("{")
    if idx >= 0:
        try:
            obj, _ = JSONDecoder().raw_decode(raw[idx:])
            return obj
        except json.JSONDecodeError:
            pass
    return {"error": "no JSON in analyze stdout"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run live retrieval ablation profiles.")
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:8000")
    parser.add_argument("--project-id", type=int, required=True)
    parser.add_argument("--source", choices=("huggingface", "local", "jsonl"), default="local")
    parser.add_argument("--data-dir", type=str, default="data/turkuaz-rag")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max samples per profile; 0 = no cap (entire Turkuaz local set — slow with *_xe).",
    )
    parser.add_argument("--ks", type=str, default="1,3,5,10,20")
    parser.add_argument("--profiles", type=str, default="baseline,hybrid,hybrid_calibrated,coverage,full")
    parser.add_argument("--max-per-question-type", type=int, default=None)
    parser.add_argument("--strict-id-match", action="store_true")
    parser.add_argument("--allow-duplicates", action="store_true")
    parser.add_argument("--error-policy", choices=("continue", "stop"), default="stop")
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=None,
        help="HTTP timeout per request in seconds. If omitted: 240 when any profile contains "
        "'_xe' (cross-encoder is slow), else 60.",
    )
    parser.add_argument("--retries", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="experiments/live_eval_results")
    parser.add_argument("--aggregate-file", type=str, default="ablation_summary.json")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Hide tqdm bars and capture subprocess output (quiet / CI).",
    )
    parser.add_argument(
        "--fusion-ce-weight",
        type=float,
        default=None,
        help="Forwarded to eval as --fusion-ce-weight (use with *_xe profiles).",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="After each profile, run analyze_live_eval (question_type + length buckets) on that run folder.",
    )
    args = parser.parse_args()

    profiles: List[str] = [p.strip() for p in args.profiles.split(",") if p.strip()]
    if args.request_timeout is None:
        args.request_timeout = 240.0 if any("_xe" in p for p in profiles) else 60.0

    if not profiles:
        print("No profiles provided.", file=sys.stderr)
        return 2

    print(f"Using request timeout: {args.request_timeout}s", flush=True)
    lim_note = "all samples" if args.limit <= 0 else f"limit={args.limit}"
    print(f"Sample cap: {lim_note}", flush=True)

    p_outer = None
    if not args.no_progress and sys.stderr.isatty():
        try:
            from tqdm.auto import tqdm

            p_outer = tqdm(
                total=len(profiles),
                desc="Ablation profiles",
                unit="profile",
                dynamic_ncols=True,
            )
        except ImportError:
            p_outer = None

    results: List[Dict] = []
    for profile in profiles:
        if p_outer is not None:
            p_outer.set_postfix_str(profile[:40])
        else:
            print(f"Running profile: {profile}", flush=True)
        rec = _run_profile(args, profile)
        if args.analyze:
            rd = Path(rec["run_dir"])
            print(f"Analyzing {profile} -> {rd.name} ...", flush=True)
            rec["analysis"] = _run_analyze_after_eval(rd, _repo_src())
            if rec["analysis"].get("error"):
                print(f"  warn: {rec['analysis']['error']}", file=sys.stderr, flush=True)
            else:
                print(f"  wrote: {rec['analysis'].get('wrote_md', '?')}", flush=True)
        results.append(rec)
        if p_outer is not None:
            p_outer.update(1)
    if p_outer is not None:
        p_outer.close()

    aggregate = {
        "profiles": profiles,
        "request_timeout": args.request_timeout,
        "limit": (args.limit if args.limit > 0 else None),
        "fusion_ce_weight": args.fusion_ce_weight,
        "analyze": bool(args.analyze),
        "results": results,
    }
    out_path = _repo_src() / args.output_dir / args.aggregate_file
    out_path.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    print(f"\nSaved ablation aggregate to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
