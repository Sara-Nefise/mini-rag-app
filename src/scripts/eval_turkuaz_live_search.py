"""
Live evaluation harness for Turkuaz over running RAG APIs.

Supports:
- retrieval mode (/index/search)
- answer mode (/index/answer)
- multi-k metrics (hit/single/both, MRR, nDCG)
- retries/timeout, error policy
- per-question-type analysis
- run artifact persistence (manifest, summary, per-sample, plots)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import statistics
import sys
import uuid
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import httpx

from experiments.turkuaz_loader import load_huggingface, load_jsonl, load_local_directory
from experiments.types import TurkuazSample


def _repo_src() -> Path:
    return Path(__file__).resolve().parents[1]


def _project_root() -> Path:
    return _repo_src().parent


def _make_run_dir(output_dir: str) -> Path:
    out_root = _repo_src() / output_dir
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _rag_config_snapshot() -> Dict[str, Any]:
    try:
        from helpers.config import get_settings

        s = get_settings()
        return {
            "generation_backend": s.GENERATION_BACKEND,
            "generation_model_id": s.GENERATION_MODEL_ID,
            "embedding_backend": s.EMBEDDING_BACKEND,
            "embedding_model_id": s.EMBEDDING_MODEL_ID,
            "embedding_model_size": s.EMBEDDING_MODEL_SIZE,
            "embedding_batch_size": s.EMBEDDING_BATCH_SIZE,
            "embedding_input_max_chars": s.EMBEDDING_INPUT_MAX_CHARS,
            "embedding_max_input_tokens": s.EMBEDDING_MAX_INPUT_TOKENS,
            "vector_db_backend": s.VECTOR_DB_BACKEND,
            "vector_db_distance_method": s.VECTOR_DB_DISTANCE_METHOD,
            "vector_db_index_threshold": s.VECTOR_DB_PGVEC_INDEX_THRESHOLD,
            "primary_lang": s.PRIMARY_LANG,
            "default_lang": s.DEFAULT_LANG,
            "reranker_enabled": getattr(s, "RERANKER_ENABLED", False),
            "reranker_model_id": getattr(s, "RERANKER_MODEL_ID", None),
            "reranker_pool_size": getattr(s, "RERANKER_POOL_SIZE", None),
            "reranker_fusion_ce_weight": getattr(s, "RERANKER_FUSION_CE_WEIGHT", None),
            "reranker_device": getattr(s, "RERANKER_DEVICE", None),
        }
    except Exception as e:
        return {"config_load_error": str(e)}


def _parse_ks(s: str) -> List[int]:
    vals = sorted({int(x.strip()) for x in s.split(",") if x.strip()})
    return [v for v in vals if v > 0]


def _parse_csv_str(s: str) -> List[str]:
    return [x.strip().lower() for x in s.split(",") if x.strip()]


def _normalize_text(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").strip().lower())


def _token_set(t: str) -> set[str]:
    return set(_normalize_text(t).split())


def _answer_exact(pred: str, gold: str) -> float:
    return 1.0 if _normalize_text(pred) == _normalize_text(gold) else 0.0


def _answer_f1(pred: str, gold: str) -> float:
    p = _token_set(pred)
    g = _token_set(gold)
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    inter = len(p & g)
    if inter == 0:
        return 0.0
    prec = inter / len(p)
    rec = inter / len(g)
    return 2 * prec * rec / (prec + rec)


def _answer_semantic(pred: str, gold: str) -> float:
    return SequenceMatcher(None, _normalize_text(pred), _normalize_text(gold)).ratio()


def _load_samples(args: argparse.Namespace) -> List[TurkuazSample]:
    lim = None if args.limit <= 0 else args.limit
    if args.source == "huggingface":
        samples = load_huggingface(limit=lim)
    elif args.source == "local":
        data_path = Path(args.data_dir)
        if not data_path.is_absolute():
            data_path = (_project_root() / data_path).resolve()
        samples = load_local_directory(data_path, limit=lim)
    else:
        path = Path(args.jsonl_path)
        if not path.is_absolute():
            path = (_repo_src() / path).resolve()
        samples = load_jsonl(path, limit=lim)

    if args.max_per_question_type and args.max_per_question_type > 0:
        grouped: Dict[str, List[TurkuazSample]] = {}
        for s in samples:
            grouped.setdefault(s.question_type or "unknown", []).append(s)
        out: List[TurkuazSample] = []
        for _, rows in grouped.items():
            out.extend(rows[: args.max_per_question_type])
        return out
    return samples


def _call_with_retries(
    client: httpx.Client,
    method: str,
    url: str,
    json_body: Dict[str, Any],
    retries: int,
) -> httpx.Response:
    last_err: Optional[Exception] = None
    for i in range(retries + 1):
        try:
            r = client.request(method, url, json=json_body)
            r.raise_for_status()
            return r
        except Exception as e:  # noqa: BLE001
            last_err = e
            if i == retries:
                raise
    raise RuntimeError(f"Unreachable retry state: {last_err}")


def _search_request_body(
    question: str,
    k: int,
    retrieval_profile: str,
    fusion_ce_weight: Optional[float],
) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "text": question,
        "limit": k,
        "retrieval_profile": retrieval_profile,
    }
    if fusion_ce_weight is not None:
        body["reranker_fusion_ce_weight"] = float(fusion_ce_weight)
    return body


def _fetch_search(
    client: httpx.Client,
    base_url: str,
    project_id: int,
    question: str,
    k: int,
    retries: int,
    retrieval_profile: str = "baseline",
    fusion_ce_weight: Optional[float] = None,
) -> List[Dict[str, Any]]:
    url = f"{base_url.rstrip('/')}/api/v1/nlp/index/search/{project_id}"
    resp = _call_with_retries(
        client,
        "POST",
        url,
        _search_request_body(question, k, retrieval_profile, fusion_ce_weight),
        retries=retries,
    )
    body = resp.json()
    results = body.get("results") or []
    return [r for r in results if isinstance(r, dict)]


def _fetch_answer(
    client: httpx.Client,
    base_url: str,
    project_id: int,
    question: str,
    k: int,
    retries: int,
    retrieval_profile: str = "baseline",
    fusion_ce_weight: Optional[float] = None,
) -> str:
    url = f"{base_url.rstrip('/')}/api/v1/nlp/index/answer/{project_id}"
    resp = _call_with_retries(
        client,
        "POST",
        url,
        _search_request_body(question, k, retrieval_profile, fusion_ce_weight),
        retries=retries,
    )
    body = resp.json()
    return str(body.get("answer") or "")


def _retrieved_mlsum_ids(
    retrieved: Sequence[Dict[str, Any]],
    min_score_threshold: Optional[float],
    allow_duplicates: bool,
) -> List[str]:
    out: List[str] = []
    seen = set()
    for rec in retrieved:
        score = rec.get("score")
        if min_score_threshold is not None:
            try:
                if score is None or float(score) < min_score_threshold:
                    continue
            except Exception:
                continue
        md = rec.get("metadata")
        if isinstance(md, dict) and md.get("mlsum_id") is not None:
            mid = str(md.get("mlsum_id")).strip()
            if not allow_duplicates and mid in seen:
                continue
            seen.add(mid)
            out.append(mid)
    return out


def _single_both_at_k(gold_ids: Tuple[str, str], ranked_ids: Sequence[str], k: int) -> Tuple[int, int]:
    g1, g2 = (str(gold_ids[0]).strip(), str(gold_ids[1]).strip())
    top = set(ranked_ids[:k])
    hit1 = g1 in top
    hit2 = g2 in top
    return (1 if (hit1 or hit2) else 0, 1 if (hit1 and hit2) else 0)


def _rank_of_first_relevant(gold_ids: Tuple[str, str], ranked_ids: Sequence[str]) -> Optional[int]:
    gold = {str(gold_ids[0]).strip(), str(gold_ids[1]).strip()}
    for i, rid in enumerate(ranked_ids, start=1):
        if rid in gold:
            return i
    return None


def _dcg(rels: Sequence[int], k: int) -> float:
    total = 0.0
    for i, rel in enumerate(rels[:k], start=1):
        if rel > 0:
            total += rel / math.log2(i + 1)
    return total


def _ndcg_at_k(gold_ids: Tuple[str, str], ranked_ids: Sequence[str], k: int) -> float:
    gold = {str(gold_ids[0]).strip(), str(gold_ids[1]).strip()}
    rels = [1 if rid in gold else 0 for rid in ranked_ids[:k]]
    dcg = _dcg(rels, k)
    ideal = _dcg([1, 1] + [0] * max(0, k - 2), k)
    return dcg / ideal if ideal > 0 else 0.0


def _write_metrics_plot(run_dir: Path, summary: Dict[str, Any]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    metrics = summary.get("overall", {})
    ks = metrics.get("ks", [])
    if not ks:
        return
    single_vals = [metrics["single_hit_rate_by_k"].get(str(k), 0.0) for k in ks]
    both_vals = [metrics["both_hit_rate_by_k"].get(str(k), 0.0) for k in ks]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ks, single_vals, marker="o", label="Single@k")
    ax.plot(ks, both_vals, marker="o", label="Both@k")
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("k")
    ax.set_ylabel("Rate")
    ax.set_title(f"Live Retrieval Metrics (n={metrics.get('n_samples', 0)})")
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "metrics.png", dpi=160)
    plt.close(fig)


def _maybe_judge_with_llm(pred: str, gold: str, judge_model: Optional[str]) -> Optional[float]:
    if not judge_model:
        return None
    try:
        from openai import OpenAI

        base_url = os.environ.get("OPENAI_API_URL")
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return None
        client = OpenAI(api_key=api_key, base_url=base_url if base_url else None)
        prompt = (
            "Score semantic agreement from 0 to 1 between prediction and gold answer.\n"
            f"Prediction: {pred}\nGold: {gold}\n"
            "Output only a number."
        )
        r = client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=8,
        )
        txt = (r.choices[0].message.content or "").strip()
        return max(0.0, min(1.0, float(txt)))
    except Exception:
        return None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate live RAG endpoints against Turkuaz.")
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:8000")
    parser.add_argument("--project-id", type=int, required=True)
    parser.add_argument("--mode", choices=("retrieval", "answer"), default="retrieval")
    parser.add_argument("--source", choices=("huggingface", "local", "jsonl"), default="local")
    parser.add_argument("--data-dir", type=str, default="data/turkuaz-rag")
    parser.add_argument("--jsonl-path", type=str, default="experiments/fixtures/sample_eval.jsonl")
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Max evaluation samples. Use 0 for no cap (load entire dataset — can be slow).",
    )
    parser.add_argument("--ks", type=str, default="1,3,5,10,20")
    parser.add_argument("--k", type=int, default=5, help="Backward-compat single k; used if --ks empty.")
    parser.add_argument("--min-score-threshold", type=float, default=None)
    parser.add_argument("--max-per-question-type", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strict-id-match", action="store_true")
    parser.add_argument("--allow-duplicates", action="store_true")
    parser.add_argument("--metrics", type=str, default="recall,mrr,hit,ndcg")
    parser.add_argument("--score-calibration", action="store_true")
    parser.add_argument("--save-topn-text", type=int, default=3)
    parser.add_argument("--error-policy", choices=("continue", "stop"), default="stop")
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=60.0,
        help="Per-request HTTP timeout (s). Cross-encoder profiles (*_xe) often need 180–300+.",
    )
    parser.add_argument("--retries", type=int, default=0)
    parser.add_argument("--judge-model", type=str, default=None)
    parser.add_argument("--answer-metrics", type=str, default="exact,f1,semantic")
    parser.add_argument("--show-failures", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="experiments/live_eval_results")
    parser.add_argument("--retrieval-profile", type=str, default="baseline")
    parser.add_argument(
        "--fusion-ce-weight",
        type=float,
        default=None,
        help="Per-request CE fusion weight (0–1); JSON reranker_fusion_ce_weight. *_xe profiles only.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bar (e.g. CI logs).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    random.seed(args.seed)
    run_dir = _make_run_dir(args.output_dir)
    ks = _parse_ks(args.ks) or [args.k]
    max_k = max(ks)
    metric_set = set(_parse_csv_str(args.metrics))
    ans_metric_set = set(_parse_csv_str(args.answer_metrics))

    samples = _load_samples(args)
    if not samples:
        print("No Turkuaz samples loaded.", file=sys.stderr)
        return 1
    if args.limit <= 0:
        print(f"Evaluating all loaded samples (n={len(samples)}, --limit 0 = no cap).", flush=True)

    failures: List[Dict[str, Any]] = []
    per_sample_rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    # retrieval aggregates
    single_hits = {k: 0 for k in ks}
    both_hits = {k: 0 for k in ks}
    mrr_sum = {k: 0.0 for k in ks}
    ndcg_sum = {k: 0.0 for k in ks}
    score_hit: List[float] = []
    score_miss: List[float] = []

    # answer aggregates
    ans_exact_sum = 0.0
    ans_f1_sum = 0.0
    ans_sem_sum = 0.0
    ans_llm_sum = 0.0
    ans_llm_count = 0

    evaluated = 0

    pbar = None
    if not args.no_progress and sys.stderr.isatty():
        try:
            from tqdm.auto import tqdm

            pbar = tqdm(
                total=len(samples),
                desc=f"{args.mode} [{args.retrieval_profile}]",
                unit="sample",
                dynamic_ncols=True,
            )
        except ImportError:
            pbar = None

    with httpx.Client(timeout=httpx.Timeout(args.request_timeout)) as client:
        for idx, s in enumerate(samples, start=1):
            row: Dict[str, Any] = {
                "sample_id": s.sample_id,
                "question_type": s.question_type or "unknown",
                "question": s.question,
                "retrieval_profile": args.retrieval_profile,
            }
            try:
                if args.mode == "retrieval":
                    retrieved = _fetch_search(
                        client=client,
                        base_url=args.base_url,
                        project_id=args.project_id,
                        question=s.question,
                        k=max_k,
                        retries=args.retries,
                        retrieval_profile=args.retrieval_profile,
                        fusion_ce_weight=args.fusion_ce_weight,
                    )
                    if not s.gold_mlsum_ids:
                        if pbar is not None:
                            pbar.update(1)
                            pbar.set_postfix_str(f"ok={evaluated} err={len(errors)} skip=no_gold")
                        continue
                    ranked_ids = _retrieved_mlsum_ids(
                        retrieved,
                        min_score_threshold=args.min_score_threshold,
                        allow_duplicates=args.allow_duplicates,
                    )
                    if args.strict_id_match:
                        ranked_ids = [x for x in ranked_ids if x.isdigit()]

                    evaluated += 1
                    row["gold_ids"] = list(s.gold_mlsum_ids)
                    row["retrieved_mlsum_ids"] = ranked_ids
                    if args.save_topn_text and args.save_topn_text > 0:
                        row["retrieved_preview"] = [
                            str(r.get("text", ""))[:300]
                            for r in retrieved[: args.save_topn_text]
                        ]

                    for k in ks:
                        s_hit, b_hit = _single_both_at_k(s.gold_mlsum_ids, ranked_ids, k)
                        single_hits[k] += s_hit
                        both_hits[k] += b_hit
                        row[f"single@{k}"] = s_hit
                        row[f"both@{k}"] = b_hit

                        if "mrr" in metric_set:
                            rnk = _rank_of_first_relevant(s.gold_mlsum_ids, ranked_ids[:k])
                            mrr = 1.0 / rnk if rnk else 0.0
                            mrr_sum[k] += mrr
                            row[f"mrr@{k}"] = mrr
                        if "ndcg" in metric_set:
                            n = _ndcg_at_k(s.gold_mlsum_ids, ranked_ids, k)
                            ndcg_sum[k] += n
                            row[f"ndcg@{k}"] = n

                    if args.score_calibration:
                        gold = {str(s.gold_mlsum_ids[0]).strip(), str(s.gold_mlsum_ids[1]).strip()}
                        for rec in retrieved:
                            md = rec.get("metadata")
                            mid = str(md.get("mlsum_id")).strip() if isinstance(md, dict) and md.get("mlsum_id") is not None else None
                            sc = rec.get("score")
                            if mid is None or sc is None:
                                continue
                            try:
                                v = float(sc)
                            except Exception:
                                continue
                            if mid in gold:
                                score_hit.append(v)
                            else:
                                score_miss.append(v)

                    if row.get(f"both@{ks[0]}", 0) == 0 and len(failures) < args.show_failures:
                        failures.append(
                            {
                                "sample_id": s.sample_id,
                                "question": s.question,
                                "gold_ids": list(s.gold_mlsum_ids),
                                "retrieved_mlsum_ids": ranked_ids[:max_k],
                                "retrieved_preview": row.get("retrieved_preview", []),
                            }
                        )
                else:
                    answer = _fetch_answer(
                        client=client,
                        base_url=args.base_url,
                        project_id=args.project_id,
                        question=s.question,
                        k=max_k,
                        retries=args.retries,
                        retrieval_profile=args.retrieval_profile,
                        fusion_ce_weight=args.fusion_ce_weight,
                    )
                    evaluated += 1
                    row["gold_answer"] = s.answer
                    row["pred_answer"] = answer
                    if "exact" in ans_metric_set:
                        x = _answer_exact(answer, s.answer)
                        ans_exact_sum += x
                        row["exact"] = x
                    if "f1" in ans_metric_set:
                        f = _answer_f1(answer, s.answer)
                        ans_f1_sum += f
                        row["f1"] = f
                    if "semantic" in ans_metric_set:
                        sem = _answer_semantic(answer, s.answer)
                        ans_sem_sum += sem
                        row["semantic"] = sem
                    if args.judge_model:
                        j = _maybe_judge_with_llm(answer, s.answer, args.judge_model)
                        row["judge_llm"] = j
                        if j is not None:
                            ans_llm_sum += j
                            ans_llm_count += 1

            except Exception as e:  # noqa: BLE001
                err = {"sample_id": s.sample_id, "question": s.question, "error": str(e)}
                errors.append(err)
                if args.error_policy == "stop":
                    print(f"[{idx}/{len(samples)}] request failed: {e}", file=sys.stderr)
                    break
                row["error"] = str(e)

            per_sample_rows.append(row)
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix_str(f"ok={evaluated} err={len(errors)}")
            elif idx % 20 == 0 or idx == len(samples):
                print(f"Processed {idx}/{len(samples)}...", flush=True)

    if pbar is not None:
        pbar.close()

    if evaluated == 0:
        print("No evaluable rows.", file=sys.stderr)
        return 3

    overall: Dict[str, Any] = {
        "mode": args.mode,
        "n_samples": evaluated,
        "ks": ks,
        "retrieval_profile": args.retrieval_profile,
    }
    if args.mode == "retrieval":
        overall["single_hit_rate_by_k"] = {str(k): round(single_hits[k] / evaluated, 4) for k in ks}
        overall["both_hit_rate_by_k"] = {str(k): round(both_hits[k] / evaluated, 4) for k in ks}
        if "mrr" in metric_set:
            overall["mrr_by_k"] = {str(k): round(mrr_sum[k] / evaluated, 4) for k in ks}
        if "ndcg" in metric_set:
            overall["ndcg_by_k"] = {str(k): round(ndcg_sum[k] / evaluated, 4) for k in ks}
        if args.score_calibration:
            def _stats(v: List[float]) -> Dict[str, float]:
                if not v:
                    return {}
                sv = sorted(v)
                return {
                    "count": float(len(v)),
                    "mean": round(statistics.fmean(v), 4),
                    "p50": round(sv[len(sv) // 2], 4),
                    "p95": round(sv[min(len(sv) - 1, int(0.95 * (len(sv) - 1)))], 4),
                }

            overall["score_calibration"] = {"hit": _stats(score_hit), "miss": _stats(score_miss)}
    else:
        if "exact" in ans_metric_set:
            overall["exact"] = round(ans_exact_sum / evaluated, 4)
        if "f1" in ans_metric_set:
            overall["f1"] = round(ans_f1_sum / evaluated, 4)
        if "semantic" in ans_metric_set:
            overall["semantic"] = round(ans_sem_sum / evaluated, 4)
        if args.judge_model:
            overall["judge_llm"] = round(ans_llm_sum / ans_llm_count, 4) if ans_llm_count else None

    # per-question-type aggregates
    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for r in per_sample_rows:
        t = r.get("question_type") or "unknown"
        by_type.setdefault(t, []).append(r)

    per_question_type: Dict[str, Any] = {}
    if args.mode == "retrieval":
        for t, rows in by_type.items():
            n = len(rows)
            rec: Dict[str, Any] = {"n": n}
            rec["single_hit_rate_by_k"] = {
                str(k): round(sum(int(rr.get(f"single@{k}", 0)) for rr in rows) / n, 4) for k in ks
            }
            rec["both_hit_rate_by_k"] = {
                str(k): round(sum(int(rr.get(f"both@{k}", 0)) for rr in rows) / n, 4) for k in ks
            }
            if "mrr" in metric_set:
                rec["mrr_by_k"] = {
                    str(k): round(sum(float(rr.get(f"mrr@{k}", 0.0)) for rr in rows) / n, 4) for k in ks
                }
            if "ndcg" in metric_set:
                rec["ndcg_by_k"] = {
                    str(k): round(sum(float(rr.get(f"ndcg@{k}", 0.0)) for rr in rows) / n, 4) for k in ks
                }
            per_question_type[t] = rec
    else:
        for t, rows in by_type.items():
            n = len(rows)
            rec = {"n": n}
            if "exact" in ans_metric_set:
                rec["exact"] = round(sum(float(rr.get("exact", 0.0)) for rr in rows) / n, 4)
            if "f1" in ans_metric_set:
                rec["f1"] = round(sum(float(rr.get("f1", 0.0)) for rr in rows) / n, 4)
            if "semantic" in ans_metric_set:
                rec["semantic"] = round(sum(float(rr.get("semantic", 0.0)) for rr in rows) / n, 4)
            per_question_type[t] = rec

    summary = {"overall": overall, "per_question_type": per_question_type, "errors": len(errors)}

    manifest = {
        "run_dir": str(run_dir),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cwd": str(Path.cwd()),
        "python": sys.version,
        "env": {"PYTHONPATH": os.environ.get("PYTHONPATH")},
        "params": {
            "base_url": args.base_url,
            "project_id": args.project_id,
            "mode": args.mode,
            "source": args.source,
            "data_dir": args.data_dir,
            "jsonl_path": args.jsonl_path,
            "limit": (args.limit if args.limit > 0 else None),
            "ks": ks,
            "min_score_threshold": args.min_score_threshold,
            "max_per_question_type": args.max_per_question_type,
            "seed": args.seed,
            "strict_id_match": args.strict_id_match,
            "allow_duplicates": args.allow_duplicates,
            "metrics": sorted(metric_set),
            "score_calibration": args.score_calibration,
            "save_topn_text": args.save_topn_text,
            "error_policy": args.error_policy,
            "request_timeout": args.request_timeout,
            "retries": args.retries,
            "judge_model": args.judge_model,
            "answer_metrics": sorted(ans_metric_set),
            "show_failures": args.show_failures,
            "retrieval_profile": args.retrieval_profile,
            "fusion_ce_weight": args.fusion_ce_weight,
            "no_progress": args.no_progress,
        },
        "rag_config": _rag_config_snapshot(),
    }

    (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "failures.json").write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "errors.json").write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")
    with (run_dir / "per_sample.jsonl").open("w", encoding="utf-8") as f:
        for row in per_sample_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    _write_metrics_plot(run_dir, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSaved run artifacts to: {run_dir}")
    if failures:
        print("\nSample failures:", flush=True)
        for f in failures[: args.show_failures]:
            print(json.dumps(f, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
