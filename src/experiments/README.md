# Turkuaz-RAG experiments (thesis benchmark harness)

This folder adds an **offline evaluation pipeline** for multi-context retrieval on `eneSadi/turkuaz-rag`, separate from the FastAPI service.

## How the production RAG stack fits together

| Step | Location | Role |
|------|----------|------|
| Chunk files → DB | [`routes/data.py`](../routes/data.py), [`ProcessController.py`](../controllers/ProcessController.py) | Load TXT/PDF, `RecursiveCharacterTextSplitter` |
| Embed + vector index | [`NLPController.index_into_vector_db`](../controllers/NLPController.py) | Embeddings via `embedding_client`, insert into Qdrant/pgvector |
| Retrieve | [`NLPController.search_vector_db_collection`](../controllers/NLPController.py) | Single query embedding, **top-k** cosine similarity |
| Generate | [`NLPController.answer_rag_question`](../controllers/NLPController.py) | Concatenate retrieved chunk texts into one prompt |

## Extension points used by benchmarks

- **Dense retrieval**: same embedding model as `embedding_client` (OpenAI/Cohere via [`LLMProviderFactory`](../stores/llm/LLMProviderFactory.py)).
- **Hybrid / rerank / fusion**: implemented in-process over a **deduplicated corpus** built from benchmark contexts (see `corpus.py`), without requiring the live DB.
- **Metrics**: `Recall@k` with **single-news** vs **both-news** hits, aligned with the Turkuaz-RAG paper.

## Download dataset to disk

From `src/` (needs `huggingface_hub`, optional `HF_TOKEN` for rate limits):

```bash
cd src
export PYTHONPATH=.
python -m experiments.download_dataset
```

Files are written to **`../data/turkuaz-rag/`** (project root). Custom output:

```bash
python -m experiments.download_dataset --out /path/to/turkuaz-rag
```

Then run benchmarks **offline from disk** (no Hub fetch during `runner`):

```bash
python -m experiments.runner --source local --systems all
```

Default `--data-dir` is `data/turkuaz-rag` relative to the **project root**. Override with `--data-dir other/path` if you used `--out` when downloading.

## Scenario 2: MLSUM full corpus + Turkuaz-RAG as test set

This matches the thesis setup: **retrieval index = Turkish MLSUM** (`reciTAL/mlsum`, config `tu`), **evaluation queries + gold ids = Turkuaz-RAG**.

1. **Build MLSUM manifest** (large file; needs **`datasets` 2.x** — HF dropped legacy loading scripts in v3+).  
   If you see `Dataset scripts are no longer supported, but found mlsum.py`:
   ```bash
   pip install 'datasets>=2.16,<3.0'
   ```
   On macOS without `python` on PATH, use **`python3`** once the venv is active.

   If you hit **`HTTP server doesn't support range requests`** when building the manifest: the script defaults to **non-streaming** load (`streaming=False`) so data is cached normally — **do not pass `--streaming`**.

   (Large file; if Python 3.14 breaks numpy/pandas, use Python 3.11/3.12 venv.)

```bash
cd src
export PYTHONPATH=.
python -m experiments.build_mlsum_manifest
```

Outputs **`../data/mlsum-tu/manifest.jsonl`** — one JSON object per line: `{"id": "<train_row_index>", "text": "..."}`.  
Turkuaz CSV fields **`1st_news_id`** / **`2nd_news_id`** are matched to these **`id`** strings.

Optional cap for debugging only (will break recall if gold ids fall outside the range):

```bash
python -m experiments.build_mlsum_manifest --max-rows 200000
```

2. **Run the benchmark** with Turkuaz local/HF source **and** MLSUM corpus mode:

```bash
python -m experiments.runner \
  --corpus-mode mlsum \
  --mlsum-manifest data/mlsum-tu/manifest.jsonl \
  --source local \
  --data-dir data/turkuaz-rag \
  --systems dense_topk \
  --limit 50
```

Embedding **~250k** articles costs time and API credits; start with `--limit` on Turkuaz rows and a **full** manifest (do not use `--mlsum-max-docs` unless it still covers every gold `news_id` you evaluate).

3. **Closed pool (previous behaviour)** — default:

```bash
python -m experiments.runner --corpus-mode closed --source local --data-dir data/turkuaz-rag
```

## Running

From repository root (after installing deps and configuring `.env` under `src/`):

```bash
cd src
export PYTHONPATH=.
python -m experiments.runner --help
```

Example (small smoke run using bundled sample; **no API keys** — deterministic embeddings):

```bash
python -m experiments.runner --source jsonl --jsonl-path experiments/fixtures/sample_eval.jsonl \
  --mock-embeddings --systems all --limit 10
```

Then aggregate by question type and export thesis tables:

```bash
RUN=$(ls -td experiments/results/* | head -1)
python -m experiments.analyze --run-dir "$RUN"
python -m experiments.report --run-dir "$RUN"
```

The Hub dataset is a CSV with columns such as `1st_news`, `2nd_news`, `question`, `answer`, `question_type` (not a single `contexts` column). The loader supports both that layout and local JSONL with a `contexts` field.

Optional: set `HF_TOKEN` in the environment for higher Hub rate limits (see Hugging Face docs).

If `datasets` fails on **Python 3.14** with `numpy` / `pandas` import errors, the loader automatically falls back to downloading the dataset ZIP via `huggingface_hub` (stdlib CSV only). For a fully stable stack, use **Python 3.11 or 3.12** in your venv.

Full dataset (downloads `eneSadi/turkuaz-rag` via Hugging Face `datasets`; requires network):

```bash
python -m experiments.runner --source huggingface --limit 100 --systems all
```

Analyze a completed run:

```bash
python -m experiments.analyze --run-dir experiments/results/<run_id>
```

Reports (CSV + Markdown summary):

```bash
python -m experiments.report --run-dir experiments/results/<run_id>
```

## Corpus note

Indexing the entire Turkish MLSUM corpus (~250k articles) matches the paper’s setting but is heavy. The default benchmark builds the retrieval pool from **unique gold news texts** appearing in Turkuaz-RAG (fair relative comparison of retrieval strategies on the same closed pool). Optional `--include-mlsum` can extend the pool when configured.
