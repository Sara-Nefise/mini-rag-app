"""
Load a manifest JSONL (each line: ``{"id": "...", "text": "..."}``), insert each line as a
``DataChunk``, embed with the app embedding client, and write vectors to pgvector/Qdrant.

**Embed the entire manifest (no row limit)**::

    cd src && .venv/bin/python scripts/ingest_mlsum_to_vectordb.py \\
      --manifest ../data/mlsum-tu/manifest.jsonl \\
      --project-id 100 \\
      --replace \\
      --batch-chunks 64

``--replace`` drops the project's vector collection and deletes existing ``chunks`` rows for that
project first (recommended before a full re-ingest).

Smoke test::

    ... --max-rows 500

Paths resolve from **cwd** or **repo root** (parent of ``src``).

Requires ``VECTOR_DB_BACKEND``, Postgres, and ``EMBEDDING_*`` in ``src/.env``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker


def _repo_src() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_env() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(_repo_src() / ".env")
    except ImportError:
        pass


def _project_root() -> Path:
    return _repo_src().parent


def _resolve_manifest_path(manifest: str) -> Path:
    """
    Resolve manifest path whether you run from ``src/`` (``../data/...``) or pass
    ``data/...`` relative to the repo root. Avoids joining ``../`` onto project root
    (which would escape ``mini-rag-app``).
    """
    p = Path(manifest)
    if p.is_absolute():
        if p.is_file():
            return p
        print(f"ERROR: manifest not found: {p}", file=sys.stderr)
        sys.exit(1)

    cwd_path = (Path.cwd() / p).resolve()
    root_path = (_project_root() / p).resolve()
    for c in (cwd_path, root_path):
        if c.is_file():
            return c
    print(
        "ERROR: manifest not found. Tried:\n"
        f"  {cwd_path}\n"
        f"  {root_path}\n"
        "Use e.g.  --manifest data/mlsum-tu/manifest.jsonl  "
        "(from mini-rag-app) or an absolute path.",
        file=sys.stderr,
    )
    sys.exit(1)


def _load_manifest_rows(
    manifest_path: Path,
    max_rows: Optional[int],
) -> List[Tuple[str, str]]:
    """Return list of (mlsum_id, text)."""
    rows: List[Tuple[str, str]] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            kid = str(obj["id"]).strip()
            text = str(obj["text"])
            rows.append((kid, text))
            if max_rows is not None and len(rows) >= max_rows:
                break
    return rows


async def _async_main(argv: Optional[Sequence[str]]) -> int:
    _load_env()
    os.chdir(_repo_src())

    parser = argparse.ArgumentParser(description="Ingest MLSUM manifest into vector DB + chunks table.")
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/mlsum-tu/manifest.jsonl",
        help="Path to manifest.jsonl (project-root-relative or absolute).",
    )
    parser.add_argument(
        "--project-id",
        type=int,
        default=100,
        help="Integer project id for this corpus (e.g. 100). Use a dedicated id for MLSUM.",
    )
    parser.add_argument(
        "--batch-chunks",
        type=int,
        default=64,
        help="How many manifest lines per DB + embed + insert batch.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Stop after N manifest lines (debug).",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop existing vector collection for this project only (does not delete Postgres chunks).",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Drop vector collection AND delete all chunks for --project-id (clean slate, then ingest).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    from helpers.config import get_settings
    from controllers.NLPController import NLPController
    from models.db_schemes import Asset, DataChunk, Project
    from models.AssetModel import AssetModel
    from models.ChunkModel import ChunkModel
    from models.ProjectModel import ProjectModel
    from stores.llm.LLMProviderFactory import LLMProviderFactory
    from stores.llm.templates.template_parser import TemplateParser
    from stores.vectordb.VectorDBProviderFactory import VectorDBProviderFactory
    from models.enums.AssetTypeEnum import AssetTypeEnum

    settings = get_settings()
    if not settings.VECTOR_DB_BACKEND or not str(settings.VECTOR_DB_BACKEND).strip():
        print("ERROR: Set VECTOR_DB_BACKEND=PGVECTOR or QDRANT in src/.env", file=sys.stderr)
        return 1

    man_path = _resolve_manifest_path(args.manifest)

    postgres_conn = (
        f"postgresql+asyncpg://{settings.POSTGRES_USERNAME}:{settings.POSTGRES_PASSWORD}"
        f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_MAIN_DATABASE}"
    )

    engine = create_async_engine(postgres_conn, echo=False)
    db_client = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    vectordb_factory = VectorDBProviderFactory(config=settings, db_client=db_client)
    vectordb = vectordb_factory.create(provider=settings.VECTOR_DB_BACKEND)
    if vectordb is None:
        print("ERROR: VECTOR_DB_BACKEND invalid.", file=sys.stderr)
        await engine.dispose()
        return 1

    llm_factory = LLMProviderFactory(settings)
    emb = llm_factory.create(provider=settings.EMBEDDING_BACKEND, embedding=True)
    if emb is None:
        print("ERROR: EMBEDDING_BACKEND invalid.", file=sys.stderr)
        await engine.dispose()
        return 1
    emb.set_embedding_model(
        model_id=settings.EMBEDDING_MODEL_ID,
        embedding_size=int(settings.EMBEDDING_MODEL_SIZE),
    )

    gen = llm_factory.create(provider=settings.GENERATION_BACKEND)
    if gen is not None and settings.GENERATION_MODEL_ID:
        gen.set_generation_model(settings.GENERATION_MODEL_ID)

    template_parser = TemplateParser(
        language=settings.PRIMARY_LANG,
        default_language=settings.DEFAULT_LANG,
    )
    nlp = NLPController(
        vectordb_client=vectordb,
        generation_client=gen,
        embedding_client=emb,
        template_parser=template_parser,
    )

    await vectordb.connect()

    collection_name = nlp.create_collection_name(project_id=str(args.project_id))
    if args.replace:
        print(f"Replacing corpus: drop {collection_name} + delete chunks for project_id={args.project_id}")
        await vectordb.delete_collection(collection_name=collection_name)
        chunk_model_wipe = await ChunkModel.create_instance(db_client)
        removed = await chunk_model_wipe.delete_chunks_by_project_id(project_id=args.project_id)
        print(f"Removed {removed} chunk row(s) from Postgres.")
    elif args.reset:
        print(f"Dropping collection {collection_name} (--reset)")
        await vectordb.delete_collection(collection_name=collection_name)

    rows = _load_manifest_rows(man_path, args.max_rows)
    if not rows:
        print("No manifest rows loaded.", file=sys.stderr)
        await vectordb.disconnect()
        await engine.dispose()
        return 1

    print(f"Loaded {len(rows)} manifest rows from {man_path}")

    project_model = await ProjectModel.create_instance(db_client)
    project = await project_model.get_project_or_create_one(project_id=args.project_id)
    if not isinstance(project, Project):
        print("ERROR: could not load/create project.", file=sys.stderr)
        await vectordb.disconnect()
        await engine.dispose()
        return 1

    asset_model = await AssetModel.create_instance(db_client)
    existing = await asset_model.get_asset_record(
        asset_project_id=project.project_id,
        asset_name="mlsum_manifest",
    )
    if existing:
        asset = existing
    else:
        asset = Asset(
            asset_project_id=project.project_id,
            asset_type=AssetTypeEnum.FILE.value,
            asset_name="mlsum_manifest",
            asset_size=0,
            asset_config={"source": "reciTAL/mlsum", "config": "tu"},
        )
        asset = await asset_model.create_asset(asset)

    global_order = 0
    for start in range(0, len(rows), args.batch_chunks):
        batch = rows[start : start + args.batch_chunks]
        chunk_objs: List[DataChunk] = []
        for kid, text in batch:
            global_order += 1
            chunk_objs.append(
                DataChunk(
                    chunk_text=text,
                    chunk_metadata={"mlsum_id": kid},
                    chunk_order=global_order,
                    chunk_project_id=project.project_id,
                    chunk_asset_id=asset.asset_id,
                )
            )

        async with db_client() as session:
            async with session.begin():
                session.add_all(chunk_objs)
                await session.flush()
                chunk_ids = [c.chunk_id for c in chunk_objs]

        await nlp.index_into_vector_db(
            project=project,
            chunks=chunk_objs,
            chunks_ids=chunk_ids,
            do_reset=False,
        )

        print(f"Indexed batch rows {start + 1}-{start + len(batch)} / {len(rows)}")

    await vectordb.disconnect()
    await engine.dispose()
    print(
        f"Done. Search this corpus via API using project_id={project.project_id} "
        f"(collection {collection_name})."
    )
    return 0


def main() -> int:
    return asyncio.run(_async_main(sys.argv[1:]))


if __name__ == "__main__":
    raise SystemExit(main())
