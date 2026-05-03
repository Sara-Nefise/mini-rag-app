"""
Rebuild vector embeddings for all projects.

Usage (from repo root or src; .env must be next to cwd as your app expects):

  cd src && .venv/bin/python scripts/rebuild_all_embeddings.py --mode full
  cd src && .venv/bin/python scripts/rebuild_all_embeddings.py --mode reindex-only

Modes:
  reindex-only — Drop all pgvector/Qdrant collection_* stores, then re-embed every chunk
                 already in Postgres (no file re-chunking).

  full         — Same as above, plus DELETE all rows from `chunks`, then re-run processing
                 from stored assets on disk for every project (same as POST /data/process
                 with do_reset for each project), then embed.

Requires network/API keys if embedding uses remote providers (OpenAI, etc.).
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Run as `python scripts/rebuild_all_embeddings.py` from `src/`
_SRC_ROOT = Path(__file__).resolve().parent.parent
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from controllers.NLPController import NLPController
from controllers.ProcessController import ProcessController
from helpers.config import get_settings
from models.AssetModel import AssetModel
from models.ChunkModel import ChunkModel
from models.ProjectModel import ProjectModel
from models.db_schemes import DataChunk
from models.enums.AssetTypeEnum import AssetTypeEnum
from stores.llm.LLMProviderFactory import LLMProviderFactory
from stores.llm.templates.template_parser import TemplateParser
from stores.vectordb.VectorDBProviderFactory import VectorDBProviderFactory
from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def embed_project(
    *,
    request_like_sessionmaker,
    vectordb_client,
    embedding_client,
    template_parser,
    project,
    do_reset: bool,
) -> int:
    """Mirror routes/nlp.py index_project pagination."""
    nlp_controller = NLPController(
        vectordb_client=vectordb_client,
        generation_client=None,
        embedding_client=embedding_client,
        template_parser=template_parser,
    )
    chunk_model = await ChunkModel.create_instance(db_client=request_like_sessionmaker)

    collection_name = nlp_controller.create_collection_name(project_id=project.project_id)
    await vectordb_client.create_collection(
        collection_name=collection_name,
        embedding_size=embedding_client.embedding_size,
        do_reset=do_reset,
    )

    total = await chunk_model.get_total_chunks_count(project_id=project.project_id)
    if total == 0:
        logger.info("Project %s: no chunks, skipping index.", project.project_id)
        return 0

    has_records = True
    page_no = 1
    inserted = 0
    pbar = tqdm(total=total, desc=f"Index project {project.project_id}", position=0)

    while has_records:
        page_chunks = await chunk_model.get_poject_chunks(
            project_id=project.project_id, page_no=page_no
        )
        if len(page_chunks):
            page_no += 1

        if not page_chunks:
            has_records = False
            break

        chunks_ids = [c.chunk_id for c in page_chunks]
        ok = await nlp_controller.index_into_vector_db(
            project=project,
            chunks=page_chunks,
            do_reset=False,
            chunks_ids=chunks_ids,
        )
        if not ok:
            raise RuntimeError(f"index_into_vector_db failed for project {project.project_id}")
        inserted += len(page_chunks)
        pbar.update(len(page_chunks))

    pbar.close()
    return inserted


async def rechunk_project(
    *,
    db_sessionmaker,
    project_id: int,
    chunk_size: int,
    overlap_size: int,
) -> int:
    """Rebuild rows in `chunks` from files on disk for one project."""
    project_model = await ProjectModel.create_instance(db_client=db_sessionmaker)
    asset_model = await AssetModel.create_instance(db_client=db_sessionmaker)
    chunk_model = await ChunkModel.create_instance(db_client=db_sessionmaker)

    project = await project_model.get_project_or_create_one(project_id=project_id)
    assets = await asset_model.get_all_project_assets(
        asset_project_id=project.project_id,
        asset_type=AssetTypeEnum.FILE.value,
    )
    if not assets:
        return 0

    pc = ProcessController(project_id=project_id)
    inserted = 0
    for asset in assets:
        file_content = pc.get_file_content(file_id=asset.asset_name)
        if file_content is None:
            logger.warning(
                "Skip missing/unreadable file asset_name=%s project=%s",
                asset.asset_name,
                project_id,
            )
            continue
        file_chunks = pc.process_file_content(
            file_content=file_content,
            file_id=asset.asset_name,
            chunk_size=chunk_size,
            overlap_size=overlap_size,
        )
        if not file_chunks:
            raise RuntimeError(
                f"No chunks produced for file {asset.asset_name} (project {project_id})"
            )
        records = [
            DataChunk(
                chunk_text=c.page_content,
                chunk_metadata=c.metadata,
                chunk_order=i + 1,
                chunk_project_id=project.project_id,
                chunk_asset_id=asset.asset_id,
            )
            for i, c in enumerate(file_chunks)
        ]
        inserted += await chunk_model.insert_many_chunks(chunks=records)
    return inserted


async def async_main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild embeddings for all projects.")
    parser.add_argument(
        "--mode",
        choices=("reindex-only", "full"),
        default="reindex-only",
        help="full = truncate chunks + re-chunk from assets + embed; "
        "reindex-only = embed existing chunks only.",
    )
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--overlap-size", type=int, default=None)
    args = parser.parse_args()

    settings = get_settings()
    chunk_size = args.chunk_size or settings.FILE_DEFAULT_CHUNK_SIZE
    overlap_size = args.overlap_size or 20

    postgres_conn = (
        f"postgresql+asyncpg://{settings.POSTGRES_USERNAME}:{settings.POSTGRES_PASSWORD}"
        f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_MAIN_DATABASE}"
    )
    engine = create_async_engine(postgres_conn)
    db_client = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    vectordb_factory = VectorDBProviderFactory(config=settings, db_client=db_client)
    vectordb_client = vectordb_factory.create(provider=settings.VECTOR_DB_BACKEND)
    if vectordb_client is None:
        logger.error("VECTOR_DB_BACKEND must be QDRANT or PGVECTOR.")
        return 1

    llm_factory = LLMProviderFactory(settings)
    embedding_client = llm_factory.create(
        provider=settings.EMBEDDING_BACKEND,
        embedding=True,
    )
    if embedding_client is None:
        logger.error("EMBEDDING_BACKEND not configured.")
        return 1
    embedding_client.set_embedding_model(
        model_id=settings.EMBEDDING_MODEL_ID,
        embedding_size=settings.EMBEDDING_MODEL_SIZE,
    )

    template_parser = TemplateParser(
        language=settings.PRIMARY_LANG,
        default_language=settings.DEFAULT_LANG,
    )

    await vectordb_client.connect()

    try:
        dropped = await vectordb_client.drop_all_project_collections()
        logger.info("Dropped %s vector collection(s).", dropped)

        chunk_model = await ChunkModel.create_instance(db_client=db_client)
        project_model = await ProjectModel.create_instance(db_client=db_client)

        if args.mode == "full":
            deleted = await chunk_model.delete_all_chunks()
            logger.info("Deleted %s chunk row(s) from Postgres.", deleted)

            project_ids = await project_model.list_all_project_ids()
            total_rechunked = 0
            for pid in project_ids:
                n = await rechunk_project(
                    db_sessionmaker=db_client,
                    project_id=pid,
                    chunk_size=chunk_size,
                    overlap_size=overlap_size,
                )
                if n:
                    logger.info("Project %s: inserted %s chunk row(s).", pid, n)
                total_rechunked += n
            logger.info("Total chunk rows inserted after re-processing: %s", total_rechunked)

        project_ids = await project_model.list_all_project_ids()
        grand_total = 0
        for pid in project_ids:
            project = await project_model.get_project_or_create_one(project_id=pid)
            n = await embed_project(
                request_like_sessionmaker=db_client,
                vectordb_client=vectordb_client,
                embedding_client=embedding_client,
                template_parser=template_parser,
                project=project,
                do_reset=True,
            )
            grand_total += n
            if n:
                logger.info("Project %s: indexed %s vector row(s).", pid, n)

        logger.info("Done. Total vector rows inserted: %s", grand_total)
        return 0
    finally:
        await vectordb_client.disconnect()
        await engine.dispose()


def main() -> None:
    raise SystemExit(asyncio.run(async_main()))


if __name__ == "__main__":
    main()
