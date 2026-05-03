"""
Drop all pgvector embedding tables (names starting with collection_).

Usage:
  cd src && .venv/bin/python scripts/clear_pgvector_stores.py

Does not delete rows in `chunks` or any other app tables — only vector store tables.
"""
from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parent.parent
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from helpers.config import get_settings
from stores.vectordb.VectorDBEnums import VectorDBEnums
from stores.vectordb.VectorDBProviderFactory import VectorDBProviderFactory

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


async def main() -> int:
    settings = get_settings()
    if str(settings.VECTOR_DB_BACKEND).strip().upper() != VectorDBEnums.PGVECTOR.value:
        logger.error(
            "VECTOR_DB_BACKEND is not PGVECTOR (got %r). This script only drops Postgres pgvector tables.",
            settings.VECTOR_DB_BACKEND,
        )
        return 1

    postgres_conn = (
        f"postgresql+asyncpg://{settings.POSTGRES_USERNAME}:{settings.POSTGRES_PASSWORD}"
        f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_MAIN_DATABASE}"
    )
    engine = create_async_engine(postgres_conn)
    db_client = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    vf = VectorDBProviderFactory(config=settings, db_client=db_client)
    client = vf.create(provider=settings.VECTOR_DB_BACKEND)
    if client is None:
        return 1

    await client.connect()
    try:
        n = await client.drop_all_project_collections()
        logger.info("Dropped %s pgvector table(s).", n)
    finally:
        await client.disconnect()
        await engine.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
