"""
Drop vector-indexed data and document chunks so a new embedding model can re-index cleanly.

- PostgreSQL + PGVECTOR: DROP all public tables named collection_* (pgvector collections).
- QDRANT: delete all collections under VECTOR_DB_PATH.
- Relational: TRUNCATE chunks + assets (optional: messages + chats).

Keeps: users, projects (unless --with-chats which clears chats/messages tied to projects).

Usage (from repo ``src/``)::

    export PYTHONPATH=.
    python -m scripts.reset_rag_data --dry-run
    python -m scripts.reset_rag_data
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker


def _repo_src() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_env():
    try:
        from dotenv import load_dotenv

        load_dotenv(_repo_src() / ".env")
    except ImportError:
        pass


async def _reset_pgvector_collections(session, dry_run: bool) -> List[str]:
    r = await session.execute(
        text(
            """
            SELECT tablename FROM pg_tables
            WHERE schemaname = 'public'
              AND tablename LIKE 'collection_%'
            ORDER BY tablename
            """
        )
    )
    names = [row[0] for row in r.fetchall()]
    for tbl in names:
        if dry_run:
            print(f"  [dry-run] DROP TABLE IF EXISTS {tbl} CASCADE")
        else:
            await session.execute(text(f'DROP TABLE IF EXISTS "{tbl}" CASCADE'))
    return names


def _reset_qdrant_collections(settings, dry_run: bool) -> None:
    from controllers.BaseController import BaseController
    from qdrant_client import QdrantClient
    from stores.vectordb.VectorDBEnums import VectorDBEnums

    if settings.VECTOR_DB_BACKEND.upper() != VectorDBEnums.QDRANT.value:
        return
    base = BaseController()
    path = base.get_database_path(settings.VECTOR_DB_PATH)
    client = QdrantClient(path=path)
    cols = client.get_collections().collections
    for c in cols:
        name = c.name
        if dry_run:
            print(f"  [dry-run] delete Qdrant collection {name!r}")
        else:
            client.delete_collection(collection_name=name)
            print(f"  deleted Qdrant collection {name!r}")


async def _truncate_relational(session, with_chats: bool, dry_run: bool) -> None:
    if dry_run:
        print("  [dry-run] TRUNCATE chunks, assets RESTART IDENTITY CASCADE")
        if with_chats:
            print("  [dry-run] TRUNCATE messages; TRUNCATE chats RESTART IDENTITY CASCADE")
        return
    await session.execute(text("TRUNCATE TABLE chunks RESTART IDENTITY CASCADE"))
    await session.execute(text("TRUNCATE TABLE assets RESTART IDENTITY CASCADE"))
    if with_chats:
        await session.execute(text("TRUNCATE TABLE messages RESTART IDENTITY CASCADE"))
        await session.execute(text("TRUNCATE TABLE chats RESTART IDENTITY CASCADE"))


async def async_main(argv: Optional[Sequence[str]]) -> int:
    _load_env()
    os.chdir(_repo_src())

    parser = argparse.ArgumentParser(description="Reset RAG vectors + chunks/assets in Postgres/Qdrant.")
    parser.add_argument("--dry-run", action="store_true", help="Print actions only.")
    parser.add_argument(
        "--skip-relational",
        action="store_true",
        help="Only remove vector collections; keep chunks/assets rows.",
    )
    parser.add_argument(
        "--with-chats",
        action="store_true",
        help="Also TRUNCATE messages and chats.",
    )
    args = parser.parse_args(argv)

    from helpers.config import get_settings
    from stores.vectordb.VectorDBEnums import VectorDBEnums

    settings = get_settings()

    postgres_conn = (
        f"postgresql+asyncpg://{settings.POSTGRES_USERNAME}:{settings.POSTGRES_PASSWORD}"
        f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_MAIN_DATABASE}"
    )

    print(f"VECTOR_DB_BACKEND={settings.VECTOR_DB_BACKEND}")
    print(f"EMBEDDING_MODEL_SIZE={settings.EMBEDDING_MODEL_SIZE}")

    engine = create_async_engine(postgres_conn, echo=False)
    session_factory = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    try:
        _reset_qdrant_collections(settings, args.dry_run)

        async with session_factory() as session:
            async with session.begin():
                print("PostgreSQL: dropping collection_* vector tables...")
                dropped = await _reset_pgvector_collections(session, args.dry_run)
                if not args.dry_run:
                    for t in dropped:
                        print(f"  dropped table {t}")
                elif not dropped:
                    print("  (no collection_* tables)")

                if not args.skip_relational:
                    print("PostgreSQL: truncating chunks + assets...")
                    await _truncate_relational(session, args.with_chats, args.dry_run)
                else:
                    print("Skipping relational TRUNCATE (--skip-relational).")

        if not args.dry_run:
            print("Done.")
        else:
            print("Dry-run finished (no changes).")
        return 0
    finally:
        await engine.dispose()


def main() -> int:
    return asyncio.run(async_main(sys.argv[1:]))


if __name__ == "__main__":
    raise SystemExit(main())
