from logging.config import fileConfig

from sqlalchemy import engine_from_config
from sqlalchemy import pool
from schemes import SQLAlchemyBase
from alembic import context
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
import asyncio

# Load config from alembic.ini


# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config
connectable = create_async_engine(
    config.get_main_option("sqlalchemy.url"),
    poolclass=pool.NullPool,  # Avoid pooling issues with async
)
# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
# from myapp import mymodel
# target_metadata = mymodel.Base.metadata
target_metadata = SQLAlchemyBase.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        pass  # Add logic here if needed

async def run_migrations_online():
    """Run migrations in 'online' mode."""
    async with connectable.connect() as connection:
        # Configure the context to use the async connection
        await connection.run_sync(
            lambda sync_conn: context.configure(
                connection=sync_conn,
                target_metadata=target_metadata,
                render_as_batch=True,  # Useful for PostgreSQL
            )
        )
        # Run migrations synchronously within the async connection
        async with connection.begin():
            await connection.run_sync(lambda sync_conn: context.run_migrations())

if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())