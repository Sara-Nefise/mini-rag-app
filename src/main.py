
from fastapi import FastAPI
from routes import base, data
from motor.motor_asyncio import AsyncIOMotorClient
from helpers.config import get_settings

app = FastAPI()

@app.on_event("startup")
async def startup_db_client():
    try:
        settings = get_settings()
        app.mongo_conn = AsyncIOMotorClient(settings.MONGODB_URI)
        app.db_client = app.mongo_conn[settings.MONGODB_DB_NAME]
        print(f"Connected to {settings.MONGODB_DB_NAME}")
    except Exception as e:
        print(f"Failed to connect to MongoDB: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_db_client():
    app.mongo_conn.close()


app.include_router(base.base_router)
app.include_router(data.data_router)
