import os

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Optional

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    APP_NAME: str
    APP_VERSION: str
    # OPENAI_API_KEY: str

    FILE_ALLOWED_TYPES: list
    FILE_MAX_SIZE: int
    FILE_DEFAULT_CHUNK_SIZE: int

    POSTGRES_USERNAME: str
    POSTGRES_PASSWORD: str
    POSTGRES_HOST: str
    POSTGRES_PORT: int
    POSTGRES_MAIN_DATABASE: str

    GENERATION_BACKEND: str
    EMBEDDING_BACKEND: str

    OPENAI_API_KEY: Optional[str] = None
    OPENAI_API_EMBEDDING_URL: Optional[str] = None
    OPENAI_API_URL: Optional[str] = None
    COHERE_API_KEY: Optional[str] = None
    #: Hugging Face Hub (datasets, gated models); also accepts lowercase `hf_token` in .env.
    HF_TOKEN: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("HF_TOKEN", "hf_token"),
    )

    GENERATION_MODEL_ID_LITERAL: Optional[List[str]] = None
    GENERATION_MODEL_ID: Optional[str] = None
    EMBEDDING_MODEL_ID: Optional[str] = None
    EMBEDDING_MODEL_SIZE: Optional[int] = None
    #: Fallback cap when token truncation is unavailable (keep below provider token limits).
    EMBEDDING_INPUT_MAX_CHARS: int = 12000
    #: Hard cap in tokens (OpenAI/OpenRouter embedding models typically max 8192 tokens).
    EMBEDDING_MAX_INPUT_TOKENS: int = 8000
    #: OpenRouter / some providers reject large embedding batches; keep conservative.
    EMBEDDING_BATCH_SIZE: int = 8
    #: For EMBEDDING_BACKEND=SENTENCE_TRANSFORMERS: "cpu", "cuda", "cuda:0", etc. None = library default.
    EMBEDDING_DEVICE: Optional[str] = None
    INPUT_DAFAULT_MAX_CHARACTERS: Optional[int] = None
    GENERATION_DAFAULT_MAX_TOKENS: Optional[int] = None
    GENERATION_DAFAULT_TEMPERATURE: Optional[float] = None

    VECTOR_DB_BACKEND_LITERAL: Optional[List[str]] = None
    VECTOR_DB_BACKEND : str
    VECTOR_DB_PATH : str
    VECTOR_DB_DISTANCE_METHOD: Optional[str] = None
    VECTOR_DB_PGVEC_INDEX_THRESHOLD: int = 100

    #: Cross-encoder reranker (sentence-transformers). Enable + use retrieval_profile *\_xe.
    RERANKER_ENABLED: bool = False
    RERANKER_MODEL_ID: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    #: Score this many candidates from the dense+RRF pool (then return top ``limit``).
    RERANKER_POOL_SIZE: int = 48
    RERANKER_MAX_INPUT_CHARS: int = 1800
    RERANKER_BATCH_SIZE: int = 16
    #: Device for cross-encoder only. Default ``cpu`` avoids MPS OOM alongside embedding models on Apple Silicon.
    RERANKER_DEVICE: str = "cpu"
    #: 0 = trust dense+RRF only; 1 = trust cross-encoder only. Lower is safer for non-English / domain mismatch.
    RERANKER_FUSION_CE_WEIGHT: float = 0.35

    PRIMARY_LANG: str = "en"
    DEFAULT_LANG: str = "en"
    FIREBASE_CREDENTIALS_PATH: Optional[str] = None
    FIREBASE_API_KEY: Optional[str] = None

def get_settings() -> Settings:
    settings = Settings()
    # huggingface_hub / transformers read HF_TOKEN from os.environ, not from Pydantic.
    token = settings.HF_TOKEN
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
    return settings
