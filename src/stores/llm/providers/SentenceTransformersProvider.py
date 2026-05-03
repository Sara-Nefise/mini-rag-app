from __future__ import annotations

import logging
from typing import List, Optional, Union

import numpy as np

from ..LLMInterface import LLMInterface
from ..LLMEnums import DocumentTypeEnum, OpenAIEnums

# intfloat E5 (incl. multilingual-e5) expects these prefixes for retrieval.
_E5_QUERY_PREFIX = "query: "
_E5_PASSAGE_PREFIX = "passage: "


class SentenceTransformersProvider(LLMInterface):
    """Local Hugging Face / sentence-transformers models (e.g. intfloat/multilingual-e5-large)."""

    def __init__(
        self,
        default_input_max_characters: int = 1000,
        default_generation_max_output_tokens: int = 1000,
        default_generation_temperature: float = 0.1,
        device: Optional[str] = None,
    ):
        self.default_input_max_characters = default_input_max_characters
        self.default_generation_max_output_tokens = default_generation_max_output_tokens
        self.default_generation_temperature = default_generation_temperature
        self.device = device

        self.generation_model_id = None
        self.embedding_model_id: Optional[str] = None
        self.embedding_size: Optional[int] = None

        self._st_model = None
        self._loaded_id: Optional[str] = None

        self.enums = OpenAIEnums
        self.logger = logging.getLogger(__name__)

    def set_generation_model(self, model_id: str):
        self.generation_model_id = model_id

    def set_embedding_model(self, model_id: str, embedding_size: int):
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size
        if self._loaded_id != model_id:
            self._st_model = None
            self._loaded_id = None

    def _ensure_model(self):
        if not self.embedding_model_id:
            return
        if self._st_model is not None and self._loaded_id == self.embedding_model_id:
            return
        from sentence_transformers import SentenceTransformer

        self.logger.info("Loading sentence-transformers model %s ...", self.embedding_model_id)
        self._st_model = SentenceTransformer(
            self.embedding_model_id,
            device=self.device if self.device else None,
        )
        self._loaded_id = self.embedding_model_id

    @staticmethod
    def _e5_prefix(document_type: Optional[str]) -> str:
        if document_type == DocumentTypeEnum.QUERY.value:
            return _E5_QUERY_PREFIX
        return _E5_PASSAGE_PREFIX

    def process_text(self, text: str) -> str:
        return text[: self.default_input_max_characters].strip()

    def generate_text(
        self,
        prompt: str,
        chat_history: list = None,
        max_output_tokens: int = None,
        temperature: float = None,
    ):
        self.logger.error("SentenceTransformersProvider does not support text generation")
        return None

    def embed_text(self, text: Union[str, List[str]], document_type: str = None):
        self._ensure_model()
        if not self._st_model:
            self.logger.error("Embedding model is not set or failed to load")
            return None
        if isinstance(text, str):
            text = [text]
        prefix = self._e5_prefix(document_type)
        inputs = [prefix + t for t in text]
        vecs = self._st_model.encode(
            inputs,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=min(32, max(1, len(inputs))),
        )
        if not isinstance(vecs, np.ndarray):
            vecs = np.asarray(vecs)
        out = vecs.tolist()
        if self.embedding_size and out and len(out[0]) != self.embedding_size:
            self.logger.warning(
                "Vector dim %s != EMBEDDING_MODEL_SIZE=%s; fix .env EMBEDDING_MODEL_SIZE",
                len(out[0]),
                self.embedding_size,
            )
        return out

    def construct_prompt(self, prompt: str, role: str):
        return {"role": role, "content": prompt}
