from ..LLMInterface import LLMInterface
from ..LLMEnums import OpenAIEnums
from openai import OpenAI
import json
import logging
import os
from typing import Any, Dict, List, Optional, Union

import httpx


def _openrouter_optional_headers() -> Optional[dict]:
    """OpenRouter recommends Referer + Title on requests (optional)."""
    extra = {}
    ref = os.environ.get("OPENROUTER_HTTP_REFERER") or os.environ.get("OPENROUTER_SITE_URL")
    if ref:
        extra["HTTP-Referer"] = ref
    title = os.environ.get("OPENROUTER_APP_TITLE")
    if title:
        extra["X-Title"] = title
    return extra or None

class OpenAIProvider(LLMInterface):

    def __init__(self, api_key: str, api_url: str=None,
                       default_input_max_characters: int=1000,
                       default_generation_max_output_tokens: int=1000,
                       default_generation_temperature: float=0.1):
        
        self.api_key = api_key
        self.api_url = api_url

        self.default_input_max_characters = default_input_max_characters
        self.default_generation_max_output_tokens = default_generation_max_output_tokens
        self.default_generation_temperature = default_generation_temperature

        self.generation_model_id = None

        self.embedding_model_id = None
        self.embedding_size = None

        client_kw = dict(
            api_key=self.api_key,
            base_url=self.api_url if self.api_url and len(self.api_url) else None,
        )
        hdr = _openrouter_optional_headers()
        if hdr:
            client_kw["default_headers"] = hdr
        self.client = OpenAI(**client_kw)

        self.enums = OpenAIEnums
        self.logger = logging.getLogger(__name__)

    def set_generation_model(self, model_id: str):
        self.generation_model_id = model_id

    def set_embedding_model(self, model_id: str, embedding_size: int):
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size

    def process_text(self, text: str):
        return text[:self.default_input_max_characters].strip()

    def generate_text(self, prompt: str, chat_history: list=[], max_output_tokens: int=None,
                            temperature: float = None):
        
        if not self.client:
            self.logger.error("OpenAI client was not set")
            return None

        if not self.generation_model_id:
            self.logger.error("Generation model for OpenAI was not set")
            return None
        
        max_output_tokens = max_output_tokens if max_output_tokens else self.default_generation_max_output_tokens
        temperature = temperature if temperature else self.default_generation_temperature

        chat_history.append(
            self.construct_prompt(prompt=prompt, role=OpenAIEnums.USER.value)
        )

        response = self.client.chat.completions.create(
            model = self.generation_model_id,
            messages = chat_history,
            max_tokens = max_output_tokens,
            temperature = temperature
        )

        if not response or not response.choices or len(response.choices) == 0 or not response.choices[0].message:
            self.logger.error("Error while generating text with OpenAI")
            return None

        return response.choices[0].message.content


    def _embeddings_http(self, inputs: List[str]) -> List[List[float]]:
        """POST /v1/embeddings via HTTP so malformed bodies (`data: null`) don't crash the SDK parser."""
        base = (self.api_url or "https://api.openai.com/v1").rstrip("/")
        url = f"{base}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        hdr = _openrouter_optional_headers()
        if hdr:
            headers.update(hdr)
        payload: Dict[str, Any] = {
            "model": self.embedding_model_id,
            "input": inputs,
        }
        try:
            with httpx.Client(timeout=httpx.Timeout(120.0)) as http:
                r = http.post(url, headers=headers, json=payload)
        except httpx.HTTPError as e:
            self.logger.error("Embedding HTTP request failed: %s", e)
            raise RuntimeError(f"Embedding HTTP request failed: {e}") from e

        try:
            body: Any = r.json()
        except json.JSONDecodeError:
            snippet = (r.text or "")[:2000]
            self.logger.error(
                "Embeddings response is not JSON. status=%s body=%s",
                r.status_code,
                snippet,
            )
            raise RuntimeError(
                f"Embedding API returned non-JSON (status {r.status_code}). "
                f"First bytes: {snippet!r}"
            ) from None

        if r.status_code >= 400:
            err = body.get("error") if isinstance(body, dict) else None
            msg = err.get("message") if isinstance(err, dict) else None
            tail = json.dumps(body)[:4000] if body is not None else (r.text or "")[:2000]
            self.logger.error(
                "Embeddings API error status=%s message=%s body=%s",
                r.status_code,
                msg,
                tail,
            )
            raise RuntimeError(
                f"Embedding API error (HTTP {r.status_code}): {msg or tail}"
            )

        if not isinstance(body, dict):
            raise RuntimeError(f"Unexpected embeddings JSON shape: {type(body)}")

        err_obj = body.get("error")
        if isinstance(err_obj, dict) and err_obj.get("message"):
            msg = str(err_obj.get("message", ""))
            tail = json.dumps(body)[:4000]
            self.logger.error("Embeddings API returned error in JSON body: %s", tail)
            hint = ""
            if "8192" in msg or "maximum input length" in msg.lower() or "tokens" in msg.lower():
                hint = (
                    " Lower EMBEDDING_MAX_INPUT_TOKENS (e.g. 8000) or EMBEDDING_INPUT_MAX_CHARS "
                    "in .env; ensure tiktoken is installed for accurate token truncation."
                )
            raise RuntimeError(
                f"Embedding API error: {msg}.{hint} Raw (truncated): {tail}"
            )

        raw_data = body.get("data")
        if raw_data is None:
            tail = json.dumps(body)[:4000]
            self.logger.error(
                "Embeddings response `data` is null or missing. Full body (truncated): %s",
                tail,
            )
            raise RuntimeError(
                "Embedding API returned `data: null` or missing `data`. "
                "Use an embedding model ID on your provider (e.g. OpenRouter: "
                "`openai/text-embedding-3-small`), not a chat/completions-only model. "
                f"Provider body (truncated): {tail}"
            )

        if not isinstance(raw_data, list) or len(raw_data) == 0:
            raise RuntimeError("Embedding API returned empty `data` array")

        items = sorted(raw_data, key=lambda x: x.get("index", 0) if isinstance(x, dict) else 0)
        out: List[List[float]] = []
        for i, rec in enumerate(items):
            if not isinstance(rec, dict) or "embedding" not in rec:
                raise RuntimeError(f"Embedding item {i} missing `embedding` field: {rec!r}")
            emb = rec["embedding"]
            if not emb:
                raise RuntimeError(f"Empty embedding at index {i}")
            out.append(emb)
        if len(out) != len(inputs):
            raise RuntimeError(
                f"Embedding count mismatch: expected {len(inputs)} vectors, got {len(out)}"
            )
        return out

    def embed_text(self, text: Union[str, List[str]], document_type: str = None):
        
        if not self.client:
            self.logger.error("OpenAI client was not set")
            return None
        
        if isinstance(text, str):
            text = [text]

        if not self.embedding_model_id:
            self.logger.error("Embedding model for OpenAI was not set")
            return None
        
        # Avoid OpenAI SDK response parser: some proxies return 200 with `data: null`, which
        # triggers TypeError inside embeddings.create() before we can inspect the response.
        return self._embeddings_http(text)

    def construct_prompt(self, prompt: str, role: str):
        return {
            "role": role,
            "content": prompt,
        }
    


    

