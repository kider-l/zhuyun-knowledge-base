import base64
import hashlib
import math
import mimetypes
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import httpx

from app.config import get_settings


def normalize_vector(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return [value / norm for value in values]


def hash_embedding(text: str, dim: int) -> list[float]:
    vector = [0.0] * dim
    tokens = list(text) if text else [" "]
    for index, token in enumerate(tokens):
        digest = hashlib.blake2b(f"{index}:{token}".encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[bucket] += sign
    return normalize_vector(vector)


class EmbeddingService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.dim = self.settings.embedding_dim

    def embed(self, text: str, target_dim: int | None = None) -> list[float]:
        text = text.strip() or "blank"
        try:
            if self.settings.model_provider == "openai_compatible":
                vector = self._embed_openai_compatible(text)
            elif self.settings.model_provider == "none":
                vector = hash_embedding(text, self.dim)
            else:
                vector = self._embed_ollama(text)
        except Exception:
            vector = hash_embedding(text, self.dim)
        target = target_dim or self.dim
        if len(vector) > target:
            vector = vector[:target]
        elif len(vector) < target and not target_dim:
            self.dim = len(vector)
        return normalize_vector(vector)

    @property
    def active_model_name(self) -> str:
        if self.settings.model_provider == "openai_compatible":
            return self.settings.embedding_api_model or self.settings.cloud_embedding_model or "hash-fallback"
        if self.settings.model_provider == "ollama":
            return self.settings.embedding_model
        return "hash-fallback"

    @property
    def active_image_model_name(self) -> str:
        if self.image_embedding_configured:
            return self.settings.image_embedding_api_model
        return "hash-fallback"

    @property
    def cloud_embedding_configured(self) -> bool:
        if self.settings.model_provider != "openai_compatible":
            return False
        base_url = self.settings.embedding_api_base_url or self.settings.cloud_api_base_url
        api_key = self.settings.embedding_api_key or self.settings.cloud_api_key
        model = self.settings.embedding_api_model or self.settings.cloud_embedding_model
        return bool(base_url and api_key and model)

    @property
    def image_embedding_configured(self) -> bool:
        return bool(
            self.settings.image_embedding_api_base_url
            and self.settings.image_embedding_api_key
            and self.settings.image_embedding_api_model
        )

    def embed_many(self, texts: Iterable[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]

    def embed_image_query(self, text: str) -> list[float]:
        text = text.strip() or "blank"
        try:
            if self.image_embedding_configured:
                if self.settings.image_embedding_provider == "dashscope":
                    return self._embed_dashscope_multimodal({"text": text})
                return self._embed_jina_multimodal(text)
        except Exception as exc:
            import warnings
            warnings.warn(f"[Embedding] image query embedding API failed: {exc}")
        return hash_embedding(text, self.dim)

    def embed_image_file(self, image_path: str | Path, fallback_text: str = "") -> list[float]:
        try:
            if self.image_embedding_configured:
                path = Path(image_path)
                encoded = base64.b64encode(path.read_bytes()).decode("ascii")
                if self.settings.image_embedding_provider == "dashscope":
                    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
                    return self._embed_dashscope_multimodal({"image": f"data:{mime_type};base64,{encoded}"})
                return self._embed_jina_multimodal({"bytes": encoded})
        except Exception as exc:
            import warnings
            warnings.warn(f"[Embedding] image embedding API failed for {image_path}: {exc}")
        return hash_embedding(fallback_text or str(image_path), self.dim)

    def _embed_ollama(self, text: str) -> list[float]:
        base_url = self.settings.ollama_base_url.rstrip("/")
        payload = {"model": self.settings.embedding_model, "input": text}
        with httpx.Client(timeout=20) as client:
            response = client.post(f"{base_url}/api/embed", json=payload)
            if response.status_code == 404:
                legacy = client.post(
                    f"{base_url}/api/embeddings",
                    json={"model": self.settings.embedding_model, "prompt": text},
                )
                legacy.raise_for_status()
                vector = legacy.json()["embedding"]
            else:
                response.raise_for_status()
                data = response.json()
                embeddings = data.get("embeddings") or []
                vector = embeddings[0] if embeddings else data.get("embedding")
            if not vector:
                raise RuntimeError("empty embedding from Ollama")
            return normalize_vector([float(item) for item in vector])

    def _embed_jina_multimodal(self, item: str | dict[str, str]) -> list[float]:
        if not self.settings.image_embedding_api_base_url or not self.settings.image_embedding_api_key:
            raise RuntimeError("image embedding API is not configured")
        base_url = self.settings.image_embedding_api_base_url.rstrip("/")
        payload: dict = {"model": self.settings.image_embedding_api_model, "input": [item]}
        if self.settings.image_embedding_dimensions:
            payload["dimensions"] = self.settings.image_embedding_dimensions
        headers = {
            "Authorization": f"Bearer {self.settings.image_embedding_api_key}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=self.settings.cloud_timeout_seconds) as client:
            response = client.post(f"{base_url}/embeddings", json=payload, headers=headers)
            if response.status_code in {400, 422} and isinstance(item, dict) and "bytes" in item:
                retry_payload = {**payload, "input": [{"image": item["bytes"]}]}
                response = client.post(f"{base_url}/embeddings", json=retry_payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            vector = data.get("data", [{}])[0].get("embedding")
            if not vector:
                raise RuntimeError("empty image embedding from Jina API")
            return normalize_vector([float(value) for value in vector])

    def _embed_dashscope_multimodal(self, content: dict[str, str]) -> list[float]:
        if not self.settings.image_embedding_api_base_url or not self.settings.image_embedding_api_key:
            raise RuntimeError("DashScope multimodal embedding API is not configured")
        base_url = self.settings.image_embedding_api_base_url.rstrip("/")
        payload: dict = {
            "model": self.settings.image_embedding_api_model,
            "input": {"contents": [content]},
        }
        parameters: dict = {}
        if self.settings.image_embedding_dimensions:
            parameters["dimension"] = self.settings.image_embedding_dimensions
        if parameters:
            payload["parameters"] = parameters
        headers = {
            "Authorization": f"Bearer {self.settings.image_embedding_api_key}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=self.settings.cloud_timeout_seconds) as client:
            response = client.post(base_url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            embeddings = data.get("output", {}).get("embeddings") or []
            vector = embeddings[0].get("embedding") if embeddings else None
            if not vector:
                raise RuntimeError("empty image embedding from DashScope API")
            return normalize_vector([float(value) for value in vector])

    def _embed_openai_compatible(self, text: str) -> list[float]:
        base_url = (self.settings.embedding_api_base_url or self.settings.cloud_api_base_url or "").rstrip("/")
        api_key = self.settings.embedding_api_key or self.settings.cloud_api_key
        model = self.settings.embedding_api_model or self.settings.cloud_embedding_model
        if not base_url or not api_key or not model:
            raise RuntimeError("cloud embedding API is not configured")
        headers = {"Authorization": f"Bearer {api_key}"}
        payload: dict = {"model": model, "input": text, "encoding_format": "float"}
        if self.settings.embedding_dimensions:
            payload["dimensions"] = self.settings.embedding_dimensions
        with httpx.Client(timeout=self.settings.cloud_timeout_seconds) as client:
            response = client.post(f"{base_url}/embeddings", json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            vector = data.get("data", [{}])[0].get("embedding")
            if not vector:
                raise RuntimeError("empty embedding from cloud API")
            return normalize_vector([float(item) for item in vector])


@lru_cache
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()
