# Lazy import sentence-transformers to avoid hard dependency at import time

from sentence_transformers import SentenceTransformer  # type: ignore
import yaml
import os
from typing import Union, Any, Optional
import random
import json
from dataclasses import dataclass
import math


def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config


def load_json(file_name: str) -> Union[list, dict]:

    if not os.path.exists(file_name):
        return None
    with open(file_name, encoding="utf-8") as f:
        return json.load(f)


def write_json(json_obj, file_name):
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(json_obj, f, indent=2, ensure_ascii=False, separators=(",", ": "))

def random_divide_list(lst: list[Any], k: int) -> list[list]:
    """
    Divides the list into chunks, each with maximum length k.

    Args:
        lst: The list to be divided.
        k: The maximum length of each chunk.

    Returns:
        A list of chunks.
    """
    if len(lst) == 0:
        return []
    
    random.shuffle(lst)
    if len(lst) <= k:
        return [lst]
    else:
        num_chunks = math.ceil(len(lst) / k)
        chunk_size = math.ceil(len(lst) / num_chunks)
        return [lst[i*chunk_size:(i+1)*chunk_size] for i in range(num_chunks)]
    

_EMBEDDING_MODEL_CACHE = {} 

@dataclass
class EmbeddingFunc:

    model_type: str = "sentence-transformers/all-MiniLM-L6-v2"
    # Optional CUDA device like 'cuda:0', 'cuda:1', or 'cpu'
    device: Optional[str] = None
    # Batch size for sentence-transformers encode; lower to reduce VRAM
    batch_size: int = 16

    def __post_init__(self):
        # Try to initialize the transformer model; if unavailable, use a simple hashing encoder
        if self.model_type not in _EMBEDDING_MODEL_CACHE:
            encoder = None
            if SentenceTransformer is not None:
                try:
                    # sentence-transformers supports `device` kw in recent versions
                    if self.device:
                        encoder = SentenceTransformer(self.model_type, device=self.device)
                    else:
                        encoder = SentenceTransformer(self.model_type)
                except Exception:
                    encoder = None

            if encoder is None:
                # Basic fallback encoder producing fixed-length vectors via token hashing
                class BasicEncoder:
                    def __init__(self, dim: int = 384):
                        self.dim = dim

                    def encode(self, text: str):
                        vec = [0.0] * self.dim
                        # Token-based hashing accumulation
                        for token in (text or "").split():
                            idx = (abs(hash(token)) % self.dim)
                            vec[idx] += 1.0
                        # L2 normalization
                        import math
                        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
                        return [v / norm for v in vec]

                encoder = BasicEncoder()

            _EMBEDDING_MODEL_CACHE[self.model_type] = encoder

        self.func = _EMBEDDING_MODEL_CACHE[self.model_type]

    def embed_documents(self, texts: list[str]) -> list[list]:
        """
        Embed a list of documents. Prefer batching with progress bars disabled.
        Falls back to per-text encoding if the encoder does not support batched input
        or the `show_progress_bar` parameter.
        """
        if not texts:
            return []
        try:
            # Try batched encoding with progress bar disabled
            # Pass device/batch_size when supported to control VRAM usage
            vecs = self.func.encode(
                texts,
                show_progress_bar=False,
                batch_size=self.batch_size,
                device=self.device if self.device else None,
            )
            # sentence-transformers may return numpy array or list
            if hasattr(vecs, "tolist"):
                vecs = vecs.tolist()
            return [v.tolist() if hasattr(v, "tolist") else v for v in vecs]
        except TypeError:
            # Encoder may not accept show_progress_bar or list inputs
            vectors = []
            for text in texts:
                try:
                    vec = self.func.encode(text, device=self.device) if self.device else self.func.encode(text)
                except TypeError:
                    vec = self.func.encode(text)
                if hasattr(vec, "tolist"):
                    vec = vec.tolist()
                vectors.append(vec)
            return vectors

    def embed_query(self, query: str) -> list:
        try:
            vec = self.func.encode(
                query,
                show_progress_bar=False,
                batch_size=self.batch_size,
                device=self.device if self.device else None,
            )
        except TypeError:
            try:
                vec = self.func.encode(query, device=self.device) if self.device else self.func.encode(query)
            except TypeError:
                vec = self.func.encode(query)
        return vec.tolist() if hasattr(vec, "tolist") else vec

import fcntl
import time

class InterProcessFileLock:
    def __init__(self, path: str, timeout: int = 60, poll_interval: float = 0.1):
        self.path = path
        self.timeout = timeout
        self.poll = poll_interval
        self.fd = None
    def __enter__(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.fd = open(self.path, "a+")
        start = time.time()
        while True:
            try:
                fcntl.flock(self.fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.time() - start > self.timeout:
                    raise TimeoutError("lock timeout")
                time.sleep(self.poll)
        return self
    def __exit__(self, exc_type, exc, tb):
        try:
            if self.fd:
                fcntl.flock(self.fd.fileno(), fcntl.LOCK_UN)
                self.fd.close()
        finally:
            self.fd = None


