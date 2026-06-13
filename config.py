from typing import Dict
import os

import yaml

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
_CONFIG_CACHE: Dict[str, object] = {}

def _load_config(path: str = _CONFIG_PATH) -> Dict[str, object]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError as exc:
        raise ValueError(f"Missing config file: {path}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in config file: {path}") from exc
    
def get_qdrant_apikey(env_name: str = "QDRANT_API_KEY") -> str:
    return os.getenv(env_name, "").strip()

def get_config() -> Dict[str, object]:
    global _CONFIG_CACHE
    if _CONFIG_CACHE:
        return _CONFIG_CACHE
    try:
        raw = _load_config()
    except ValueError:
        raw = {}
        
    qdrant_api_key_env = str(raw.get("qdrant_api_key_env", "QDRANT_API_KEY"))

    _CONFIG_CACHE = {
        "qdrant_url": raw.get("qdrant_url", ""),
        "qdrant_api_key": get_qdrant_apikey(qdrant_api_key_env),
        "qdrant_api_key_env": qdrant_api_key_env,
        "collection_name": raw.get("collection_name", "knowledge_base"),
        "embed_model": raw.get("embed_model", "all-MiniLM-L6-v2"),
        "embed_text_batch_size": int(raw.get("embed_text_batch_size", 32)),
        "chunk_target_tokens": int(raw.get("chunk_target_tokens", 220)),
        "chunk_overlap_tokens": int(raw.get("chunk_overlap_tokens", 40)),
        "rerank_model": str(raw.get("rerank_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")),
        "rerank_candidate_k": int(raw.get("rerank_candidate_k", 20)),
        "rerank_top_k": int(raw.get("rerank_top_k", 5)),
        "max_context_chars": int(raw.get("max_context_chars", 6000)),
        "llm_model": str(raw.get("llm_model", "deepseek-chat")),
        "llm_api_key_env": str(raw.get("llm_api_key_env", "DEEPSEEK_API_KEY")),
        "llm_url": str(raw.get("llm_url", raw.get("llm_base_url", "https://api.deepseek.com"))),
    }
    return _CONFIG_CACHE
