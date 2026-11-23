"""Centralized path, environment, and configuration loader for RAGOps."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

# Project root (used as fallback when running locally without K8s env vars)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv_if_present() -> None:
    """Load key.env or .env into os.environ without overwriting existing vars."""
    for candidate in (PROJECT_ROOT / "key.env", PROJECT_ROOT / ".env"):
        if candidate.exists():
            for line in candidate.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v


_load_dotenv_if_present()


def get_lake_dir() -> Path:
    return Path(os.environ.get("LAKE_DIR", str(PROJECT_ROOT / "corpus")))


def get_incoming_dir() -> Path:
    return get_lake_dir() / "incoming"


def get_raw_dir() -> Path:
    return get_lake_dir() / "raw"


def get_quarantine_dir() -> Path:
    return get_lake_dir() / "quarantine"


def get_archive_dir() -> Path:
    return get_lake_dir() / "archive"


def get_index_dir() -> Path:
    return Path(os.environ.get("INDEX_DIR", str(PROJECT_ROOT / "data")))


def get_results_dir() -> Path:
    return Path(os.environ.get("RESULTS_DIR", str(PROJECT_ROOT / "eval" / "results")))


def get_traces_dir() -> Path:
    return Path(os.environ.get("TRACES_DIR", str(PROJECT_ROOT / "traces")))


def get_collection_name() -> str:
    return os.environ.get("COLLECTION_NAME", "fast-policies-v1")


def get_embedding_version() -> str:
    return os.environ.get("EMBEDDING_VERSION", "lexical-proj-v1")


def get_chunk_size() -> int:
    return int(os.environ.get("CHUNK_SIZE", "320"))


def get_chunk_overlap() -> int:
    return int(os.environ.get("CHUNK_OVERLAP", "80"))


def get_index_path(collection_name: str | None = None, index_dir: Path | None = None) -> Path:
    col = collection_name or get_collection_name()
    idx_dir = index_dir or get_index_dir()
    return idx_dir / f"{col}.json"


def get_manifest_path(collection_name: str | None = None, index_dir: Path | None = None) -> Path:
    col = collection_name or get_collection_name()
    idx_dir = index_dir or get_index_dir()
    return idx_dir / f"{col}.manifest.json"


def get_gate_config_path() -> Path:
    return Path(os.environ.get("GATE_CONFIG_PATH", str(PROJECT_ROOT / "eval" / "gate_config.yaml")))


def load_gate_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load gate and engine config from YAML, applying generic env overrides."""
    path = config_path or get_gate_config_path()
    cfg: dict[str, Any] = {}
    if path.exists():
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if isinstance(raw, dict):
            cfg = raw

    # Generic env override for every scalar key in gate config (case-insensitive)
    for key, val in list(cfg.items()):
        env_key = key.upper()
        if env_key in os.environ:
            env_val = os.environ[env_key]
            if isinstance(val, bool):
                cfg[key] = env_val.lower() in ("1", "true", "yes")
            elif isinstance(val, int):
                cfg[key] = int(env_val)
            elif isinstance(val, float):
                cfg[key] = float(env_val)
            else:
                cfg[key] = env_val

    # Also support lowercase env keys directly (e.g. from ConfigMap envFrom)
    for key, val in list(cfg.items()):
        if key in os.environ:
            env_val = os.environ[key]
            if isinstance(val, bool):
                cfg[key] = env_val.lower() in ("1", "true", "yes")
            elif isinstance(val, int):
                cfg[key] = int(env_val)
            elif isinstance(val, float):
                cfg[key] = float(env_val)
            else:
                cfg[key] = env_val

    return cfg


def get_relevance_threshold() -> float:
    cfg = load_gate_config()
    if "RELEVANCE_THRESHOLD" in os.environ:
        return float(os.environ["RELEVANCE_THRESHOLD"])
    return float(cfg.get("relevance_threshold", 0.30))
