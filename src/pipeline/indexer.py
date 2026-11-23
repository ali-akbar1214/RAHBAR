"""Part A2: Incremental Ingestion Engine & Collection-Scoped Vector Store.

Satisfies all RAGOps data-management requirements:
  - Collection-scoped storage: ${INDEX_DIR}/${COLLECTION_NAME}.json + manifest.json.
  - Real <!-- page: N --> marker parsing into ChunkRecord.page (B-1).
  - Drops header-only chunks lacking body content (D-5).
  - Atomic saves via temp file + os.replace (A-10).
  - Strict doc_id matching on delete_by_doc_id / count_by_doc_id (A-4).
  - Deletion pass skips catalog entries with superseded=True and archived_path (A-3).
  - mtime reload support for API pods (F-4).
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import timezone
from pathlib import Path
from typing import Any

from .config import (
    get_archive_dir,
    get_chunk_overlap,
    get_chunk_size,
    get_collection_name,
    get_embedding_version,
    get_index_dir,
    get_index_path,
    get_manifest_path,
    get_raw_dir,
)
from .embed import DEFAULT_EMBEDDING_VERSION, cosine_similarity, embed_text, get_embedding_spec
from .verification import parse_front_matter


@dataclass
class ChunkRecord:
    chunk_id: str
    doc_id: str
    source: str
    doc_version: str
    title: str
    section: str
    text: str
    vector: list[float]
    superseded: bool = False
    page: int = 1


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp_path, path)


class VectorStore:
    """Persistent, lightweight, atomic Vector Store with metadata filtering and mtime reload."""

    def __init__(self, storage_path: str | Path | None = None):
        self.storage_path = Path(storage_path) if storage_path else get_index_path()
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.chunks: dict[str, ChunkRecord] = {}
        self._last_mtime: float = 0.0
        self.load()

    def load(self) -> None:
        if self.storage_path.exists():
            try:
                stat = self.storage_path.stat()
                data = json.loads(self.storage_path.read_text(encoding="utf-8"))
                loaded: dict[str, ChunkRecord] = {}
                for item in data.get("chunks", []):
                    rec = ChunkRecord(**item)
                    loaded[rec.chunk_id] = rec
                self.chunks = loaded
                self._last_mtime = stat.st_mtime
            except Exception as e:
                print(f"[VectorStore] Warning: load failed ({e}), starting fresh.")

    def reload_if_changed(self) -> bool:
        """Reload store from disk if storage_path mtime changed. Returns True if reloaded."""
        if not self.storage_path.exists():
            return False
        try:
            current_mtime = self.storage_path.stat().st_mtime
            if current_mtime != self._last_mtime:
                self.load()
                return True
        except OSError:
            pass
        return False

    def save(self) -> None:
        serialized = [asdict(c) for c in self.chunks.values()]
        _atomic_write_json(self.storage_path, {"chunks": serialized})
        if self.storage_path.exists():
            self._last_mtime = self.storage_path.stat().st_mtime

    def upsert_chunks(self, new_chunks: list[ChunkRecord]) -> int:
        for chunk in new_chunks:
            self.chunks[chunk.chunk_id] = chunk
        self.save()
        return len(new_chunks)

    def delete_by_doc_id(self, doc_id: str) -> int:
        to_del = [cid for cid, c in self.chunks.items() if c.doc_id == doc_id]
        for cid in to_del:
            del self.chunks[cid]
        if to_del:
            self.save()
        return len(to_del)

    def count_by_doc_id(self, doc_id: str, include_superseded: bool = True) -> int:
        return sum(
            1
            for c in self.chunks.values()
            if c.doc_id == doc_id and (include_superseded or not c.superseded)
        )

    def chunk_counts_by_doc(self, include_superseded: bool = True) -> dict[str, int]:
        counts: dict[str, int] = {}
        for c in self.chunks.values():
            if not include_superseded and c.superseded:
                continue
            counts[c.doc_id] = counts.get(c.doc_id, 0) + 1
        return dict(sorted(counts.items()))

    def total_chunks(self, include_superseded: bool = True) -> int:
        if include_superseded:
            return len(self.chunks)
        return sum(1 for c in self.chunks.values() if not c.superseded)

    def query(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filter_superseded: bool = True,
    ) -> list[tuple[ChunkRecord, float]]:
        results: list[tuple[ChunkRecord, float]] = []
        for chunk in self.chunks.values():
            if filter_superseded and chunk.superseded:
                continue
            sim = cosine_similarity(query_vector, chunk.vector)
            results.append((chunk, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


_PAGE_MARKER_RE = re.compile(r"<!--\s*page:\s*(\d+)\s*-->", re.IGNORECASE)


def chunk_document(
    doc_id: str,
    meta: dict[str, Any],
    body: str,
    chunk_size: int = 320,
    chunk_overlap: int = 80,
) -> list[dict[str, Any]]:
    """Split markdown text into semantic sections and overlapping chunks with real page tracking.

    Header-only sections (containing only markdown headings and no content lines)
    are dropped so title stubs do not pollute top-1 retrieval (D-5).
    """
    source = str(meta.get("source", doc_id))
    title = str(meta.get("title", source))
    version = str(meta.get("doc_version", "unknown"))
    superseded = bool(meta.get("superseded", False))

    # Parse sections while tracking <!-- page: N --> markers across the document
    current_page = 1
    parsed_sections: list[tuple[str, int, str]] = []
    cur_header = title
    cur_page = 1
    cur_lines: list[str] = []

    for line in body.splitlines():
        m_page = _PAGE_MARKER_RE.search(line)
        if m_page:
            current_page = int(m_page.group(1))
            cleaned_line = _PAGE_MARKER_RE.sub("", line).strip()
            if not cleaned_line:
                continue
            line = cleaned_line

        if re.match(r"^#{2,3}\s+", line.strip()):
            # Flush previous section if it had non-heading content
            non_heading = [ln for ln in cur_lines if ln.strip() and not ln.strip().startswith("#")]
            if non_heading:
                parsed_sections.append((cur_header, cur_page, "\n".join(cur_lines).strip()))
            cur_header = line.strip().lstrip("#").strip()
            cur_page = current_page
            cur_lines = []
        else:
            cur_lines.append(line)

    non_heading = [ln for ln in cur_lines if ln.strip() and not ln.strip().startswith("#")]
    if non_heading:
        parsed_sections.append((cur_header, cur_page, "\n".join(cur_lines).strip()))

    raw_chunks: list[dict[str, Any]] = []
    chunk_idx = 0
    step = max(1, chunk_size - chunk_overlap)

    for header, sec_page, sec_body in parsed_sections:
        full_sec = f"{header}: {sec_body}"
        if len(full_sec) <= chunk_size:
            chunk_idx += 1
            raw_chunks.append({
                "chunk_id": f"{doc_id}_c{chunk_idx:02d}",
                "doc_id": doc_id,
                "source": source,
                "title": title,
                "doc_version": version,
                "section": header,
                "text": full_sec,
                "superseded": superseded,
                "page": sec_page,
            })
        else:
            start = 0
            while start < len(full_sec):
                end = start + chunk_size
                piece = full_sec[start:end]
                chunk_idx += 1
                raw_chunks.append({
                    "chunk_id": f"{doc_id}_c{chunk_idx:02d}",
                    "doc_id": doc_id,
                    "source": source,
                    "title": title,
                    "doc_version": version,
                    "section": header,
                    "text": piece,
                    "superseded": superseded,
                    "page": sec_page,
                })
                if end >= len(full_sec):
                    break
                start += step

    return raw_chunks


class IncrementalIndexer:
    """Manages collection catalog, manifest, and incremental updates against VectorStore."""

    def __init__(
        self,
        corpus_dir: str | Path | None = None,
        archive_dir: str | Path | None = None,
        data_dir: str | Path | None = None,
        collection_name: str | None = None,
        embedding_version: str | None = None,
    ):
        self.corpus_dir = Path(corpus_dir) if corpus_dir else get_raw_dir()
        self.archive_dir = Path(archive_dir) if archive_dir else get_archive_dir()
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir = Path(data_dir) if data_dir else get_index_dir()
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.collection_name = collection_name or get_collection_name()
        self.embedding_version = embedding_version or get_embedding_version() or DEFAULT_EMBEDDING_VERSION

        self.index_path = get_index_path(self.collection_name, self.data_dir)
        self.manifest_path = get_manifest_path(self.collection_name, self.data_dir)
        self.catalog_path = self.data_dir / f"{self.collection_name}.catalog.json"

        self.vector_store = VectorStore(self.index_path)
        self.catalog: dict[str, dict[str, Any]] = self._load_catalog()

    def _load_catalog(self) -> dict[str, dict[str, Any]]:
        if self.catalog_path.exists():
            try:
                return json.loads(self.catalog_path.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _save_catalog(self) -> None:
        _atomic_write_json(self.catalog_path, self.catalog)

    def _save_manifest(self, chunk_size: int, chunk_overlap: int) -> None:
        spec = get_embedding_spec(self.embedding_version)
        manifest = {
            "collection_name": self.collection_name,
            "embedding_model": self.embedding_version,
            "embedding_version": self.embedding_version,
            "embedding_dim": spec["dim"],
            "embedding_description": spec["description"],
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "total_chunks": self.vector_store.total_chunks(include_superseded=True),
            "active_chunks": self.vector_store.total_chunks(include_superseded=False),
            "document_hashes": {k: v.get("hash") for k, v in self.catalog.items()},
            "updated_at": datetime.datetime.now(timezone.utc).isoformat(),
        }
        _atomic_write_json(self.manifest_path, manifest)

    def run_ingestion(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> dict[str, Any]:
        """Ingest corpus files incrementally.

        - Embeds 0 chunks on unchanged files.
        - Replaces only the modified file's chunks.
        - Removes orphan chunks for deleted files while preserving archived/superseded entries (A-3).
        """
        c_size = chunk_size if chunk_size is not None else get_chunk_size()
        c_overlap = chunk_overlap if chunk_overlap is not None else get_chunk_overlap()

        all_files = sorted(self.corpus_dir.glob("*.md"))
        current_filenames = {f.name for f in all_files}

        stats: dict[str, Any] = {
            "collection_name": self.collection_name,
            "embedding_version": self.embedding_version,
            "chunk_size": c_size,
            "chunk_overlap": c_overlap,
            "scanned": len(all_files),
            "unchanged": 0,
            "modified": 0,
            "added": 0,
            "deleted": 0,
            "archived_preserved": 0,
            "chunks_embedded": 0,
            "chunks_deleted": 0,
            "total_chunks_in_store": 0,
            "active_chunks_in_store": 0,
        }

        # 1. Deletion pass: remove orphan chunks for truly deleted files; skip archived files (A-3)
        for doc_name in list(self.catalog.keys()):
            entry = self.catalog[doc_name]
            if doc_name not in current_filenames:
                if entry.get("superseded") is True and entry.get("archived_path"):
                    stats["archived_preserved"] += 1
                    continue
                deleted_chunks = self.vector_store.delete_by_doc_id(doc_name)
                del self.catalog[doc_name]
                stats["deleted"] += 1
                stats["chunks_deleted"] += deleted_chunks
                print(f"[A2 Incremental] Deleted document: {doc_name} -> Removed {deleted_chunks} orphan chunks.")

        # 2. Check existing manifest embedding version mismatch (forces re-embed on new embedding version)
        reembed_all = False
        if self.manifest_path.exists():
            try:
                old_manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
                if old_manifest.get("embedding_version") != self.embedding_version or (
                    old_manifest.get("chunk_size") != c_size or old_manifest.get("chunk_overlap") != c_overlap
                ):
                    reembed_all = True
            except Exception:
                pass

        # 3. Process each file in corpus_dir
        for f_path in all_files:
            doc_name = f_path.name
            raw_text = f_path.read_text(encoding="utf-8", errors="ignore")
            meta, body = parse_front_matter(raw_text)
            content_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()

            if not reembed_all and doc_name in self.catalog and self.catalog[doc_name]["hash"] == content_hash:
                stats["unchanged"] += 1
                continue

            is_mod = doc_name in self.catalog
            if is_mod:
                del_count = self.vector_store.delete_by_doc_id(doc_name)
                stats["modified"] += 1
                stats["chunks_deleted"] += del_count
                print(f"[A2 Incremental] Modified document: {doc_name} -> Replaced {del_count} old chunks.")
            else:
                stats["added"] += 1
                print(f"[A2 Incremental] New document: {doc_name}")

            chunks_meta = chunk_document(doc_name, meta, body, c_size, c_overlap)
            vectors = [embed_text(c["text"], version=self.embedding_version) for c in chunks_meta]
            stats["chunks_embedded"] += len(chunks_meta)

            records = [
                ChunkRecord(
                    chunk_id=cm["chunk_id"],
                    doc_id=cm["doc_id"],
                    source=cm["source"],
                    doc_version=cm["doc_version"],
                    title=cm["title"],
                    section=cm["section"],
                    text=cm["text"],
                    vector=v,
                    superseded=cm["superseded"],
                    page=cm["page"],
                )
                for cm, v in zip(chunks_meta, vectors)
            ]
            self.vector_store.upsert_chunks(records)

            self.catalog[doc_name] = {
                "hash": content_hash,
                "source": str(meta.get("source", doc_name)),
                "doc_version": str(meta.get("doc_version", "unknown")),
                "superseded": bool(meta.get("superseded", False)),
                "chunk_count": len(records),
                "indexed_at": datetime.datetime.now(timezone.utc).isoformat(),
            }

        self._save_catalog()
        self._save_manifest(c_size, c_overlap)
        stats["total_chunks_in_store"] = self.vector_store.total_chunks(include_superseded=True)
        stats["active_chunks_in_store"] = self.vector_store.total_chunks(include_superseded=False)
        stats["counts_by_doc"] = self.vector_store.chunk_counts_by_doc(include_superseded=True)
        return stats

    def archive_superseded(self, filename: str) -> bool:
        """Archive superseded policy file to archive_dir and exclude from retrieval."""
        src_file = self.corpus_dir / filename
        if not src_file.exists():
            return False

        target_file = self.archive_dir / filename
        shutil.move(str(src_file), str(target_file))

        count_updated = 0
        for c in self.vector_store.chunks.values():
            if c.doc_id == filename:
                c.superseded = True
                count_updated += 1
        self.vector_store.save()

        if filename in self.catalog:
            self.catalog[filename]["superseded"] = True
            self.catalog[filename]["archived_path"] = str(target_file)
            self._save_catalog()

        print(f"[A2 Archival] {filename} moved to archive. {count_updated} chunks tagged superseded: true.")
        return True

    def rollback_version(self, filename: str) -> bool:
        """Restore archived document and reinstate chunks for active retrieval."""
        archived_file = self.archive_dir / filename
        if not archived_file.exists():
            return False

        target_file = self.corpus_dir / filename
        shutil.move(str(archived_file), str(target_file))

        count_reinstated = 0
        for c in self.vector_store.chunks.values():
            if c.doc_id == filename:
                c.superseded = False
                count_reinstated += 1
        self.vector_store.save()

        if filename in self.catalog:
            self.catalog[filename]["superseded"] = False
            self.catalog[filename].pop("archived_path", None)
            self._save_catalog()

        print(f"[A2 Rollback] {filename} restored from archive. {count_reinstated} chunks reinstated (superseded: false).")
        return True
