"""SBERT embeddings generation with document ID mapping and incremental updates."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from ..config import FAST_MODE, MODELS_DIR, PROC_DIR, SBERT_MODEL_NAME
from ..utils_io import load_parquet, save_parquet, load_json, save_json
from loguru import logger

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:
    SentenceTransformer = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def _normalize(matrix: np.ndarray) -> np.ndarray:
    """Normalize embeddings matrix."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12
    return matrix / norms


def _generate_doc_id(row: pd.Series) -> str:
    """
    Generate a stable document ID from row data.
    
    Priority order:
    1. URL (for news articles) - most stable
    2. post_id (for Reddit posts) - stable unique identifier
    3. id (generic ID field) - if available
    4. Hash of text content (only if no unique ID exists)
    
    This ensures IDs remain stable even if preprocessing changes.
    """
    # Priority 1: URL (for news articles)
    url = row.get('url', '')
    if url and pd.notna(url) and str(url).strip():
        return hashlib.md5(str(url).encode()).hexdigest()[:16]
    
    # Priority 2: post_id (for Reddit posts)
    post_id = row.get('post_id', row.get('id', ''))
    if post_id and pd.notna(post_id) and str(post_id).strip():
        # Use post_id directly (it's already unique) or hash it
        post_id_str = str(post_id)
        # If it's already a short alphanumeric ID, use it directly
        if len(post_id_str) <= 16 and post_id_str.isalnum():
            return post_id_str
        # Otherwise hash it
        return hashlib.md5(post_id_str.encode()).hexdigest()[:16]
    
    # Priority 3: Generic id field
    doc_id = row.get('id', '')
    if doc_id and pd.notna(doc_id) and str(doc_id).strip():
        doc_id_str = str(doc_id)
        if len(doc_id_str) <= 16 and doc_id_str.isalnum():
            return doc_id_str
        return hashlib.md5(doc_id_str.encode()).hexdigest()[:16]
    
    # Last resort: hash text content (only if no unique ID exists)
    # This is less stable but better than nothing
    text = str(row.get('text', row.get('clean_text', row.get('title', ''))))
    if text:
        return hashlib.md5(text.encode()).hexdigest()[:16]
    
    # Absolute last resort: use index
    return hashlib.md5(str(row.name).encode()).hexdigest()[:16]


def _load_existing_mapping() -> Tuple[Dict[str, int], Optional[np.ndarray]]:
    """
    Load existing document ID mapping and embeddings if available.
    
    Returns:
        Tuple of (doc_id_to_index mapping, existing embeddings or None)
    """
    mapping_path = MODELS_DIR / "doc_id_mapping.json"
    emb_path = MODELS_DIR / "sbert_embeddings.npy"
    
    if mapping_path.exists() and emb_path.exists():
        try:
            doc_id_to_index = load_json(mapping_path)
            existing_embeddings = np.load(emb_path)
            logger.info(f"Loaded existing mapping: {len(doc_id_to_index)} documents, embeddings shape: {existing_embeddings.shape}")
            return doc_id_to_index, existing_embeddings
        except Exception as e:
            logger.warning(f"Failed to load existing mapping: {e}")
    
    return {}, None


def build_embeddings(force: bool = False, incremental: bool = True) -> tuple[pd.DataFrame, np.ndarray, Dict[str, int]]:
    """
    Build embeddings from processed corpus with document ID mapping.
    
    Args:
        force: Force rebuild even if embeddings exist
        incremental: Allow incremental updates (only process new documents)
    
    Returns:
        Tuple of (corpus, embeddings, doc_id_to_index mapping)
    """
    corpus_path = PROC_DIR / "corpus.parquet"
    if not corpus_path.exists():
        raise FileNotFoundError("Processed corpus not found. Run preprocess step first.")

    corpus = load_parquet(corpus_path)
    
    # Generate document IDs
    corpus["doc_id"] = corpus.apply(_generate_doc_id, axis=1)
    
    if "lemma_text" not in corpus.columns:
        logger.info("No lemma_text column found, using text column")
        texts = corpus.get("text", corpus.get("clean_text", "")).fillna("").tolist()
    else:
        texts = corpus["lemma_text"].fillna("").tolist()

    if SentenceTransformer is None:
        logger.warning("sentence-transformers not installed. Install via `pip install sentence-transformers`. Skipping embeddings.")
        embeddings = np.zeros((len(texts), 1), dtype=np.float32)
        doc_id_to_index = {doc_id: idx for idx, doc_id in enumerate(corpus["doc_id"])}
    else:
        # Try incremental update if enabled
        doc_id_to_index, existing_embeddings = _load_existing_mapping()
        
        if incremental and not force and existing_embeddings is not None:
            # Incremental mode: only process new documents
            existing_doc_ids = set(doc_id_to_index.keys())
            new_mask = ~corpus["doc_id"].isin(existing_doc_ids)
            new_indices = corpus[new_mask].index.tolist()
            
            if len(new_indices) > 0:
                logger.info(f"Incremental update: {len(new_indices)} new documents, {len(existing_doc_ids)} existing")
                new_texts = [texts[i] for i in new_indices]
                new_doc_ids = corpus.loc[new_indices, "doc_id"].tolist()
                
                # Generate embeddings for new documents
                model = SentenceTransformer(SBERT_MODEL_NAME)
                batch_size = 16 if FAST_MODE else 64
                logger.info(f"Encoding {len(new_texts)} new documents...")
                new_embeddings = model.encode(new_texts, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=False)
                new_embeddings = _normalize(new_embeddings.astype(np.float32))
                
                # Combine with existing
                embeddings = np.vstack([existing_embeddings, new_embeddings])
                
                # Update mapping for new documents
                next_index = len(existing_embeddings)
                for doc_id in new_doc_ids:
                    doc_id_to_index[doc_id] = next_index
                    next_index += 1
            else:
                logger.info("No new documents, using existing embeddings")
                embeddings = existing_embeddings
        else:
            # Full rebuild
            model = SentenceTransformer(SBERT_MODEL_NAME)
            batch_size = 16 if FAST_MODE else 64
            logger.info(f"Encoding {len(texts)} documents with SBERT model {SBERT_MODEL_NAME}")
            embeddings = model.encode(texts, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=False)
            embeddings = _normalize(embeddings.astype(np.float32))
            
            # Create mapping: doc_id -> FAISS index position
            doc_id_to_index = {doc_id: idx for idx, doc_id in enumerate(corpus["doc_id"])}

    # Save embeddings and mapping
    models_dir = MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)
    
    emb_path = models_dir / "sbert_embeddings.npy"
    np.save(emb_path, embeddings)
    logger.info(f"Saved embeddings to {emb_path} (shape: {embeddings.shape})")
    
    mapping_path = models_dir / "doc_id_mapping.json"
    save_json(doc_id_to_index, mapping_path)
    logger.info(f"Saved document ID mapping to {mapping_path} ({len(doc_id_to_index)} documents)")

    slim_path = MODELS_DIR / "sbert_embeddings_small.npy"
    np.save(slim_path, embeddings[: min(50, len(embeddings))])
    logger.info(f"Saved slim embeddings to {slim_path}")

    small_corpus_path = PROC_DIR / "corpus_small.parquet"
    if not small_corpus_path.exists() and "platform" in corpus.columns:
        cols = ["platform", "title", "lemma_text", "city", "url", "doc_id"]
        available_cols = [c for c in cols if c in corpus.columns]
        if available_cols:
            corpus[available_cols].to_parquet(small_corpus_path, index=False)

    return corpus, embeddings, doc_id_to_index


if __name__ == "__main__":
    build_embeddings()

