"""FAISS index with scalable options and incremental updates."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import INDEX_DIR, MODELS_DIR, PROC_DIR, SBERT_MODEL_NAME, config
from ..utils_io import load_parquet, load_json, save_json
from loguru import logger

try:
    import faiss
except ImportError:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

INDEX_PATH = INDEX_DIR / "sbert.cosine.faiss"
MAPPING_PATH = MODELS_DIR / "doc_id_mapping.json"


def _load_embeddings() -> Tuple[np.ndarray, pd.DataFrame, Dict[str, int]]:
    """
    Load embeddings, corpus, and document ID mapping.
    
    Returns:
        Tuple of (embeddings, corpus, doc_id_to_index mapping)
    """
    emb_path = MODELS_DIR / "sbert_embeddings.npy"
    if not emb_path.exists():
        raise FileNotFoundError("Embeddings not found. Run embeddings.py first.")
    corpus_path = PROC_DIR / "corpus.parquet"
    if not corpus_path.exists():
        raise FileNotFoundError("Corpus not found. Run preprocess.py first.")
    
    embeddings = np.load(emb_path).astype(np.float32)
    corpus = load_parquet(corpus_path)
    
    # Load document ID mapping
    doc_id_to_index = {}
    if MAPPING_PATH.exists():
        try:
            doc_id_to_index = load_json(MAPPING_PATH)
        except Exception as e:
            logger.warning(f"Could not load document ID mapping: {e}")
    
    return embeddings, corpus, doc_id_to_index


def _create_index(dim: int, index_type: str = "flat") -> "faiss.Index":
    """
    Create FAISS index based on type.
    
    Args:
        dim: Embedding dimension
        index_type: Type of index ("flat", "hnsw", "ivf")
    
    Returns:
        FAISS index
    """
    if faiss is None:
        raise ImportError("faiss is not installed. Install via `pip install faiss-cpu`.")
    
    if index_type == "flat":
        # IndexFlatIP: Exact search, good for small datasets (<100K)
        return faiss.IndexFlatIP(dim)
    
    elif index_type == "hnsw":
        # IndexHNSWFlat: Approximate search, good for large datasets
        # M=32, efConstruction=200 are good defaults
        index = faiss.IndexHNSWFlat(dim, 32)
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = 50
        return index
    
    elif index_type == "ivf":
        # IVF (Inverted File Index): Good for very large datasets
        # nlist = number of clusters (use sqrt of dataset size)
        quantizer = faiss.IndexFlatIP(dim)
        nlist = min(100, max(10, int(np.sqrt(10000))))  # Adaptive nlist
        index = faiss.IndexIVFFlat(quantizer, dim, nlist)
        index.nprobe = 10  # Number of clusters to search
        return index
    
    else:
        raise ValueError(f"Unknown index type: {index_type}. Use 'flat', 'hnsw', or 'ivf'")


def build_index(force: bool = False, incremental: bool = True, index_type: Optional[str] = None) -> Path:
    """
    Build FAISS index from embeddings with incremental update support.
    
    Args:
        force: Force full rebuild
        incremental: Allow incremental updates
        index_type: Index type ("flat", "hnsw", "ivf"). If None, uses config or auto-selects.
    
    Returns:
        Path to saved index
    """
    if faiss is None:
        raise ImportError("faiss is not installed. Install via `pip install faiss-cpu`.")
    
    embeddings, corpus, doc_id_to_index = _load_embeddings()
    dim = embeddings.shape[1]
    
    # Determine index type
    if index_type is None:
        index_type = config.get("index", {}).get("type", "auto")
    
    if index_type == "auto":
        # Auto-select based on dataset size
        n_vectors = len(embeddings)
        if n_vectors < 10000:
            index_type = "flat"  # Small: use exact search
        elif n_vectors < 100000:
            index_type = "hnsw"  # Medium: use HNSW
        else:
            index_type = "ivf"  # Large: use IVF
    
    logger.info(f"Building {index_type} index for {len(embeddings)} vectors (dim={dim})")
    
    # Check for incremental update
    existing_index = None
    existing_doc_ids = set()
    
    if incremental and not force and INDEX_PATH.exists():
        try:
            existing_index = faiss.read_index(str(INDEX_PATH))
            # Load the doc_ids that were in the index when it was last built
            metadata_path = INDEX_DIR / "index_metadata.json"
            if metadata_path.exists():
                try:
                    metadata = load_json(metadata_path)
                    existing_doc_ids = set(metadata.get("doc_ids_in_index", []))
                except Exception as e:
                    logger.warning(f"Could not load index metadata: {e}. Will rebuild.")
                    existing_index = None
            else:
                # No metadata means we can't do incremental update safely
                logger.warning("No index metadata found. Will rebuild.")
                existing_index = None
            
            if existing_index is not None:
                logger.info(f"Found existing index with {existing_index.ntotal} vectors and {len(existing_doc_ids)} doc IDs")
        except Exception as e:
            logger.warning(f"Could not load existing index: {e}. Building fresh.")
    
    # Determine which embeddings are new
    if incremental and existing_index is not None:
        new_doc_ids = set(doc_id_to_index.keys()) - existing_doc_ids
        if new_doc_ids:
            # Get indices of new documents
            new_indices = [doc_id_to_index[doc_id] for doc_id in new_doc_ids if doc_id in doc_id_to_index]
            new_embeddings = embeddings[new_indices]
            
            logger.info(f"Incremental update: adding {len(new_embeddings)} new vectors to index")
            existing_index.add(new_embeddings)
            index = existing_index
        else:
            logger.info("No new documents, using existing index")
            index = existing_index
    else:
        # Full rebuild
        index = _create_index(dim, index_type)
        
        # For IVF, need to train before adding
        if index_type == "ivf":
            logger.info("Training IVF index...")
            index.train(embeddings)
        
        index.add(embeddings)
    
    # Save index
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    
    # Save index metadata (including doc_ids that are in this index)
    metadata = {
        "index_type": index_type,
        "dim": dim,
        "n_vectors": index.ntotal,
        "doc_id_mapping_path": str(MAPPING_PATH),
        "doc_ids_in_index": list(doc_id_to_index.keys())  # Store doc IDs for incremental updates
    }
    metadata_path = INDEX_DIR / "index_metadata.json"
    save_json(metadata, metadata_path)
    
    logger.info(f"FAISS {index_type} index with {index.ntotal} vectors saved to {INDEX_PATH}")
    return INDEX_PATH


def load_index() -> Tuple["faiss.Index", Dict[str, int]]:
    """
    Load FAISS index and document ID mapping.
    
    Returns:
        Tuple of (index, doc_id_to_index mapping)
    """
    if faiss is None:
        raise ImportError("faiss is not installed. Install via `pip install faiss-cpu`.")
    if not INDEX_PATH.exists():
        logger.info("Index not found; building one now.")
        build_index()
    
    index = faiss.read_index(str(INDEX_PATH))
    
    # Load document ID mapping
    doc_id_to_index = {}
    if MAPPING_PATH.exists():
        try:
            doc_id_to_index = load_json(MAPPING_PATH)
        except Exception as e:
            logger.warning(f"Could not load document ID mapping: {e}")
    
    return index, doc_id_to_index


def encode_query(query: str) -> np.ndarray:
    """Encode query text to embedding."""
    if SentenceTransformer is None:
        raise ImportError("sentence-transformers missing; install via `pip install sentence-transformers`.")
    model = SentenceTransformer(SBERT_MODEL_NAME)
    vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    return vec.astype(np.float32)


def search(query: str, city: str | None, top_k: int) -> pd.DataFrame:
    """
    Search corpus with query using document ID mapping.
    
    Args:
        query: Search query text
        city: Optional city filter
        top_k: Number of results to return
    
    Returns:
        DataFrame with search results
    """
    embeddings, corpus, doc_id_to_index = _load_embeddings()
    if faiss is None:
        raise ImportError("faiss is not installed; cannot perform search.")
    
    index, index_doc_id_mapping = load_index()
    query_vec = encode_query(query)
    
    # Search index - fetch a large k to ensure we have enough results after filtering
    # If filtering by city/topic, we need many more candidates than top_k
    search_k = max(500, top_k * 10)  # Fetch at least 500, or 10x top_k, whichever is larger
    scores, indices = index.search(query_vec, min(search_k, index.ntotal))  # Don't exceed index size
    
    results = []
    seen_doc_ids = set()
    
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        
        # Map FAISS index position to document ID
        doc_id = None
        row = None
        
        if index_doc_id_mapping:
            # Find doc_id by index position (reverse lookup)
            for did, pos in index_doc_id_mapping.items():
                if pos == int(idx):
                    doc_id = did
                    break
            
            # Get corpus row by doc_id
            if doc_id and "doc_id" in corpus.columns:
                matching_rows = corpus[corpus["doc_id"] == doc_id]
                if len(matching_rows) > 0:
                    row = matching_rows.iloc[0]
                else:
                    logger.debug(f"Doc ID {doc_id} not found in corpus")
                    continue
        
        # Fallback: use corpus row index (for backward compatibility or if no mapping)
        if row is None:
            if int(idx) < len(corpus):
                row = corpus.iloc[int(idx)]
                # Get doc_id from row if available
                if "doc_id" in row:
                    doc_id = row.get("doc_id")
            else:
                logger.warning(f"Index position {idx} out of range for corpus (len={len(corpus)})")
                continue
        
        # Deduplicate by doc_id if available
        if doc_id:
            if doc_id in seen_doc_ids:
                continue
            seen_doc_ids.add(doc_id)
        
        # Apply city filter
        if city and row.get("city") and city.lower() not in str(row.get("city", "")).lower():
            continue
        
        results.append({
            "score": float(score),
            "doc_id": doc_id,
            "city": row.get("city"),
            "title": row.get("title"),
            "url": row.get("url"),
        })
        
        if len(results) >= top_k:
            break
    
    return pd.DataFrame(results)


def main(argv: List[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Semantic search over city discourse")
    parser.add_argument("--build", action="store_true", help="Build the FAISS index")
    parser.add_argument("--query", type=str, help="Query text")
    parser.add_argument("--city", type=str, default=None, help="Optional city filter")
    parser.add_argument("--k", type=int, default=5, help="Top k results")
    args = parser.parse_args(argv)

    if args.build:
        try:
            build_index()
        except Exception as exc:
            logger.error(f"Failed to build index: {exc}")
            return 1
        return 0

    if not args.query:
        parser.error("--query required unless --build is used")

    try:
        results = search(args.query, args.city, args.k)
    except Exception as exc:
        logger.error(f"Search failed: {exc}")
        return 1

    if results.empty:
        print("No results found")
        return 0

    output = results.head(args.k)
    print(output[["score", "city", "title", "url"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

