"""Data preprocessing pipeline."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import spacy
from nltk import download as nltk_download
from nltk.corpus import stopwords

from ..config import config, FAST_MODE, PROC_DIR, RAW_DIR, SPACY_MODEL_SMALL
from ..city_registry import get_city_registry
from ..utils_io import load_df, save_df
from loguru import logger

# Text cleaning regex patterns
CLEAN_REGEX = re.compile(r"https?://\S+|www\.\S+")
PUNCT_REGEX = re.compile(r"[^A-Za-z0-9\s]")

# Get city registry
city_registry = get_city_registry()

try:
    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    nltk_download("stopwords", quiet=True)
    STOPWORDS = set(stopwords.words("english"))

_nlp = None


def get_nlp():
    """Get or load spaCy NLP model."""
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load(SPACY_MODEL_SMALL, disable=["parser", "textcat"])
        except OSError as exc:
            logger.error(
                "spaCy model missing. Run `python -m spacy download en_core_web_sm` or set SPACY_MODEL_SMALL."
            )
            raise exc
    return _nlp


def clean_text(text: str) -> str:
    """
    Clean text by removing URLs and special characters.
    Consolidated implementation - used throughout the codebase.
    """
    if not text:
        return ""
    
    # Remove URLs
    text = CLEAN_REGEX.sub(" ", text)
    # Remove special characters (keep alphanumeric and spaces)
    text = PUNCT_REGEX.sub(" ", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def lemmatize_batch(texts: List[str], batch_size: int = 1000) -> List[str]:
    """
    Lemmatize texts using spaCy batch processing (nlp.pipe).
    Much faster than processing one at a time.
    
    Args:
        texts: List of texts to lemmatize
        batch_size: Batch size for processing
    
    Returns:
        List of lemmatized texts
    """
    nlp = get_nlp()
    results = []
    
    # Process in batches using nlp.pipe for efficiency
    for doc in nlp.pipe(texts, batch_size=batch_size, disable=["parser", "textcat"]):
        tokens = [token.lemma_ for token in doc if token.lemma_ and token.lemma_ not in STOPWORDS]
        results.append(" ".join(tokens))
    
    return results


def lemmatize(text: str) -> str:
    """
    Lemmatize single text (for backward compatibility).
    For batch processing, use lemmatize_batch() instead.
    """
    nlp = get_nlp()
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if token.lemma_ and token.lemma_ not in STOPWORDS]
    return " ".join(tokens)


def guess_city(text: str, existing: str | None = None) -> str | None:
    """
    Guess city from text using city registry (regex and NER).
    Returns standardized city name.
    
    Args:
        text: Text to search for city
        existing: Existing city name to normalize
    
    Returns:
        Standard city name or None
    """
    # If existing city provided, normalize it
    if existing:
        return city_registry.normalize(existing)
    
    # Detect from text using registry
    return city_registry.detect_from_text(text, use_ner=True)


def preprocess_frames(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """
    Preprocess multiple dataframes with batch processing for performance.
    Uses nlp.pipe() for efficient batch lemmatization.
    """
    records: List[pd.DataFrame] = []
    for frame in frames:
        if frame.empty:
            continue
        frame = frame.copy()
        frame["text"] = frame["text"].fillna("")
        
        # Clean text (vectorized operation)
        frame["clean_text"] = frame["text"].apply(clean_text)
        
        # Batch lemmatization using nlp.pipe() for performance
        clean_texts = frame["clean_text"].tolist()
        logger.info(f"Lemmatizing {len(clean_texts)} texts in batch...")
        frame["lemma_text"] = lemmatize_batch(clean_texts, batch_size=1000)
        
        # City detection (vectorized)
        frame["city"] = frame.apply(lambda row: guess_city(row["clean_text"], row.get("city")), axis=1)
        
        records.append(frame)
    
    if not records:
        return pd.DataFrame(columns=["platform", "title", "lemma_text", "city"])
    
    df = pd.concat(records, ignore_index=True)
    logger.info(f"Preprocessed {len(df):,} rows")
    return df


def preprocess_data(reddit_path: Path | None = None, news_path: Path | None = None) -> Path:
    """Preprocess Reddit and news data."""
    if FAST_MODE and reddit_path is None and news_path is None:
        base = Path(__file__).resolve().parent.parent.parent / "data" / "sample"
        reddit_path = base / "sample_reddit.parquet"
        news_path = base / "sample_news.parquet"
        if reddit_path.exists() or news_path.exists():
            logger.info("FAST_MODE=1 -> using sample parquet files for preprocessing")
        else:
            logger.info("FAST_MODE=1 but no sample data found, will try raw data")

    frames: List[pd.DataFrame] = []
    for path in filter(None, [reddit_path, news_path]):
        if path and Path(path).exists():
            frames.append(load_df(path))

    if not frames:
        raw_files = sorted(RAW_DIR.glob("*.parquet"))
        if not raw_files:
            # Create empty corpus if no data available
            logger.warning("No raw data available, creating empty corpus")
            df = pd.DataFrame(columns=["platform", "title", "lemma_text", "city", "url", "text", "clean_text"])
        else:
            frames = [load_df(path) for path in raw_files]
            df = preprocess_frames(frames)
    else:
        df = preprocess_frames(frames)
    
    # Merge with existing corpus if it exists
    existing_corpus = PROC_DIR / "corpus.parquet"
    if existing_corpus.exists() and len(df) > 0:
        try:
            existing_df = load_df(existing_corpus)
            logger.info(f"Merging {len(df)} new documents with {len(existing_df)} existing documents")
            
            # Combine dataframes, removing duplicates based on URL or title+text hash
            # Simple deduplication: if URL exists, use it; otherwise use title+text
            if 'url' in df.columns and 'url' in existing_df.columns:
                # Remove duplicates by URL
                existing_urls = set(existing_df['url'].dropna().astype(str))
                df = df[~df['url'].astype(str).isin(existing_urls)]
                logger.info(f"After URL deduplication: {len(df)} new documents")
            
            # Align columns to prevent schema mismatch (NaNs from missing columns)
            # Get union of all columns from both dataframes
            all_columns = set(existing_df.columns) | set(df.columns)
            
            # Ensure both dataframes have the same columns (fill missing with NaN)
            for col in all_columns:
                if col not in existing_df.columns:
                    existing_df[col] = None
                if col not in df.columns:
                    df[col] = None
            
            # Reorder columns consistently (use existing_df's column order as base, add new ones at end)
            common_cols = [c for c in existing_df.columns if c in df.columns]
            new_cols = [c for c in df.columns if c not in existing_df.columns]
            column_order = common_cols + new_cols
            
            existing_df = existing_df[column_order]
            df = df[column_order]
            
            # Combine
            if len(df) > 0:
                df = pd.concat([existing_df, df], ignore_index=True)
                logger.info(f"Final merged corpus: {len(df)} documents")
            else:
                logger.info("All new documents were duplicates, using existing corpus")
                df = existing_df
        except Exception as e:
            logger.warning(f"Could not merge with existing corpus: {e}. Using new data only.")
    
    corpus_path = PROC_DIR / "corpus.parquet"
    save_df(df, corpus_path)
    small_path = PROC_DIR / "corpus_small.parquet"
    columns = [col for col in ["platform", "title", "lemma_text", "city", "url"] if col in df.columns]
    if columns:
        save_df(df[columns], small_path)
    return corpus_path


if __name__ == "__main__":
    out = preprocess_data()
    logger.info(f"Preprocessed corpus saved to {out}")

