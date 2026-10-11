"""NER and sentiment enrichment with structured entity storage and optional transformer sentiment."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import spacy
from nltk import download as nltk_download
from nltk.sentiment import SentimentIntensityAnalyzer

from ..config import PROC_DIR, SPACY_MODEL_SMALL, config
from ..utils_io import load_parquet, save_parquet
from loguru import logger

try:
    nltk_download("vader_lexicon", quiet=True)
except Exception:
    pass

# Optional transformer-based sentiment
try:
    from transformers import pipeline as transformers_pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    transformers_pipeline = None

_nlp = None
_sia = SentimentIntensityAnalyzer()
_transformer_sentiment_pipeline = None


def get_nlp():
    """Get or load spaCy NLP model."""
    global _nlp
    if _nlp is None:
        _nlp = spacy.load(SPACY_MODEL_SMALL)
    return _nlp


def get_transformer_sentiment(model_name: str = None):
    """Get or load transformer-based sentiment pipeline."""
    global _transformer_sentiment_pipeline
    if not TRANSFORMERS_AVAILABLE:
        return None
    
    if _transformer_sentiment_pipeline is None and model_name:
        try:
            logger.info(f"Loading transformer sentiment model: {model_name}")
            _transformer_sentiment_pipeline = transformers_pipeline(
                "sentiment-analysis",
                model=model_name,
                device=-1,  # CPU
                batch_size=32,
                truncation=True,
                max_length=512
            )
        except Exception as e:
            logger.warning(f"Failed to load transformer sentiment model: {e}")
            return None
    
    return _transformer_sentiment_pipeline


def extract_entities(text: str) -> List[Dict[str, Any]]:
    """
    Extract named entities from text with structured data.
    
    Returns:
        List of dicts with keys: 'text', 'type', 'count'
    """
    doc = get_nlp()(text)
    target_types = {"GPE", "LOC", "ORG", "PERSON", "LAW", "DATE", "EVENT"}
    
    # Count entities by (text, type)
    entity_counts = Counter()
    for ent in doc.ents:
        if ent.label_ in target_types:
            # Normalize text (lowercase, strip)
            normalized_text = ent.text.strip()
            if normalized_text:
                entity_counts[(normalized_text, ent.label_)] += 1
    
    # Convert to list of dicts
    entities = []
    for (text, ent_type), count in entity_counts.items():
        entities.append({
            "text": text,
            "type": ent_type,
            "count": count
        })
    
    return entities


def compute_sentiment_vader(text: str) -> float:
    """Compute sentiment score using VADER."""
    return _sia.polarity_scores(text).get("compound", 0.0)


def compute_sentiment_transformer(text: str, pipeline) -> float:
    """
    Compute sentiment score using transformer model.
    
    Returns:
        Normalized score: -1.0 (negative) to 1.0 (positive)
    """
    if not pipeline:
        return 0.0
    
    try:
        # Truncate if too long
        if len(text) > 512:
            text = text[:512]
        
        result = pipeline(text)[0]
        label = result.get("label", "").upper()
        score = result.get("score", 0.5)
        
        # Convert to -1 to 1 scale
        if "POSITIVE" in label or "POS" in label:
            return score  # 0.5 to 1.0 -> 0.5 to 1.0
        elif "NEGATIVE" in label or "NEG" in label:
            return -score  # 0.5 to 1.0 -> -0.5 to -1.0
        else:
            # Neutral
            return 0.0
    except Exception as e:
        logger.warning(f"Transformer sentiment failed for text: {e}")
        return 0.0


def compute_sentiment_batch(texts: List[str], method: str = "vader", model_name: str = None) -> List[float]:
    """
    Compute sentiment scores for a batch of texts.
    
    Args:
        texts: List of text strings
        method: "vader" or "transformer"
        model_name: Transformer model name (if method is "transformer")
    
    Returns:
        List of sentiment scores
    """
    if method == "vader":
        return [compute_sentiment_vader(text) for text in texts]
    
    elif method == "transformer":
        pipeline = get_transformer_sentiment(model_name)
        if not pipeline:
            logger.warning("Transformer pipeline not available, falling back to VADER")
            return [compute_sentiment_vader(text) for text in texts]
        
        try:
            # Batch processing
            results = pipeline(texts, batch_size=32, truncation=True, max_length=512)
            scores = []
            for result in results:
                label = result.get("label", "").upper()
                score = result.get("score", 0.5)
                
                if "POSITIVE" in label or "POS" in label:
                    scores.append(score)
                elif "NEGATIVE" in label or "NEG" in label:
                    scores.append(-score)
                else:
                    scores.append(0.0)
            
            return scores
        except Exception as e:
            logger.warning(f"Transformer batch sentiment failed: {e}, falling back to VADER")
            return [compute_sentiment_vader(text) for text in texts]
    
    else:
        logger.warning(f"Unknown sentiment method: {method}, using VADER")
        return [compute_sentiment_vader(text) for text in texts]


def _generate_entity_summaries(df: pd.DataFrame) -> Dict:
    """
    Generate summary artifacts for entities by city and topic.
    
    Returns:
        Dictionary with entity summaries
    """
    summaries = {
        "by_city": {},
        "by_topic": {},
        "overall": {}
    }
    
    # Overall entity statistics
    all_entities = []
    entity_type_counts = Counter()
    entity_text_counts = Counter()
    
    for _, row in df.iterrows():
        entities = row.get("entities", [])
        if isinstance(entities, list):
            for ent in entities:
                if isinstance(ent, dict):
                    all_entities.append(ent)
                    entity_type_counts[ent.get("type", "UNKNOWN")] += ent.get("count", 1)
                    entity_text_counts[ent.get("text", "")] += ent.get("count", 1)
                elif isinstance(ent, str):
                    # Legacy format: just text
                    all_entities.append({"text": ent, "type": "UNKNOWN", "count": 1})
                    entity_text_counts[ent] += 1
    
    summaries["overall"] = {
        "total_entities": len(all_entities),
        "unique_entities": len(entity_text_counts),
        "top_types": dict(entity_type_counts.most_common(10)),
        "top_entities": [
            {"text": text, "count": count}
            for text, count in entity_text_counts.most_common(20)
        ]
    }
    
    # By city
    if "city" in df.columns:
        city_entities = defaultdict(lambda: {"entities": Counter(), "types": Counter()})
        
        for _, row in df.iterrows():
            city = row.get("city", "Unknown")
            if pd.isna(city):
                city = "Unknown"
            
            entities = row.get("entities", [])
            if isinstance(entities, list):
                for ent in entities:
                    if isinstance(ent, dict):
                        text = ent.get("text", "")
                        ent_type = ent.get("type", "UNKNOWN")
                        count = ent.get("count", 1)
                        city_entities[city]["entities"][text] += count
                        city_entities[city]["types"][ent_type] += count
                    elif isinstance(ent, str):
                        city_entities[city]["entities"][ent] += 1
        
        for city, data in city_entities.items():
            summaries["by_city"][city] = {
                "total_entities": sum(data["entities"].values()),
                "unique_entities": len(data["entities"]),
                "top_types": dict(data["types"].most_common(10)),
                "top_entities": [
                    {"text": text, "count": count}
                    for text, count in data["entities"].most_common(20)
                ]
            }
    
    # By topic (BERTopic)
    if "bertopic" in df.columns:
        topic_entities = defaultdict(lambda: {"entities": Counter(), "types": Counter()})
        
        for _, row in df.iterrows():
            topic = row.get("bertopic", -1)
            if pd.isna(topic) or topic == -1:
                continue
            
            entities = row.get("entities", [])
            if isinstance(entities, list):
                for ent in entities:
                    if isinstance(ent, dict):
                        text = ent.get("text", "")
                        ent_type = ent.get("type", "UNKNOWN")
                        count = ent.get("count", 1)
                        topic_entities[topic]["entities"][text] += count
                        topic_entities[topic]["types"][ent_type] += count
                    elif isinstance(ent, str):
                        topic_entities[topic]["entities"][ent] += 1
        
        for topic, data in topic_entities.items():
            summaries["by_topic"][f"topic_{topic}"] = {
                "total_entities": sum(data["entities"].values()),
                "unique_entities": len(data["entities"]),
                "top_types": dict(data["types"].most_common(10)),
                "top_entities": [
                    {"text": text, "count": count}
                    for text, count in data["entities"].most_common(20)
                ]
            }
    
    # By topic (LDA)
    if "lda_topic" in df.columns:
        lda_topic_entities = defaultdict(lambda: {"entities": Counter(), "types": Counter()})
        
        for _, row in df.iterrows():
            topic = row.get("lda_topic", -1)
            if pd.isna(topic) or topic == -1:
                continue
            
            entities = row.get("entities", [])
            if isinstance(entities, list):
                for ent in entities:
                    if isinstance(ent, dict):
                        text = ent.get("text", "")
                        ent_type = ent.get("type", "UNKNOWN")
                        count = ent.get("count", 1)
                        lda_topic_entities[topic]["entities"][text] += count
                        lda_topic_entities[topic]["types"][ent_type] += count
                    elif isinstance(ent, str):
                        lda_topic_entities[topic]["entities"][ent] += 1
        
        for topic, data in lda_topic_entities.items():
            summaries["by_topic"][f"lda_topic_{topic}"] = {
                "total_entities": sum(data["entities"].values()),
                "unique_entities": len(data["entities"]),
                "top_types": dict(data["types"].most_common(10)),
                "top_entities": [
                    {"text": text, "count": count}
                    for text, count in data["entities"].most_common(20)
                ]
            }
    
    return summaries


def enrich_corpus(corpus_path: Path | None = None) -> Path:
    """
    Enrich corpus with NER and sentiment.
    
    Uses structured entity storage (text, type, count) and supports
    both VADER (default) and transformer-based sentiment analysis.
    """
    if corpus_path is None:
        corpus_path = PROC_DIR / "corpus.parquet"
    if not corpus_path.exists():
        raise FileNotFoundError("Corpus not found. Run preprocess step first.")

    df = load_parquet(corpus_path)
    logger.info(f"Enriching {len(df)} documents with NER and sentiment")
    
    # Use available text column
    text_col = df.get("clean_text", df.get("text", ""))
    
    # Extract entities with structured data
    logger.info("Extracting named entities...")
    df["entities"] = text_col.fillna("").apply(extract_entities)
    
    # Compute sentiment
    sentiment_method = config.get("nlp.sentiment.method", "vader")
    sentiment_model = config.get("nlp.models.sentiment", None)
    
    logger.info(f"Computing sentiment using {sentiment_method}...")
    
    if sentiment_method == "transformer" and TRANSFORMERS_AVAILABLE:
        # Batch processing for transformer
        texts = text_col.fillna("").tolist()
        df["sentiment"] = compute_sentiment_batch(texts, method="transformer", model_name=sentiment_model)
    else:
        # VADER (default, row-by-row is fine)
        df["sentiment"] = text_col.fillna("").apply(compute_sentiment_vader)
    
    # Save enriched corpus
    out_path = PROC_DIR / "corpus_ner_sent.parquet"
    save_parquet(df, out_path)
    logger.info(f"NER + sentiment saved to {out_path}")
    
    # Generate entity summaries
    logger.info("Generating entity summaries...")
    summaries = _generate_entity_summaries(df)
    
    # Save summaries
    summary_path = PROC_DIR / "entity_summaries.json"
    with open(summary_path, "w") as f:
        json.dump(summaries, f, indent=2, default=str)
    logger.info(f"Entity summaries saved to {summary_path}")
    
    return out_path


if __name__ == "__main__":
    path = enrich_corpus()
    logger.info(f"NER + sentiment saved to {path}")
