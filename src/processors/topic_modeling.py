"""Topic modeling (LDA + BERTopic)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ..config import ENABLE_BERTOPIC, FAST_MODE, MODELS_DIR, PROC_DIR, TOPIC_PARAMS
from ..utils_io import load_parquet, save_parquet
from loguru import logger

try:
    from gensim.corpora.dictionary import Dictionary
    from gensim.models import LdaModel
    from gensim.utils import simple_preprocess
except ImportError as exc:
    Dictionary = None
    LdaModel = None
    simple_preprocess = None
    _GENSIM_ERROR = exc
else:
    _GENSIM_ERROR = None

try:
    from bertopic import BERTopic
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP
    import hdbscan
except ImportError:
    BERTopic = None
    CountVectorizer = None
    UMAP = None
    hdbscan = None


def _tokenize(text: str) -> list[str]:
    """
    Tokenize text using Gensim's simple_preprocess.
    
    Handles:
    - Lowercasing
    - Punctuation removal
    - Stop word filtering (optional, but better than split)
    - Minimum word length filtering
    """
    if simple_preprocess is None:
        # Fallback to basic split if gensim not available
        return [token.lower().strip() for token in text.split() if token.strip()]
    
    # Use Gensim's simple_preprocess: lowercases, removes punctuation, filters by length
    # deacc=True removes accents, min_len=2 filters very short tokens
    tokens = simple_preprocess(str(text), deacc=True, min_len=2, max_len=50)
    return tokens


def _generate_topic_labels(topic_model, num_topics: int) -> Dict[int, str]:
    """
    Generate automatic topic labels from keywords.
    
    Args:
        topic_model: Trained BERTopic model
        num_topics: Number of topics
    
    Returns:
        Dictionary mapping topic_id -> label
    """
    labels = {}
    try:
        topics_dict = topic_model.get_topics()
        
        for topic_id in range(num_topics):
            if topic_id in topics_dict:
                # Get top keywords for this topic
                keywords = topics_dict[topic_id][:5]  # Top 5 keywords
                keyword_str = ", ".join([word for word, _ in keywords])
                
                # Create readable label
                label = keyword_str.title()
                if len(label) > 60:
                    label = label[:57] + "..."
                
                labels[topic_id] = label
            else:
                labels[topic_id] = f"Topic {topic_id}"
        
        # Add outlier label
        labels[-1] = "Outliers"
        
    except Exception as e:
        logger.warning(f"Error generating topic labels: {e}")
        # Fallback: simple numeric labels
        for topic_id in range(num_topics):
            labels[topic_id] = f"Topic {topic_id}"
        labels[-1] = "Outliers"
    
    return labels


def _calculate_topic_coherence(topic_model, texts: List[str], topics: np.ndarray) -> Dict[int, float]:
    """
    Calculate topic coherence scores.
    
    Uses simple coherence metric: average pairwise word co-occurrence.
    
    Args:
        topic_model: Trained BERTopic model
        texts: List of texts
        topics: Topic assignments
    
    Returns:
        Dictionary mapping topic_id -> coherence score
    """
    coherence_scores = {}
    
    try:
        topics_dict = topic_model.get_topics()
        
        for topic_id, topic_words in topics_dict.items():
            if topic_id == -1:
                coherence_scores[-1] = None  # Outliers don't have coherence
                continue
            
            if len(topic_words) < 2:
                coherence_scores[topic_id] = None
                continue
            
            # Get top words for this topic
            top_words = [word for word, _ in topic_words[:10]]
            
            # Simple coherence: count co-occurrences of top words in documents
            cooccurrence_count = 0
            total_pairs = 0
            
            # Get documents assigned to this topic
            topic_docs = [texts[i] for i, t in enumerate(topics) if t == topic_id]
            
            if len(topic_docs) < 2:
                coherence_scores[topic_id] = None
                continue
            
            # Count word co-occurrences in documents
            for doc in topic_docs[:100]:  # Limit to first 100 docs for performance
                doc_words = set(doc.lower().split())
                for i, word1 in enumerate(top_words[:5]):
                    for word2 in top_words[i+1:6]:
                        total_pairs += 1
                        if word1.lower() in doc_words and word2.lower() in doc_words:
                            cooccurrence_count += 1
            
            # Calculate coherence (normalized co-occurrence rate)
            if total_pairs > 0:
                coherence = cooccurrence_count / total_pairs
                coherence_scores[topic_id] = float(coherence)
            else:
                coherence_scores[topic_id] = None
                
    except Exception as e:
        logger.warning(f"Error calculating topic coherence: {e}")
        return {}
    
    return coherence_scores


def _generate_city_topic_slices(corpus_df: pd.DataFrame, topics: np.ndarray, topic_labels: Dict[int, str]) -> Dict[str, Dict]:
    """
    Generate city-specific topic distributions.
    
    Args:
        corpus_df: Corpus dataframe with city column
        topics: Topic assignments
        topic_labels: Topic ID to label mapping
    
    Returns:
        Dictionary mapping city -> topic distribution
    """
    city_slices = {}
    
    if "city" not in corpus_df.columns:
        return city_slices
    
    # Group by city
    for city in corpus_df["city"].dropna().unique():
        if not city:
            continue
        
        city_mask = corpus_df["city"] == city
        city_topics = topics[city_mask]
        
        if len(city_topics) == 0:
            continue
        
        # Count topic frequencies
        topic_counts = {}
        for topic_id in city_topics:
            topic_counts[topic_id] = topic_counts.get(topic_id, 0) + 1
        
        # Calculate percentages
        total = len(city_topics)
        topic_distribution = {
            topic_labels.get(topic_id, f"Topic {topic_id}"): {
                "count": count,
                "percentage": round(100 * count / total, 2)
            }
            for topic_id, count in topic_counts.items()
        }
        
        # Sort by count
        topic_distribution = dict(sorted(topic_distribution.items(), key=lambda x: x[1]["count"], reverse=True))
        
        city_slices[str(city)] = {
            "total_documents": total,
            "topics": topic_distribution,
            "top_topic": max(topic_counts.items(), key=lambda x: x[1])[0] if topic_counts else None
        }
    
    return city_slices


def run_lda(corpus_df: pd.DataFrame) -> Tuple[pd.DataFrame, Path]:
    """Run LDA topic modeling."""
    if Dictionary is None or LdaModel is None:
        logger.warning("gensim not installed. Install via `pip install gensim`. Skipping LDA.")
        corpus_df["lda_topic"] = -1
        out_path = PROC_DIR / "corpus_lda.parquet"
        save_parquet(corpus_df, out_path)
        return corpus_df, out_path

    if "lemma_text" not in corpus_df.columns:
        logger.info("No lemma_text column, using text or clean_text")
        text_col = corpus_df.get("lemma_text", corpus_df.get("clean_text", corpus_df.get("text", "")))
    else:
        text_col = corpus_df["lemma_text"]
    
    tokens = text_col.fillna("").apply(_tokenize)
    
    # Handle case with very few documents
    if len(tokens) < 2:
        logger.warning(f"Too few documents ({len(tokens)}) for LDA. Assigning default topic.")
        corpus_df["lda_topic"] = 0
        out_path = PROC_DIR / "corpus_lda.parquet"
        save_parquet(corpus_df, out_path)
        return corpus_df, out_path
    
    dictionary = Dictionary(tokens)
    bow_corpus = [dictionary.doc2bow(text) for text in tokens]
    
    # Adjust number of topics if we have fewer documents than topics
    num_topics = min(TOPIC_PARAMS["num_topics"], max(1, len(tokens) - 1))
    if num_topics < TOPIC_PARAMS["num_topics"]:
        logger.info(f"Reducing topics from {TOPIC_PARAMS['num_topics']} to {num_topics} due to small dataset")
    
    logger.info(f"Training LDA with {num_topics} topics on {len(tokens)} documents")
    lda_model = LdaModel(
        corpus=bow_corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=TOPIC_PARAMS["passes"],
        iterations=TOPIC_PARAMS["iterations"],
        random_state=42,
    )

    topics = [int(max(lda_model[bow], key=lambda x: x[1])[0]) if bow else -1 for bow in bow_corpus]
    corpus_df["lda_topic"] = topics

    # Generate automatic topic labels
    lda_labels = {}
    for topic_id in range(num_topics):
        topic_words = lda_model.show_topic(topic_id, topn=5)
        keyword_str = ", ".join([word for word, _ in topic_words])
        lda_labels[topic_id] = keyword_str.title()[:60]
    
    # Calculate topic coherence for LDA
    try:
        from gensim.models import CoherenceModel
        coherence_model = CoherenceModel(
            model=lda_model,
            texts=tokens.tolist(),
            dictionary=dictionary,
            coherence='c_v'
        )
        coherence_score = coherence_model.get_coherence()
        logger.info(f"LDA topic coherence (c_v): {coherence_score:.4f}")
        
        # Per-topic coherence
        per_topic_coherence = coherence_model.get_coherence_per_topic()
        coherence_scores = {i: float(score) for i, score in enumerate(per_topic_coherence)}
    except Exception as e:
        logger.warning(f"Could not calculate LDA coherence: {e}")
        coherence_score = None
        coherence_scores = {}
    
    # Save topic info
    lda_topic_info = {
        "labels": lda_labels,
        "coherence": coherence_score,
        "per_topic_coherence": coherence_scores,
        "num_topics": num_topics
    }
    from ..utils_io import save_json
    topic_info_path = MODELS_DIR / "lda_topic_info.json"
    save_json(lda_topic_info, topic_info_path)
    logger.info(f"Saved LDA topic labels and coherence to {topic_info_path}")
    
    # Generate city-specific topic slices for LDA
    if "city" in corpus_df.columns:
        city_topic_slices = _generate_city_topic_slices(corpus_df, np.array(topics), lda_labels)
        city_slices_path = MODELS_DIR / "lda_city_topic_slices.json"
        save_json(city_topic_slices, city_slices_path)
        logger.info(f"Generated LDA city-specific topic slices: {len(city_topic_slices)} cities")

    MODELS_DIR.mkdir(exist_ok=True, parents=True)
    lda_model.save(str(MODELS_DIR / "lda_model"))
    dictionary.save(str(MODELS_DIR / "lda_dictionary.dict"))

    out_path = PROC_DIR / "corpus_lda.parquet"
    save_parquet(corpus_df, out_path)
    return corpus_df, out_path


def run_bertopic(corpus_df: pd.DataFrame, embeddings: np.ndarray | None = None) -> Path | None:
    """
    Run BERTopic modeling with improved logic and reporting.
    
    Logic:
    - ENABLE_BERTOPIC=False: Skip (highest priority)
    - ENABLE_BERTOPIC=True: Run (overrides FAST_MODE)
    - FAST_MODE=True + large dataset: Skip only if ENABLE_BERTOPIC not explicitly set
    """
    # Priority 1: Check if BERTopic is explicitly disabled
    if not ENABLE_BERTOPIC:
        logger.info("ENABLE_BERTOPIC=0 -> skipping BERTopic")
        return None
    
    # Priority 2: Check if BERTopic library is available
    if BERTopic is None:
        logger.warning("bertopic not installed. Install via `pip install bertopic`. Skipping BERTopic")
        return None
    
    # Priority 3: Check minimum document requirement
    if len(corpus_df) < 15:
        logger.warning(f"Dataset too small ({len(corpus_df)} docs) for BERTopic. Need at least 15 documents. Skipping BERTopic.")
        return None
    
    # Priority 4: FAST_MODE check (only applies if ENABLE_BERTOPIC is not explicitly True)
    # If ENABLE_BERTOPIC=True, it overrides FAST_MODE
    if FAST_MODE and len(corpus_df) > 200:
        # In FAST_MODE with large dataset, skip unless explicitly enabled
        # Since we already checked ENABLE_BERTOPIC above, if we reach here it's enabled
        logger.info(f"FAST_MODE=1 with large dataset ({len(corpus_df)} docs), but ENABLE_BERTOPIC=1 -> running BERTopic")
    if embeddings is None:
        emb_path = MODELS_DIR / "sbert_embeddings.npy"
        if not emb_path.exists():
            logger.warning("Embeddings file missing for BERTopic")
            return None
        embeddings = np.load(emb_path)
    
    # Check embeddings shape matches corpus
    if embeddings.shape[0] != len(corpus_df):
        logger.warning(f"Embeddings shape {embeddings.shape[0]} doesn't match corpus size {len(corpus_df)}. Skipping BERTopic.")
        return None

    if "lemma_text" not in corpus_df.columns:
        text_col = corpus_df.get("lemma_text", corpus_df.get("clean_text", corpus_df.get("text", "")))
    else:
        text_col = corpus_df["lemma_text"]
    
    logger.info(f"Training BERTopic on {len(corpus_df)} documents with {embeddings.shape[1]}-dimensional embeddings...")
    topic_model = BERTopic(
        umap_model=UMAP(random_state=42, n_neighbors=15, n_components=5, min_dist=0.0),
        hdbscan_model=hdbscan.HDBSCAN(min_cluster_size=5, prediction_data=True),
        vectorizer_model=CountVectorizer(stop_words="english"),
        verbose=False,
    )
    topics, probs = topic_model.fit_transform(text_col.tolist(), embeddings)
    
    # Ensure topics is a numpy array
    topics = np.array(topics) if not isinstance(topics, np.ndarray) else topics
    corpus_df["bertopic"] = topics
    
    # Count outliers (topic -1)
    num_outliers = int(np.sum(topics == -1)) if len(topics) > 0 else 0
    num_topics = len(set(topics)) - (1 if -1 in topics else 0)  # Exclude outlier topic -1
    logger.info(f"BERTopic training complete! Found {num_topics} topics (+ {num_outliers} outliers)")
    
    # Generate automatic topic labels
    topic_labels = _generate_topic_labels(topic_model, num_topics)
    logger.info(f"Generated labels for {len(topic_labels)} topics")
    
    # Calculate and report topic coherence
    coherence_scores = _calculate_topic_coherence(topic_model, text_col.tolist(), topics)
    if coherence_scores:
        avg_coherence = np.mean([score for score in coherence_scores.values() if score is not None])
        logger.info(f"Average topic coherence: {avg_coherence:.4f}")
        logger.info(f"Topic coherence range: {min([s for s in coherence_scores.values() if s is not None]):.4f} - {max([s for s in coherence_scores.values() if s is not None]):.4f}")
    
    # Save topic labels and coherence
    topic_info = {
        "labels": topic_labels,
        "coherence": coherence_scores,
        "num_topics": num_topics,
        "num_outliers": num_outliers
    }
    topic_info_path = MODELS_DIR / "bertopic_topic_info.json"
    from ..utils_io import save_json
    save_json(topic_info, topic_info_path)
    logger.info(f"Saved topic labels and coherence to {topic_info_path}")

    MODELS_DIR.mkdir(exist_ok=True, parents=True)
    topic_model.save(str(MODELS_DIR / "bertopic_model"))
    logger.info(f"Saved BERTopic model to {MODELS_DIR / 'bertopic_model'}")

    out_path = PROC_DIR / "corpus_bertopic.parquet"
    save_parquet(corpus_df, out_path)
    logger.info(f"Saved BERTopic corpus to {out_path}")
    
    # Generate city-specific topic slices
    if "city" in corpus_df.columns:
        city_topic_slices = _generate_city_topic_slices(corpus_df, topics, topic_labels)
        city_slices_path = MODELS_DIR / "city_topic_slices.json"
        save_json(city_topic_slices, city_slices_path)
        logger.info(f"Generated city-specific topic slices: {len(city_topic_slices)} cities")
    
    return out_path


def topic_pipeline(embeddings: np.ndarray | None = None, use_bertopic: bool = True) -> tuple[pd.DataFrame, Path | None]:
    """Run complete topic modeling pipeline."""
    corpus_path = PROC_DIR / "corpus.parquet"
    if not corpus_path.exists():
        raise FileNotFoundError("corpus.parquet missing. Run preprocessing first.")
    corpus = load_parquet(corpus_path)
    corpus, lda_path = run_lda(corpus)
    
    # Only run BERTopic if explicitly requested and embeddings are available
    bertopic_path = None
    if use_bertopic and embeddings is not None:
        bertopic_path = run_bertopic(corpus.copy(), embeddings)
    elif use_bertopic and ENABLE_BERTOPIC:
        # Try to load embeddings if not provided
        bertopic_path = run_bertopic(corpus.copy(), None)
    
    return corpus, bertopic_path or lda_path


if __name__ == "__main__":
    topic_pipeline()

