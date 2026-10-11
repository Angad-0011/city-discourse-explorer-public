"""Utility to extract and manage BERTopic labels."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from .config import MODELS_DIR
from loguru import logger

try:
    from bertopic import BERTopic
except ImportError:
    BERTopic = None


def get_topic_labels(topic_id: int, topic_model: Optional[BERTopic] = None) -> str:
    """
    Get human-readable label for a topic.
    
    Args:
        topic_id: Topic ID (can be -1 for outliers)
        topic_model: BERTopic model instance (will load if None)
    
    Returns:
        Topic label string (e.g., "Pollution & Air Quality")
    """
    if topic_id == -1:
        return "Outliers"
    
    if topic_model is None:
        topic_model = load_bertopic_model()
        if topic_model is None:
            return f"BERTopic {topic_id}"
    
    try:
        # Get topic keywords
        topics = topic_model.get_topics()
        if topic_id not in topics:
            return f"BERTopic {topic_id}"
        
        # Get top 3 keywords
        keywords = topics[topic_id][:3]
        keyword_str = ", ".join([word for word, _ in keywords])
        
        # Create a readable label (capitalize first letter, limit length)
        label = keyword_str.title()
        if len(label) > 50:
            label = label[:47] + "..."
        
        return label
    except Exception as e:
        logger.warning(f"Error getting topic label for {topic_id}: {e}")
        return f"BERTopic {topic_id}"


def load_bertopic_model() -> Optional[BERTopic]:
    """Load BERTopic model from disk."""
    if BERTopic is None:
        return None
    
    bertopic_path = MODELS_DIR / "bertopic_model"
    if not bertopic_path.exists():
        return None
    
    try:
        return BERTopic.load(str(bertopic_path))
    except Exception as e:
        logger.warning(f"Error loading BERTopic model: {e}")
        return None


def get_all_topic_labels() -> Dict[int, str]:
    """
    Get all topic labels as a dictionary.
    
    Returns:
        Dictionary mapping topic_id -> label
    """
    topic_model = load_bertopic_model()
    if topic_model is None:
        return {}
    
    labels = {}
    try:
        topics = topic_model.get_topics()
        for topic_id in topics.keys():
            if topic_id == -1:
                labels[-1] = "Outliers"
            else:
                keywords = topics[topic_id][:3]
                keyword_str = ", ".join([word for word, _ in keywords])
                label = keyword_str.title()
                if len(label) > 50:
                    label = label[:47] + "..."
                labels[topic_id] = label
    except Exception as e:
        logger.warning(f"Error getting all topic labels: {e}")
    
    return labels

