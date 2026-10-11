"""Unified NLP pipeline orchestrator."""
from pathlib import Path
from typing import Tuple
import numpy as np
from loguru import logger

from .ner_sentiment import enrich_corpus
from .topic_modeling import topic_pipeline
from ..config import PROC_DIR, config, settings


class NLPPipeline:
    """Main NLP processing pipeline orchestrator."""
    
    def __init__(self):
        """Initialize the pipeline."""
        logger.info("Initializing NLP pipeline")
    
    def process_corpus(
        self, 
        corpus_path: Path | None = None,
        embeddings: np.ndarray | None = None,
        use_bertopic: bool | None = None
    ) -> Path:
        """
        Process corpus through complete NLP pipeline.
        
        Args:
            corpus_path: Path to corpus parquet file
            embeddings: Pre-computed embeddings (optional)
            use_bertopic: Whether to use BERTopic (None = auto-detect)
        
        Returns:
            Path to enriched corpus file
        """
        if corpus_path is None:
            corpus_path = PROC_DIR / "corpus.parquet"
        
        if not corpus_path.exists():
            logger.warning(f"Corpus not found at {corpus_path}, skipping NLP processing")
            return corpus_path
        
        logger.info(f"Processing corpus from {corpus_path}")
        
        # Step 1: NER & Sentiment Analysis
        logger.info("Running NER and sentiment analysis...")
        try:
            enriched_path = enrich_corpus(corpus_path)
            logger.info(f"NER and sentiment completed: {enriched_path}")
        except Exception as e:
            logger.error(f"Error in NER/sentiment processing: {e}", exc_info=True)
            raise
        
        # Step 2: Topic Modeling (if embeddings provided)
        if embeddings is not None:
            logger.info("Running topic modeling...")
            try:
                # Auto-detect BERTopic preference if not specified
                if use_bertopic is None:
                    topic_method = config.get_topic_modeling_method()
                    use_bertopic = (
                        settings.enable_bertopic or 
                        topic_method == "bertopic"
                    ) and len(embeddings) >= 15
                
                if use_bertopic:
                    logger.info(f"Using BERTopic for topic modeling with {len(embeddings)} documents")
                else:
                    logger.info("Using LDA for topic modeling")
                
                topic_pipeline(embeddings, use_bertopic=use_bertopic)
                logger.info("Topic modeling completed")
            except Exception as e:
                logger.warning(f"Topic modeling failed: {e}", exc_info=True)
                # Don't fail the entire pipeline if topic modeling fails
        
        return enriched_path


if __name__ == "__main__":
    pipeline = NLPPipeline()
    pipeline.process_corpus()

