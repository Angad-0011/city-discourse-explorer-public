"""
Main pipeline runner - unified approach
Combines best practices from both projects
"""
from pathlib import Path
import os
import sys

# Set tokenizer parallelism to avoid warnings
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from .config import config, settings, FAST_MODE
from .utils import setup_logging
from loguru import logger

# Setup logging
setup_logging(
    log_file=config.get('logging', {}).get('file', 'logs/pipeline.log'),
    level=config.get('logging', {}).get('level', 'INFO')
)


def run_pipeline():
    """
    Run the complete data processing pipeline
    
    Steps:
    1. Scrape Reddit and news data
    2. Preprocess and clean data
    3. Generate embeddings
    4. Run NLP processing (NER, sentiment, topics)
    5. Build search index
    6. Store in database
    """
    logger.info("=" * 60)
    logger.info("Starting Geospatial NLP Pipeline")
    logger.info("=" * 60)
    
    try:
        # Check if we have existing good corpus BEFORE scraping
        from .config import PROC_DIR
        import pandas as pd
        existing_corpus = PROC_DIR / "corpus.parquet"
        use_existing = False
        corpus_path = existing_corpus
        
        if existing_corpus.exists():
            try:
                existing_df = pd.read_parquet(existing_corpus)
                if len(existing_df) >= 50:  # If we have 50+ documents, check for new data
                    # Check if there are newer raw files that should be merged
                    from .config import RAW_DIR
                    raw_dir = RAW_DIR
                    
                    # Find latest news and reddit files
                    latest_news = sorted(raw_dir.glob("news_*.parquet"), key=lambda p: p.stat().st_mtime, reverse=True)
                    latest_reddit = sorted(raw_dir.glob("reddit*.parquet"), key=lambda p: p.stat().st_mtime, reverse=True)
                    
                    # Check if we have new data that's not in corpus
                    has_new_data = False
                    if latest_news:
                        latest_news_df = pd.read_parquet(latest_news[0])
                        # Simple check: if latest news file has more articles than in corpus
                        news_in_corpus = len(existing_df[existing_df.get('platform', '') == 'news']) if 'platform' in existing_df.columns else 0
                        if len(latest_news_df) > news_in_corpus + 10:  # Allow some margin
                            has_new_data = True
                            logger.info(f"Found new news data ({len(latest_news_df)} articles) not in corpus. Will merge.")
                    
                    if has_new_data:
                        logger.info(f"Found existing corpus with {len(existing_df)} documents, but new data available. Will merge.")
                        use_existing = False  # Force re-processing to merge
                    else:
                        logger.info(f"Found existing corpus with {len(existing_df)} documents. Using existing data.")
                        logger.info("Skipping scraping. To re-scrape, delete data/processed/corpus.parquet first.")
                        use_existing = True
                        corpus_path = existing_corpus
            except Exception as e:
                logger.warning(f"Could not read existing corpus: {e}")
        
        if not use_existing:
            # Step 1: Scraping
            logger.info("Step 1: Scraping data...")
            from .scrapers.reddit_scraper import scrape_reddit
            from .scrapers.news_scraper import scrape_news
            
            reddit_path = scrape_reddit()
            news_path = scrape_news()
            
            logger.info(f"Reddit data: {reddit_path}")
            logger.info(f"News data: {news_path}")
            
            # Step 2: Preprocessing
            logger.info("Step 2: Preprocessing data...")
            from .processors.preprocess import preprocess_data
            
            corpus_path = preprocess_data(reddit_path, news_path)
            logger.info(f"Corpus saved to {corpus_path}")
            
            # Check if we got enough data - restore if needed
            new_df = pd.read_parquet(corpus_path)
            if len(new_df) < 50:
                # Try to find a backup with more data
                from .config import RAW_DIR
                raw_dir = RAW_DIR
                backup_files = sorted(raw_dir.glob("news_*.parquet"), key=lambda p: p.stat().st_size, reverse=True)
                
                for backup_file in backup_files:
                    try:
                        backup_df = pd.read_parquet(backup_file)
                        if len(backup_df) > len(new_df) and len(backup_df) >= 50:
                            logger.warning(f"New scraping only got {len(new_df)} documents. Restoring from backup with {len(backup_df)} documents.")
                            backup_df.to_parquet(corpus_path)
                            corpus_path = existing_corpus
                            break
                    except:
                        continue
        
        # Step 3: Embeddings
        logger.info("Step 3: Generating embeddings...")
        from .processors.embeddings import build_embeddings
        
        # Use incremental updates by default
        incremental = config.get("index", {}).get("incremental", True)
        corpus, embeddings, doc_id_mapping = build_embeddings(incremental=incremental)
        logger.info(f"Generated embeddings: {embeddings.shape} (mapped {len(doc_id_mapping)} documents)")
        
        # Step 4: NLP Processing (NER, Sentiment, Topics)
        logger.info("Step 4: Running NLP processing...")
        from .processors.nlp_pipeline import NLPPipeline
        
        pipeline = NLPPipeline()
        # Pass embeddings so pipeline can run topic modeling
        pipeline.process_corpus(corpus_path, embeddings=embeddings)
        
        # Step 7: Build Search Index
        logger.info("Step 7: Building search index...")
        from .search.index_search import build_index
        
        try:
            # Use incremental updates by default
            incremental = config.get("index", {}).get("incremental", True)
            build_index(force=False, incremental=incremental)
            logger.info("Search index built successfully")
        except Exception as e:
            logger.warning(f"Index build skipped: {e}")
        
        # Step 8: Pre-compute Causal Graph (if enabled)
        if config.get("features", {}).get("enable_causal", True):
            logger.info("Step 8: Pre-computing causal graph...")
            try:
                from .causal.causal_graph import CausalGraphBuilder
                
                # Load the most enriched corpus
                enriched_corpus_paths = [
                    PROC_DIR / "corpus_bertopic.parquet",
                    PROC_DIR / "corpus_lda.parquet",
                    PROC_DIR / "corpus_ner_sent.parquet",
                    PROC_DIR / "corpus.parquet"
                ]
                
                enriched_corpus = None
                for path in enriched_corpus_paths:
                    if path.exists():
                        enriched_corpus = pd.read_parquet(path)
                        logger.info(f"Using corpus from {path.name} for causal graph")
                        break
                
                if enriched_corpus is not None and len(enriched_corpus) >= 10:
                    builder = CausalGraphBuilder()
                    min_confidence = config.get("causal", {}).get("min_confidence", 0.3)
                    graph = builder.build_graph(enriched_corpus, city=None, min_confidence=min_confidence)
                    
                    if graph and graph.number_of_nodes() > 0:
                        builder.save_graph()
                        summary = builder.get_graph_summary()
                        logger.info(f"Causal graph pre-computed: {summary.get('num_events', 0)} events, {summary.get('num_relationships', 0)} relationships")
                    else:
                        logger.warning("Causal graph could not be built (insufficient data or relationships)")
                else:
                    logger.info("Skipping causal graph: insufficient corpus data")
            except Exception as e:
                logger.warning(f"Causal graph pre-computation failed: {e}", exc_info=True)
        
        # Step 9: Store in Database
        logger.info("Step 9: Storing in database...")
        try:
            from .storage.database import DatabaseManager
            
            db_manager = DatabaseManager()
            # Import and store articles
            # (Implementation depends on your database schema)
        except Exception as e:
            logger.warning(f"Database storage skipped: {e}")
        
        logger.info("=" * 60)
        logger.info("Pipeline completed successfully!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    run_pipeline()

