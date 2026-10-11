"""Reddit scraping utilities - uses PRAW method (snscrape is deprecated)."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from ..config import FAST_MODE, RAW_DIR, config
from ..utils_io import save_parquet
from loguru import logger


def scrape_reddit(force: bool = False) -> Path:
    """
    Scrape Reddit data using PRAW API.
    
    Requires Reddit API credentials in .env or environment variables:
    - REDDIT_CLIENT_ID
    - REDDIT_CLIENT_SECRET
    - REDDIT_USER_AGENT
    
    Checks config/config.yaml for scraping.reddit.method (should be "praw").
    """
    # Check which method to use
    method = config.get_reddit_method()
    
    # Use PRAW (required - snscrape is deprecated)
    if method == "praw":
        try:
            from .reddit_scraper_praw import scrape_reddit_praw
            logger.info("Using PRAW method for Reddit scraping")
            return scrape_reddit_praw(force=force)
        except ImportError:
            logger.error("PRAW scraper not available. Install praw: pip install praw")
            raise
        except ValueError as e:
            logger.error(f"PRAW credentials not set: {e}")
            logger.error("Please set REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, and REDDIT_USER_AGENT in .env")
            raise
    else:
        # Method is not "praw" - warn and try PRAW anyway
        logger.warning(f"Reddit scraping method '{method}' is deprecated. Using PRAW instead.")
        try:
            from .reddit_scraper_praw import scrape_reddit_praw
            logger.info("Using PRAW method for Reddit scraping")
            return scrape_reddit_praw(force=force)
        except (ImportError, ValueError) as e:
            logger.error(f"PRAW scraping failed: {e}")
            logger.error("Please configure PRAW credentials. snscrape is no longer supported.")
            raise
    
    # If we reach here, PRAW failed and FAST_MODE didn't help
    # Create empty dataset
    logger.warning("No Reddit data available. Creating empty dataset.")
    df = pd.DataFrame(columns=["platform", "title", "text"])
    
    from datetime import timezone
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = RAW_DIR / f"reddit_{timestamp}.parquet"
    save_parquet(df, out_path)
    logger.info(f"Empty Reddit dataset saved to {out_path}")
    return out_path


if __name__ == "__main__":
    path = scrape_reddit()
    logger.info(f"Reddit data saved to {path}")

