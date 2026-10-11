"""Reddit scraping using PRAW (Python Reddit API Wrapper)."""
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Set

import pandas as pd
import praw
from prawcore.exceptions import PrawcoreException, RequestException, ResponseException

from ..config import config, settings, FAST_MODE, RAW_DIR
from ..utils import extract_city_from_text, clean_text, normalize_city_name
from ..utils_io import save_parquet
from loguru import logger


class RedditScraperPRAW:
    """Reddit scraper using PRAW API with retry logic and improved metadata."""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        """
        Initialize PRAW Reddit client.
        
        Args:
            max_retries: Maximum number of retries for failed requests
            base_delay: Base delay in seconds for exponential backoff
        """
        if not settings.reddit_client_id or not settings.reddit_client_secret:
            raise ValueError(
                "Reddit API credentials not set. "
                "Please set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET in .env file"
            )
        
        self.max_retries = max_retries
        self.base_delay = base_delay
        
        try:
            self.reddit = praw.Reddit(
                client_id=settings.reddit_client_id,
                client_secret=settings.reddit_client_secret,
                user_agent=settings.reddit_user_agent
            )
            self.reddit.read_only = True
            logger.info("Reddit API client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Reddit client: {e}")
            raise
    
    def _retry_with_backoff(self, func, *args, **kwargs):
        """
        Execute function with exponential backoff retry logic.
        
        Args:
            func: Function to execute
            *args, **kwargs: Arguments to pass to function
        
        Returns:
            Function result
        
        Raises:
            Exception: If all retries fail
        """
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except (RequestException, ResponseException, PrawcoreException) as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    delay = self.base_delay * (2 ** attempt)
                    logger.warning(f"Request failed (attempt {attempt + 1}/{self.max_retries}): {e}. Retrying in {delay:.2f}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"Request failed after {self.max_retries} attempts: {e}")
            except Exception as e:
                # Non-retryable exceptions
                logger.error(f"Non-retryable error: {e}")
                raise
        
        if last_exception:
            raise last_exception
    
    def scrape(
        self,
        subreddits: Optional[List[str]] = None,
        keywords: Optional[List[str]] = None,
        cities: Optional[List[str]] = None,
        time_filter: str = "month",
        limit: int = 1000,
        include_comments: bool = False,
        max_comments_per_post: int = 10
    ) -> Path:
        """
        Scrape Reddit posts and optionally comments.
        
        Args:
            subreddits: List of subreddit names
            keywords: List of keywords to search for
            cities: List of cities to filter by
            time_filter: Time filter for top posts (hour, day, week, month, year, all)
            limit: Maximum number of posts per subreddit
            include_comments: Whether to scrape comments from posts
            max_comments_per_post: Maximum comments to scrape per post
        
        Returns:
            Path to saved parquet file
        """
        if FAST_MODE and not subreddits:
            sample_path = Path(__file__).resolve().parent.parent.parent / "data" / "sample" / "sample_reddit.parquet"
            if sample_path.exists():
                logger.info(f"FAST_MODE=1 -> using sample data at {sample_path}")
                return sample_path
        
        if not subreddits:
            subreddits = config.get_subreddits() or ['india', 'bangalore', 'chennai', 'mumbai']
        
        if not keywords:
            keywords = config.get_keywords() or ['waste management', 'flood', 'traffic']
        
        if not cities:
            cities = config.get_cities() or ['bangalore', 'chennai', 'mumbai', 'delhi']
        
        scraped_data = []
        seen_items: Set[str] = set()  # For robust deduplication (post_id + url)
        scrape_timestamp = datetime.now(timezone.utc)
        
        # Scrape from each subreddit
        for subreddit_name in subreddits:
            try:
                logger.info(f"Scraping r/{subreddit_name}...")
                
                # Use retry logic for subreddit access
                subreddit = self._retry_with_backoff(lambda: self.reddit.subreddit(subreddit_name))
                
                # Get hot posts
                posts_scraped = 0
                try:
                    hot_posts = self._retry_with_backoff(lambda: list(subreddit.hot(limit=limit // 3)))
                except Exception as e:
                    logger.warning(f"Failed to get hot posts from r/{subreddit_name}: {e}")
                    hot_posts = []
                
                for post in hot_posts:
                    if posts_scraped >= limit:
                        break
                    data = self._process_post(
                        post, cities, keywords, subreddit_name, 
                        scrape_timestamp, include_comments, max_comments_per_post
                    )
                    if data:
                        # Robust deduplication: use post_id + url
                        dedup_key = f"{data.get('post_id', '')}_{data.get('url', '')}"
                        if dedup_key not in seen_items:
                            seen_items.add(dedup_key)
                            scraped_data.append(data)
                            posts_scraped += 1
                
                # Get top posts
                try:
                    top_posts = self._retry_with_backoff(lambda: list(subreddit.top(time_filter=time_filter, limit=limit // 3)))
                except Exception as e:
                    logger.warning(f"Failed to get top posts from r/{subreddit_name}: {e}")
                    top_posts = []
                
                for post in top_posts:
                    if posts_scraped >= limit:
                        break
                    data = self._process_post(
                        post, cities, keywords, subreddit_name,
                        scrape_timestamp, include_comments, max_comments_per_post
                    )
                    if data:
                        dedup_key = f"{data.get('post_id', '')}_{data.get('url', '')}"
                        if dedup_key not in seen_items:
                            seen_items.add(dedup_key)
                            scraped_data.append(data)
                            posts_scraped += 1
                
                # Get new posts
                try:
                    new_posts = self._retry_with_backoff(lambda: list(subreddit.new(limit=limit // 3)))
                except Exception as e:
                    logger.warning(f"Failed to get new posts from r/{subreddit_name}: {e}")
                    new_posts = []
                
                for post in new_posts:
                    if posts_scraped >= limit:
                        break
                    data = self._process_post(
                        post, cities, keywords, subreddit_name,
                        scrape_timestamp, include_comments, max_comments_per_post
                    )
                    if data:
                        dedup_key = f"{data.get('post_id', '')}_{data.get('url', '')}"
                        if dedup_key not in seen_items:
                            seen_items.add(dedup_key)
                            scraped_data.append(data)
                            posts_scraped += 1
                
                logger.info(f"Scraped {posts_scraped} posts from r/{subreddit_name}")
                
            except PrawcoreException as e:
                logger.error(f"Error scraping r/{subreddit_name}: {e}")
                continue
            except Exception as e:
                logger.error(f"Unexpected error scraping r/{subreddit_name}: {e}")
                continue
        
        # Convert to DataFrame
        if not scraped_data:
            logger.warning("No data scraped from Reddit")
            df = pd.DataFrame(columns=["platform", "title", "text", "city", "url", "post_id", "subreddit", "scrape_timestamp"])
        else:
            df = pd.DataFrame(scraped_data)
            logger.info(f"Total posts scraped: {len(df)} (after deduplication)")
        
        # Save to parquet
        timestamp = scrape_timestamp.strftime("%Y%m%d_%H%M%S")
        out_path = RAW_DIR / f"reddit_praw_{timestamp}.parquet"
        save_parquet(df, out_path)
        logger.info(f"Reddit data saved to {out_path}")
        return out_path
    
    def _process_post(
        self, 
        post, 
        cities: List[str], 
        keywords: List[str],
        subreddit_name: str,
        scrape_timestamp: datetime,
        include_comments: bool = False,
        max_comments_per_post: int = 10
    ) -> Optional[Dict[str, Any]]:
        """
        Process a Reddit post with consistent metadata.
        
        Args:
            post: PRAW post object
            cities: List of cities to match
            keywords: List of keywords to match
            subreddit_name: Name of subreddit
            scrape_timestamp: Timestamp when scraping occurred
            include_comments: Whether to include comments
            max_comments_per_post: Maximum comments to include
        
        Returns:
            Dictionary with post data or None
        """
        try:
            # Skip if post is removed or deleted
            if not post.selftext and not post.title:
                return None
            
            # Combine title and text
            text = f"{post.title} {post.selftext}".strip()
            if not text:
                return None
            
            # Check if post matches keywords
            text_lower = text.lower()
            matched_keywords = [kw for kw in keywords if kw.lower() in text_lower]
            if not matched_keywords:
                return None
            
            # Extract city
            city = extract_city_from_text(text, cities)
            if city:
                city = normalize_city_name(city)
            
            # Get comments if requested
            comments_data = []
            if include_comments and post.num_comments > 0:
                try:
                    post.comments.replace_more(limit=0)  # Load all comments
                    for comment in post.comments.list()[:max_comments_per_post]:
                        if hasattr(comment, 'body') and comment.body:
                            comments_data.append({
                                "comment_id": comment.id,
                                "body": comment.body[:1000],  # Limit length
                                "score": getattr(comment, 'score', 0),
                                "author": str(comment.author) if hasattr(comment, 'author') and comment.author else None,
                                "created_utc": datetime.fromtimestamp(comment.created_utc, tz=timezone.utc) if hasattr(comment, 'created_utc') else None,
                            })
                except Exception as e:
                    logger.debug(f"Error fetching comments for post {post.id}: {e}")
            
            # Create data item with consistent metadata
            data = {
                "platform": "reddit",
                "post_id": post.id,
                "title": post.title or "",
                "text": text,
                "city": city,
                "url": f"https://reddit.com{post.permalink}",
                "author": str(post.author) if post.author else None,
                "subreddit": subreddit_name,  # Use provided name for consistency
                "score": post.score,
                "created_utc": datetime.fromtimestamp(post.created_utc, tz=timezone.utc) if post.created_utc else None,
                "num_comments": post.num_comments,
                "matched_keywords": matched_keywords,  # Store matched keywords
                "scrape_timestamp": scrape_timestamp,  # When this was scraped
                "comments": comments_data if include_comments else None,
            }
            
            return data
            
        except Exception as e:
            logger.debug(f"Error processing post {post.id}: {e}")
            return None


def scrape_reddit_praw(force: bool = False) -> Path:
    """Scrape Reddit using PRAW method."""
    scraper = RedditScraperPRAW()
    return scraper.scrape()


if __name__ == "__main__":
    try:
        path = scrape_reddit_praw()
        logger.info(f"Reddit data saved to {path}")
    except ValueError as e:
        logger.error(f"Error: {e}")
        logger.error("Please set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET in .env file")

