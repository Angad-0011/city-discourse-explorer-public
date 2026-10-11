"""Improved news scraping with RSS feed parsing and trafilatura (replaces abandoned newspaper3k)."""
from __future__ import annotations

import asyncio
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional, Set
from urllib.parse import urljoin, urlparse

import aiohttp
import pandas as pd
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from loguru import logger

try:
    from dateutil import parser as date_parser
except ImportError:
    date_parser = None
    logger.warning("python-dateutil not installed. Date parsing will be limited. Install via `pip install python-dateutil`")

# Suppress XML parsing warnings
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

try:
    import feedparser
except ImportError:
    feedparser = None

try:
    import trafilatura
except ImportError:
    trafilatura = None
    logger.warning("trafilatura not installed. Install via `pip install trafilatura`. Text extraction will be limited.")

from ..config import config, FAST_MODE, RAW_DIR
from ..utils_io import save_parquet, save_json
from ..processors.preprocess import guess_city

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# Get cities from config
CITIES = config.get_cities() or ["Mumbai", "Delhi", "Bengaluru", "Chennai", "Hyderabad", "Pune", "Kolkata"]

# Domain allowlist for polite crawling
ALLOWED_DOMAINS = set()
FETCH_LOG: List[Dict[str, any]] = []  # URL fetch log


def _fetch_rss_feed(feed_url: str) -> List[Dict[str, str]]:
    """Parse RSS feed and extract article links."""
    if feedparser is None:
        logger.warning("feedparser not installed. Install via `pip install feedparser`. Skipping RSS feeds.")
        return []
    
    try:
        feed = feedparser.parse(feed_url)
        articles = []
        
        for entry in feed.entries[:50]:  # Limit to 50 per feed
            article_url = entry.get("link", "")
            if article_url:
                articles.append({
                    "url": article_url,
                    "title": entry.get("title", ""),
                    "published": entry.get("published", ""),
                    "summary": entry.get("summary", "")
                })
        
        logger.info(f"Parsed {len(articles)} articles from RSS feed: {feed_url}")
        return articles
    except Exception as e:
        logger.error(f"Error parsing RSS feed {feed_url}: {e}")
        return []


def _normalize_date(date_str: Optional[str], fallback_date: Optional[datetime] = None) -> Optional[datetime]:
    """
    Normalize published date to consistent datetime format.
    
    Args:
        date_str: Date string from various sources
        fallback_date: Fallback date if parsing fails
    
    Returns:
        Normalized datetime in UTC or None
    """
    if not date_str:
        return fallback_date
    
    try:
        # Try parsing with dateutil if available
        if date_parser:
            parsed = date_parser.parse(date_str)
            # Ensure timezone-aware (default to UTC if naive)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        else:
            # Fallback to basic datetime parsing
            from datetime import datetime as dt
            # Try common formats
            for fmt in ["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"]:
                try:
                    parsed = dt.strptime(date_str[:19], fmt)
                    return parsed.replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
            return fallback_date
    except Exception:
        return fallback_date


async def _extract_with_trafilatura(session: aiohttp.ClientSession, url: str) -> Optional[Dict[str, str]]:
    """
    Extract article using trafilatura library (better than abandoned newspaper3k).
    
    Args:
        session: aiohttp session for async HTTP requests
        url: Article URL
    
    Returns:
        Dictionary with article data and extraction metadata
    """
    if trafilatura is None:
        return None
    
    extraction_method = "trafilatura"
    status_code = None
    
    try:
        # Fetch HTML asynchronously
        async with session.get(url, headers=HEADERS, timeout=aiohttp.ClientTimeout(total=15)) as response:
            status_code = response.status
            if status_code != 200:
                FETCH_LOG.append({
                    "url": url,
                    "status_code": status_code,
                    "extraction_method": extraction_method,
                    "success": False,
                    "error": f"HTTP {status_code}",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                return None
            
            html = await response.text()
        
        # Extract with trafilatura
        extracted = trafilatura.extract(html, include_comments=False, include_tables=False)
        
        if not extracted or len(extracted) < 100:
            FETCH_LOG.append({
                "url": url,
                "status_code": status_code,
                "extraction_method": extraction_method,
                "success": False,
                "error": "Insufficient text content",
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            return None
        
        # Get metadata
        metadata = trafilatura.extract_metadata(html)
        title = metadata.title if metadata else None
        
        # Detect city from article text
        combined_text = f"{title or ''} {extracted}"
        detected_city = guess_city(combined_text)
        
        # Try to extract published date from metadata
        published_date = None
        if metadata and metadata.date:
            published_date = _normalize_date(metadata.date)
        
        result = {
            "platform": "news",
            "city": detected_city,
            "title": title or "",
            "text": extracted[:10000],  # Limit text length
            "url": url,
            "published": published_date.isoformat() if published_date else None,
            "published_utc": published_date,  # Store as datetime object
            "extraction_method": extraction_method,
        }
        
        FETCH_LOG.append({
            "url": url,
            "status_code": status_code,
            "extraction_method": extraction_method,
            "success": True,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        return result
    except asyncio.TimeoutError:
        FETCH_LOG.append({
            "url": url,
            "status_code": status_code,
            "extraction_method": extraction_method,
            "success": False,
            "error": "Timeout",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        logger.debug(f"Timeout extracting article from {url}")
        return None
    except Exception as e:
        FETCH_LOG.append({
            "url": url,
            "status_code": status_code,
            "extraction_method": extraction_method,
            "success": False,
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        logger.debug(f"Error extracting article from {url}: {e}")
        return None


async def _extract_with_bs4(session: aiohttp.ClientSession, url: str) -> Optional[Dict[str, str]]:
    """
    Fallback: Extract article using BeautifulSoup.
    
    Args:
        session: aiohttp session for async HTTP requests
        url: Article URL
    
    Returns:
        Dictionary with article data and extraction metadata
    """
    extraction_method = "beautifulsoup"
    status_code = None
    
    try:
        async with session.get(url, headers=HEADERS, timeout=aiohttp.ClientTimeout(total=15)) as response:
            status_code = response.status
            if status_code != 200:
                FETCH_LOG.append({
                    "url": url,
                    "status_code": status_code,
                    "extraction_method": extraction_method,
                    "success": False,
                    "error": f"HTTP {status_code}",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                return None
            
            html = await response.text()
        
        soup = BeautifulSoup(html, "html.parser")
        
        # Try to find title
        title = None
        for tag in ["title", "h1", "h2"]:
            title_elem = soup.find(tag)
            if title_elem:
                title = title_elem.get_text(strip=True)
                break
        
        # Extract text from paragraphs
        paragraphs = []
        for p in soup.find_all("p"):
            text = p.get_text(strip=True)
            if text and len(text) > 20:  # Filter out short paragraphs
                paragraphs.append(text)
        
        text = " ".join(paragraphs)
        if not text or len(text) < 100:
            FETCH_LOG.append({
                "url": url,
                "status_code": status_code,
                "extraction_method": extraction_method,
                "success": False,
                "error": "Insufficient text content",
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            return None
        
        # Detect city
        combined_text = f"{title} {text}"
        detected_city = guess_city(combined_text)
        
        # Try to extract published date from meta tags
        published_date = None
        for meta_tag in soup.find_all("meta"):
            prop = meta_tag.get("property") or meta_tag.get("name", "").lower()
            if "date" in prop or "published" in prop or "time" in prop:
                content = meta_tag.get("content")
                if content:
                    published_date = _normalize_date(content)
                    break
        
        result = {
            "platform": "news",
            "city": detected_city,
            "title": title or url,
            "text": text[:10000],
            "url": url,
            "published": published_date.isoformat() if published_date else None,
            "published_utc": published_date,
            "extraction_method": extraction_method,
        }
        
        FETCH_LOG.append({
            "url": url,
            "status_code": status_code,
            "extraction_method": extraction_method,
            "success": True,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        return result
    except asyncio.TimeoutError:
        FETCH_LOG.append({
            "url": url,
            "status_code": status_code,
            "extraction_method": extraction_method,
            "success": False,
            "error": "Timeout",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        logger.debug(f"Timeout extracting with BeautifulSoup from {url}")
        return None
    except Exception as e:
        FETCH_LOG.append({
            "url": url,
            "status_code": status_code,
            "extraction_method": extraction_method,
            "success": False,
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        logger.debug(f"Error extracting with BeautifulSoup from {url}: {e}")
        return None


def _is_allowed_domain(url: str, allowed_domains: Set[str]) -> bool:
    """Check if URL domain is in allowlist."""
    if not allowed_domains:
        return True  # No restrictions
    
    try:
        domain = urlparse(url).netloc.lower()
        # Remove www. prefix for comparison
        domain = domain.replace("www.", "")
        return domain in allowed_domains or any(allowed in domain for allowed in allowed_domains)
    except Exception:
        return False


def _is_likely_article_link(href: str, base_url: str) -> bool:
    """
    Improved heuristic to determine if a link is likely an article.
    Reduces false positives from city pages.
    """
    href_lower = href.lower()
    
    # Exclude common non-article patterns
    exclude_patterns = [
        "/tag/", "/category/", "/author/", "/page/", "/archive/",
        "/search", "/login", "/register", "/contact", "/about",
        "/privacy", "/terms", "/cookie", "/sitemap", "/feed",
        ".jpg", ".png", ".gif", ".pdf", ".zip", ".css", ".js",
        "#", "javascript:", "mailto:", "tel:"
    ]
    
    if any(pattern in href_lower for pattern in exclude_patterns):
        return False
    
    # Must contain article-like keywords AND be a reasonable length
    article_keywords = ["article", "news", "story", "report", "post", "blog"]
    has_keyword = any(keyword in href_lower for keyword in article_keywords)
    
    # URL should be reasonable length (not too short, not too long)
    url_length = len(href)
    reasonable_length = 20 <= url_length <= 200
    
    # Should have a path (not just domain)
    has_path = "/" in href_lower.replace("://", "")
    
    return (has_keyword or reasonable_length) and has_path


async def _fetch_article(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    url: str,
    allowed_domains: Set[str] = None
) -> Optional[Dict[str, str]]:
    """
    Fetch and extract article content asynchronously with rate limiting.
    
    Args:
        session: aiohttp session
        semaphore: Semaphore for rate limiting
        url: Article URL
        allowed_domains: Set of allowed domains (None = no restriction)
    
    Returns:
        Dictionary with article data or None
    """
    if allowed_domains and not _is_allowed_domain(url, allowed_domains):
        FETCH_LOG.append({
            "url": url,
            "status_code": None,
            "extraction_method": None,
            "success": False,
            "error": "Domain not in allowlist",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        return None
    
    async with semaphore:  # Rate limiting
        # Try trafilatura first (better extraction)
        article = await _extract_with_trafilatura(session, url)
        if article:
            return article
        
        # Fallback to BeautifulSoup
        return await _extract_with_bs4(session, url)


async def _scrape_articles_async(
    urls: List[str],
    allowed_domains: Set[str],
    max_concurrent: int = 5
) -> List[Dict[str, str]]:
    """
    Scrape multiple articles asynchronously with rate limiting.
    
    Args:
        urls: List of article URLs
        allowed_domains: Set of allowed domains
        max_concurrent: Maximum concurrent requests (default: 5)
    
    Returns:
        List of extracted articles
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    articles = []
    
    async with aiohttp.ClientSession() as session:
        tasks = [_fetch_article(session, semaphore, url, allowed_domains) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, Exception):
                logger.debug(f"Error in async fetch: {result}")
            elif result:
                articles.append(result)
    
    return articles


def scrape_news(force: bool = False) -> Path:
    """
    Scrape news articles from configured RSS feeds and sources.
    
    Features:
    - Normalized published dates
    - URL fetch logging (status codes, extraction methods)
    - Domain allowlist support
    - Improved city page link heuristics
    - Async scraping with rate limiting (much faster than sequential)
    - Uses trafilatura instead of abandoned newspaper3k
    """
    global FETCH_LOG, ALLOWED_DOMAINS
    FETCH_LOG = []  # Reset fetch log for this scrape
    
    if FAST_MODE and not force:
        sample_path = Path(__file__).resolve().parent.parent.parent / "data" / "sample" / "sample_news.parquet"
        if sample_path.exists():
            logger.info(f"FAST_MODE=1 -> using sample data at {sample_path}")
            return sample_path
    
    news_config = config.get("scraping", {}).get("news", {})
    max_articles = news_config.get("max_articles", 1000)
    delay_seconds = news_config.get("delay_seconds", 2)
    max_concurrent = news_config.get("max_concurrent", 5)  # Concurrent requests for async scraping
    
    # Get domain allowlist from config
    allowed_domains_list = news_config.get("allowed_domains", [])
    ALLOWED_DOMAINS = {domain.lower().replace("www.", "") for domain in allowed_domains_list}
    if ALLOWED_DOMAINS:
        logger.info(f"Using domain allowlist: {ALLOWED_DOMAINS}")
    
    all_articles = []
    sources = news_config.get("sources", [])
    seen_urls: Set[str] = set()  # For deduplication
    
    logger.info(f"Starting async news scraping from {len(sources)} sources (max {max_articles} articles, {max_concurrent} concurrent)")
    
    # Collect all URLs first
    all_urls = []
    url_metadata = {}  # Store RSS metadata (title, published date) for each URL
    
    for source in sources:
        source_name = source.get("name", "Unknown")
        logger.info(f"Collecting URLs from {source_name}...")
        
        # Process RSS feeds
        rss_feeds = source.get("rss_feeds", [])
        for feed_url in rss_feeds:
            logger.debug(f"  Parsing RSS feed: {feed_url}")
            feed_articles = _fetch_rss_feed(feed_url)
            
            for feed_article in feed_articles:
                if len(all_urls) >= max_articles:
                    break
                
                article_url = feed_article["url"]
                if article_url not in seen_urls:
                    all_urls.append(article_url)
                    seen_urls.add(article_url)
                    url_metadata[article_url] = feed_article
        
        # Process city pages (direct HTML scraping)
        city_pages = source.get("city_pages", {})
        for city_name, page_url in city_pages.items():
            if len(all_urls) >= max_articles:
                break
            
            logger.debug(f"  Scraping city page: {city_name} - {page_url}")
            try:
                import requests
                response = requests.get(page_url, headers=HEADERS, timeout=15)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, "html.parser")
                    # Find article links (common patterns)
                    links = []
                    for a in soup.find_all("a", href=True):
                        href = a.get("href", "")
                        if href:
                            # Make absolute URL
                            if href.startswith("/"):
                                href = urljoin(page_url, href)
                            elif not href.startswith("http"):
                                continue
                            
                            # Use improved heuristic to check if it looks like an article link
                            if _is_likely_article_link(href, page_url):
                                links.append(href)
                    
                    # Add links to URL list
                    for link in links[:20]:  # Limit to 20 per city page
                        if len(all_urls) >= max_articles:
                            break
                        
                        if link not in seen_urls:
                            all_urls.append(link)
                            seen_urls.add(link)
                            url_metadata[link] = {"city": city_name.title()}
            except Exception as e:
                logger.error(f"  Error scraping city page {page_url}: {e}")
        
        if len(all_urls) >= max_articles:
            break
    
    # Limit to max_articles
    all_urls = all_urls[:max_articles]
    logger.info(f"Collected {len(all_urls)} URLs. Starting async extraction...")
    
    # Scrape all URLs asynchronously
    articles = asyncio.run(_scrape_articles_async(all_urls, ALLOWED_DOMAINS, max_concurrent))
    
    # Merge RSS metadata into articles
    for article in articles:
        url = article.get("url")
        if url in url_metadata:
            metadata = url_metadata[url]
            # Use title from RSS if available
            if metadata.get("title") and not article.get("title"):
                article["title"] = metadata["title"]
            
            # Normalize RSS published date if available
            if metadata.get("published") and not article.get("published_utc"):
                article["published_utc"] = _normalize_date(metadata["published"])
                article["published"] = article["published_utc"].isoformat() if article["published_utc"] else None
            
            # Set city from metadata if not detected
            if metadata.get("city") and not article.get("city"):
                article["city"] = metadata["city"]
    
    # Create dataframe
    if not articles:
        logger.warning("No articles extracted. Creating empty dataframe.")
        df = pd.DataFrame(columns=["platform", "city", "title", "text", "url"])
    else:
        df = pd.DataFrame(articles)
        logger.info(f"✅ Successfully extracted {len(df)} articles")
        
        # Show statistics
        if "city" in df.columns:
            cities = df["city"].dropna().value_counts()
            logger.info(f"   Cities found: {len(cities)} unique cities")
            logger.info(f"   Top cities: {dict(cities.head(5))}")
    
    # Save articles
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = RAW_DIR / f"news_{timestamp}.parquet"
    save_parquet(df, out_path)
    logger.info(f"News data saved to {out_path}")
    
    # Save fetch log
    if FETCH_LOG:
        log_path = RAW_DIR / f"news_fetch_log_{timestamp}.json"
        save_json(FETCH_LOG, log_path)
        logger.info(f"Fetch log saved to {log_path} ({len(FETCH_LOG)} entries)")
        
        # Log statistics
        successful = sum(1 for entry in FETCH_LOG if entry.get("success"))
        failed = len(FETCH_LOG) - successful
        logger.info(f"Fetch statistics: {successful} successful, {failed} failed")
    
    return out_path


if __name__ == "__main__":
    path = scrape_news()
    logger.info(f"News scraping complete: {path}")
