"""
Utility functions for the project
"""
import re
from pathlib import Path
from typing import List, Optional
from loguru import logger
import sys


def setup_logging(log_file: Optional[str] = None, level: str = "INFO"):
    """
    Setup logging configuration
    
    Args:
        log_file: Path to log file (optional)
        level: Logging level
    """
    logger.remove()  # Remove default handler
    
    # Add console handler
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level=level
    )
    
    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function} - {message}",
            level=level,
            rotation="10 MB",
            retention="7 days"
        )


def extract_city_from_text(text: str, cities: List[str]) -> Optional[str]:
    """
    Extract city name from text
    
    Args:
        text: Input text
        cities: List of city names to search for
    
    Returns:
        City name if found, None otherwise
    """
    if not text:
        return None
    
    text_lower = text.lower()
    
    # Check for exact matches first
    for city in cities:
        city_lower = city.lower()
        if city_lower in text_lower:
            return city
    
    # Check for common city name variations
    city_variations = {
        'bangalore': ['bengaluru', 'blr'],
        'mumbai': ['bombay'],
        'chennai': ['madras'],
        'kolkata': ['calcutta'],
        'delhi': ['new delhi', 'ncr'],
    }
    
    for city, variations in city_variations.items():
        if city in text_lower:
            return city
        for variation in variations:
            if variation in text_lower:
                return city
    
    return None


def clean_text(text: str) -> str:
    """
    Clean text by removing URLs and special characters.
    Consolidated implementation - same as preprocess.clean_text for consistency.
    """
    import re
    
    if not text:
        return ""
    
    # Remove URLs
    clean_regex = re.compile(r"https?://\S+|www\.\S+")
    punct_regex = re.compile(r"[^A-Za-z0-9\s]")
    
    text = clean_regex.sub(" ", text)
    text = punct_regex.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def normalize_city_name(city: str) -> Optional[str]:
    """
    Normalize city name to standard form using city registry.
    
    Args:
        city: City name
    
    Returns:
        Normalized city name (standard form) or None
    """
    from .city_registry import get_city_registry
    registry = get_city_registry()
    return registry.normalize(city)


def create_directories(directories: List[str]):
    """
    Create directories if they don't exist
    
    Args:
        directories: List of directory paths
    """
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

