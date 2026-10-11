"""
Unified configuration management module
Combines best practices from both projects
"""
import os
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field, ConfigDict
from dotenv import load_dotenv
from dataclasses import dataclass

# Load environment variables (ignore if .env file doesn't exist or can't be read)
try:
    env_path = Path(".env")
    if env_path.exists():
        load_dotenv(override=False)
except Exception:
    # If .env file can't be loaded, continue without it
    pass

# Feature flags - read from environment
FAST_MODE = os.getenv("FAST_MODE", "1") not in {"0", "false", "False"}
ENABLE_BERTOPIC = os.getenv("ENABLE_BERTOPIC", "1") not in {"0", "false", "False"}
ENABLE_CAUSAL = os.getenv("ENABLE_CAUSAL", "1") not in {"0", "false", "False"}


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Reddit API (optional - only needed if using PRAW)
    reddit_client_id: str = Field(default="", env="REDDIT_CLIENT_ID")
    reddit_client_secret: str = Field(default="", env="REDDIT_CLIENT_SECRET")
    reddit_user_agent: str = Field(default="CityNewsScraper/1.0", env="REDDIT_USER_AGENT")
    
    # News API
    news_api_key: str = Field(default="", env="NEWS_API_KEY")
    
    # Database
    database_url: str = Field(default="sqlite:///data/city_news.db", env="DATABASE_URL")
    vector_db_path: str = Field(default="data/vector_db", env="VECTOR_DB_PATH")
    
    # Scraping
    scrape_delay: int = Field(default=2, env="SCRAPE_DELAY")
    max_posts_per_subreddit: int = Field(default=1000, env="MAX_POSTS_PER_SUBREDDIT")
    max_articles_per_source: int = Field(default=500, env="MAX_ARTICLES_PER_SOURCE")
    
    # NLP Models
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        env="EMBEDDING_MODEL"
    )
    ner_model: str = Field(default="en_core_web_sm", env="NER_MODEL")
    sentiment_model: str = Field(
        default="distilbert-base-uncased-finetuned-sst-2-english",
        env="SENTIMENT_MODEL"
    )
    
    # Cities
    target_cities: str = Field(
        default="bangalore,chennai,mumbai,delhi",
        env="TARGET_CITIES"
    )
    
    # API
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    
    # Feature flags
    fast_mode: bool = Field(default=True, env="FAST_MODE")
    enable_bertopic: bool = Field(default=True, env="ENABLE_BERTOPIC")
    enable_causal: bool = Field(default=True, env="ENABLE_CAUSAL")
    
    # Pydantic v2 configuration
    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore"  # Ignore extra environment variables not defined in Settings
    )


@dataclass
class ScrapeConfig:
    """Scraping configuration"""
    reddit_query: str = os.getenv("REDDIT_QUERY", "city public services")
    reddit_limit: int = int(os.getenv("REDDIT_LIMIT", "200"))
    news_sources: List[str] = tuple(
        os.getenv(
            "NEWS_SOURCES",
            "https://www.thehindu.com,https://indianexpress.com,https://www.deccanherald.com",
        ).split(",")
    )
    news_limit_per_source: int = int(os.getenv("NEWS_LIMIT", "100"))


class ConfigManager:
    """Manages configuration from YAML file and environment variables"""
    
    def __init__(self, config_path: str = None):
        if config_path is None:
            # Try to find config relative to project root
            base_dir = Path(__file__).resolve().parent.parent
            config_path = base_dir / "config" / "config.yaml"
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        self.settings = Settings()
        self.scrape_config = ScrapeConfig()
        self.load_config()
        self._setup_directories()
    
    def load_config(self) -> None:
        """Load configuration from YAML file"""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f) or {}
        else:
            # Create default config if not exists
            self.config = self._get_default_config()
            self._save_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "scraping": {
                "reddit": {
                    "method": "praw",
                    "subreddits": ["india", "bangalore", "chennai", "mumbai"],
                    "keywords": ["waste management", "flood", "traffic"],
                    "max_posts": 1000
                }
            },
            "nlp": {
                "models": {
                    "embedding": "sentence-transformers/all-MiniLM-L6-v2",
                    "ner": "en_core_web_sm"
                }
            },
            "storage": {
                "vector_store": "faiss",
                "sqlite_path": "data/city_news.db"
            }
        }
    
    def _save_config(self) -> None:
        """Save configuration to file"""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
    
    def _setup_directories(self) -> None:
        """Create necessary directories"""
        directories = [
            'data/raw',
            'data/processed',
            'data/vector_db',
            'index',
            'models',
            'logs',
            'notebooks'
        ]
        
        # Use BASE_DIR to ensure paths are relative to project root, not CWD
        base_dir = Path(__file__).resolve().parent.parent
        for directory in directories:
            (base_dir / directory).mkdir(parents=True, exist_ok=True)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key (supports nested keys with dot notation)"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        
        return value if value is not None else default
    
    def get_cities(self) -> List[str]:
        """Get list of target cities"""
        return self.config.get('cities', [])
    
    def get_subreddits(self) -> List[str]:
        """Get list of subreddits to scrape"""
        return self.config.get('scraping', {}).get('reddit', {}).get('subreddits', [])
    
    def get_keywords(self) -> List[str]:
        """Get list of keywords to search for"""
        return self.config.get('scraping', {}).get('reddit', {}).get('keywords', [])
    
    def get_reddit_method(self) -> str:
        """Get Reddit scraping method"""
        return self.config.get('scraping', {}).get('reddit', {}).get('method', 'praw')
    
    def get_vector_store_type(self) -> str:
        """Get vector store type"""
        return self.config.get('storage', {}).get('vector_store', 'faiss')
    
    def get_topic_modeling_method(self) -> str:
        """Get topic modeling method"""
        return self.config.get('nlp', {}).get('topic_modeling', {}).get('method', 'lda')


# Global config instance
config = ConfigManager()
settings = Settings()
scrape_config = ScrapeConfig()

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
INDEX_DIR = BASE_DIR / "index"
LOGS_DIR = BASE_DIR / "logs"

# NLP models
SPACY_MODEL_SMALL = os.getenv("SPACY_MODEL_SMALL", "en_core_web_sm")
SPACY_MODEL_TRF = os.getenv("SPACY_MODEL_TRF", "en_core_web_trf")
SBERT_MODEL_NAME = os.getenv("SBERT_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

# Topic modeling
TOPIC_PARAMS: Dict[str, int] = {
    "num_topics": int(os.getenv("LDA_TOPICS", "10")),
    "passes": int(os.getenv("LDA_PASSES", "2")),
    "iterations": int(os.getenv("LDA_ITERATIONS", "50")),
}

__all__ = [
    "config",
    "settings",
    "scrape_config",
    "FAST_MODE",
    "ENABLE_BERTOPIC",
    "ENABLE_CAUSAL",
    "BASE_DIR",
    "DATA_DIR",
    "RAW_DIR",
    "PROC_DIR",
    "MODELS_DIR",
    "INDEX_DIR",
    "LOGS_DIR",
    "SPACY_MODEL_SMALL",
    "SPACY_MODEL_TRF",
    "SBERT_MODEL_NAME",
    "TOPIC_PARAMS",
]

