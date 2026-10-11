"""
Canonical city registry - single source of truth for city names and variations.
Loads from config/cities.yaml and provides utilities for city detection and normalization.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import yaml
from loguru import logger

# Path to city registry YAML
CITIES_YAML_PATH = Path(__file__).parent.parent / "config" / "cities.yaml"


class CityRegistry:
    """Canonical city registry loaded from YAML configuration."""
    
    def __init__(self, cities_yaml_path: Path = CITIES_YAML_PATH):
        """
        Initialize city registry from YAML file.
        
        Args:
            cities_yaml_path: Path to cities.yaml file
        """
        self.cities_yaml_path = cities_yaml_path
        self.cities_data: List[Dict] = []
        self.standard_names: List[str] = []
        self.variation_to_standard: Dict[str, str] = {}
        self.standard_to_variations: Dict[str, List[str]] = {}
        self.exclude_terms: Set[str] = set()
        self.regex_pattern: Optional[re.Pattern] = None
        
        self._load_registry()
        self._build_mappings()
        self._build_regex()
    
    def _load_registry(self) -> None:
        """Load city registry from YAML file."""
        if not self.cities_yaml_path.exists():
            logger.warning(f"City registry not found at {self.cities_yaml_path}. Using defaults.")
            self._load_defaults()
            return
        
        try:
            with open(self.cities_yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            self.cities_data = data.get('cities', [])
            self.exclude_terms = set(term.lower() for term in data.get('exclude_terms', []))
            
            logger.info(f"Loaded {len(self.cities_data)} cities from registry")
        except Exception as e:
            logger.error(f"Error loading city registry: {e}. Using defaults.")
            self._load_defaults()
    
    def _load_defaults(self) -> None:
        """Load default city data if YAML file is missing."""
        self.cities_data = [
            {"standard_name": "Mumbai", "variations": ["mumbai", "bombay", "bombai"]},
            {"standard_name": "Delhi", "variations": ["delhi", "new delhi", "ncr", "nct"]},
            {"standard_name": "Bengaluru", "variations": ["bengaluru", "bangalore", "blr"]},
            {"standard_name": "Chennai", "variations": ["chennai", "madras"]},
            {"standard_name": "Hyderabad", "variations": ["hyderabad", "hyd"]},
        ]
        self.exclude_terms = {"india", "indian", "country", "state", "nation", "world", "asia"}
    
    def _build_mappings(self) -> None:
        """Build internal mappings from city data."""
        for city_info in self.cities_data:
            standard_name = city_info['standard_name']
            variations = [v.lower() for v in city_info.get('variations', [])]
            aliases = [a.lower() for a in city_info.get('aliases', [])]
            
            self.standard_names.append(standard_name)
            self.standard_to_variations[standard_name] = variations
            
            # Map all variations and aliases to standard name
            for variation in variations + aliases:
                self.variation_to_standard[variation.lower()] = standard_name
    
    def _build_regex(self) -> None:
        """Build regex pattern from all city variations."""
        all_patterns = []
        for city_info in self.cities_data:
            variations = city_info.get('variations', [])
            aliases = city_info.get('aliases', [])
            
            for variation in variations + aliases:
                # Escape special regex characters
                escaped = re.escape(variation.lower())
                all_patterns.append(escaped)
        
        if all_patterns:
            # Create word boundary pattern
            pattern = r"\b(" + "|".join(all_patterns) + r")\b"
            self.regex_pattern = re.compile(pattern, flags=re.IGNORECASE)
        else:
            self.regex_pattern = re.compile(r"(?!)")  # Never matches
    
    def normalize(self, city_name: str) -> Optional[str]:
        """
        Normalize city name to standard form.
        
        Args:
            city_name: City name to normalize
        
        Returns:
            Standard city name or None if not found
        """
        if not city_name:
            return None
        
        city_lower = city_name.lower().strip()
        
        # Check if it's an exclude term
        if city_lower in self.exclude_terms:
            return None
        
        # Direct lookup in variation mapping
        if city_lower in self.variation_to_standard:
            return self.variation_to_standard[city_lower]
        
        # Check if it's already a standard name (case-insensitive)
        for standard in self.standard_names:
            if city_lower == standard.lower():
                return standard
        
        # Return None for unknown cities (don't guess)
        return None
    
    def detect_from_text(self, text: str, use_ner: bool = True) -> Optional[str]:
        """
        Detect city from text using regex and optionally NER.
        
        Args:
            text: Text to search
            use_ner: Whether to use NER (requires spaCy)
        
        Returns:
            Standard city name or None
        """
        if not text:
            return None
        
        text_lower = text.lower()
        
        # First, try regex matching
        if self.regex_pattern:
            match = self.regex_pattern.search(text)
            if match:
                matched_text = match.group(1).lower()
                return self.normalize(matched_text)
        
        # Second, try NER if enabled
        if use_ner:
            try:
                import spacy
                from ..config import SPACY_MODEL_SMALL
                
                nlp = spacy.load(SPACY_MODEL_SMALL, disable=["parser", "textcat"])
                doc = nlp(text)
                
                for ent in doc.ents:
                    if ent.label_ in {"GPE", "LOC"}:
                        ent_text = ent.text.strip()
                        ent_lower = ent_text.lower()
                        
                        # Skip exclude terms
                        if ent_lower in self.exclude_terms:
                            continue
                        
                        # Try to normalize NER entity
                        normalized = self.normalize(ent_text)
                        if normalized:
                            return normalized
            except Exception:
                pass  # NER failed, continue
        
        return None
    
    def get_variations(self, standard_name: str) -> List[str]:
        """Get all variations for a standard city name."""
        return self.standard_to_variations.get(standard_name, [])
    
    def get_all_standard_names(self) -> List[str]:
        """Get list of all standard city names."""
        return self.standard_names.copy()
    
    def is_exclude_term(self, term: str) -> bool:
        """Check if term should be excluded from city detection."""
        return term.lower() in self.exclude_terms


# Global city registry instance
_city_registry: Optional[CityRegistry] = None


def get_city_registry() -> CityRegistry:
    """Get or create global city registry instance."""
    global _city_registry
    if _city_registry is None:
        _city_registry = CityRegistry()
    return _city_registry

