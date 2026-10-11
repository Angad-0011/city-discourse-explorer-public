"""
Tests for city detection and normalization.
Tests edge cases like "NCR", "Bombay", typos, etc.
"""
import pytest
from src.city_registry import get_city_registry
from src.utils import normalize_city_name
from src.processors.preprocess import guess_city


class TestCityNormalization:
    """Test city name normalization."""
    
    def test_standard_names(self):
        """Test that standard names are preserved."""
        assert normalize_city_name("Mumbai") == "Mumbai"
        assert normalize_city_name("Delhi") == "Delhi"
        assert normalize_city_name("Bengaluru") == "Bengaluru"
        assert normalize_city_name("Chennai") == "Chennai"
    
    def test_variations(self):
        """Test common city name variations."""
        assert normalize_city_name("Bombay") == "Mumbai"
        assert normalize_city_name("Bangalore") == "Bengaluru"
        assert normalize_city_name("Madras") == "Chennai"
        assert normalize_city_name("Calcutta") == "Kolkata"
        assert normalize_city_name("Baroda") == "Vadodara"
        assert normalize_city_name("Vizag") == "Visakhapatnam"
    
    def test_ncr_variations(self):
        """Test NCR/Delhi variations."""
        assert normalize_city_name("NCR") == "Delhi"
        assert normalize_city_name("NCT") == "Delhi"
        assert normalize_city_name("New Delhi") == "Delhi"
        assert normalize_city_name("Delhi NCR") == "Delhi"
        assert normalize_city_name("National Capital Region") == "Delhi"
    
    def test_abbreviations(self):
        """Test city abbreviations."""
        assert normalize_city_name("BLR") == "Bengaluru"
        assert normalize_city_name("LKO") == "Lucknow"
        assert normalize_city_name("HYD") == "Hyderabad"
        assert normalize_city_name("AMD") == "Ahmedabad"
    
    def test_case_insensitive(self):
        """Test that normalization is case-insensitive."""
        assert normalize_city_name("MUMBAI") == "Mumbai"
        assert normalize_city_name("mumbai") == "Mumbai"
        assert normalize_city_name("MuMbAi") == "Mumbai"
        assert normalize_city_name("BOMBAY") == "Mumbai"
    
    def test_typos(self):
        """Test handling of common typos."""
        # These should still work if they're in the registry
        assert normalize_city_name("Bombai") == "Mumbai"  # Common typo
        assert normalize_city_name("Kolkatta") == "Kolkata"  # Common typo
        assert normalize_city_name("Hyderbad") == "Hyderabad"  # Common typo
        assert normalize_city_name("Ahmedbad") == "Ahmedabad"  # Common typo
    
    def test_exclude_terms(self):
        """Test that generic terms are excluded."""
        assert normalize_city_name("India") is None
        assert normalize_city_name("Indian") is None
        assert normalize_city_name("Country") is None
        assert normalize_city_name("State") is None
        assert normalize_city_name("Nation") is None
        assert normalize_city_name("World") is None
        assert normalize_city_name("Asia") is None
    
    def test_unknown_cities(self):
        """Test that unknown cities return None."""
        assert normalize_city_name("New York") is None
        assert normalize_city_name("London") is None
        assert normalize_city_name("Tokyo") is None
        assert normalize_city_name("") is None
        assert normalize_city_name(None) is None


class TestCityDetection:
    """Test city detection from text."""
    
    def test_direct_mentions(self):
        """Test direct city mentions in text."""
        assert guess_city("I live in Mumbai") == "Mumbai"
        assert guess_city("Delhi is the capital") == "Delhi"
        assert guess_city("Bengaluru weather is nice") == "Bengaluru"
    
    def test_variations_in_text(self):
        """Test city variations in text."""
        assert guess_city("Bombay is a great city") == "Mumbai"
        assert guess_city("I visited Bangalore last week") == "Bengaluru"
        assert guess_city("Madras is hot") == "Chennai"
        assert guess_city("Calcutta has good food") == "Kolkata"
    
    def test_ncr_in_text(self):
        """Test NCR mentions in text."""
        assert guess_city("NCR region is growing") == "Delhi"
        assert guess_city("I work in NCR") == "Delhi"
        assert guess_city("New Delhi is crowded") == "Delhi"
    
    def test_abbreviations_in_text(self):
        """Test abbreviations in text."""
        assert guess_city("BLR airport is busy") == "Bengaluru"
        assert guess_city("LKO has good infrastructure") == "Lucknow"
        assert guess_city("HYD is expanding") == "Hyderabad"
    
    def test_multiple_cities(self):
        """Test text with multiple cities (should return first match)."""
        result = guess_city("Mumbai and Delhi are both major cities")
        assert result in ["Mumbai", "Delhi"]  # Either is acceptable
    
    def test_no_city(self):
        """Test text with no city mentions."""
        assert guess_city("This is a generic text") is None
        assert guess_city("") is None
        assert guess_city("India is a country") is None  # Exclude term
    
    def test_existing_city_normalization(self):
        """Test normalization of existing city field."""
        assert guess_city("", existing="Bombay") == "Mumbai"
        assert guess_city("", existing="NCR") == "Delhi"
        assert guess_city("", existing="Bangalore") == "Bengaluru"
        assert guess_city("", existing="Mumbai") == "Mumbai"  # Already standard


class TestCityRegistry:
    """Test city registry functionality."""
    
    def test_registry_loading(self):
        """Test that registry loads correctly."""
        registry = get_city_registry()
        assert len(registry.get_all_standard_names()) > 0
        assert "Mumbai" in registry.get_all_standard_names()
        assert "Delhi" in registry.get_all_standard_names()
    
    def test_variation_mapping(self):
        """Test variation to standard mapping."""
        registry = get_city_registry()
        assert registry.normalize("Bombay") == "Mumbai"
        assert registry.normalize("NCR") == "Delhi"
        assert registry.normalize("Bangalore") == "Bengaluru"
    
    def test_regex_pattern(self):
        """Test that regex pattern is generated correctly."""
        registry = get_city_registry()
        assert registry.regex_pattern is not None
        
        # Test that it matches city names
        assert registry.regex_pattern.search("I live in Mumbai")
        assert registry.regex_pattern.search("Bombay is great")
        assert registry.regex_pattern.search("NCR region")
    
    def test_exclude_terms(self):
        """Test exclude terms functionality."""
        registry = get_city_registry()
        assert registry.is_exclude_term("India")
        assert registry.is_exclude_term("indian")
        assert not registry.is_exclude_term("Mumbai")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

