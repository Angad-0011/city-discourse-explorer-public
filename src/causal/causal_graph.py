"""Causal graph analysis for event relationships with improved event extraction and confidence explanations."""
from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
from loguru import logger

try:
    import networkx as nx
except ImportError:
    nx = None
    logger.warning("networkx not installed. Install via `pip install networkx`")

try:
    from dateutil import parser as date_parser
except ImportError:
    date_parser = None

from ..config import PROC_DIR, MODELS_DIR, SPACY_MODEL_SMALL
from ..utils_io import load_parquet, save_pickle, load_pickle

# Try to import spaCy for dependency parsing
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    spacy = None
    SPACY_AVAILABLE = False


class CausalGraphBuilder:
    """Build causal graphs from articles to identify event relationships."""
    
    def __init__(self):
        """Initialize causal graph builder."""
        if nx is None:
            raise ImportError("networkx is required for causal graph analysis")
        
        self.graph = nx.DiGraph()
        
        # Load spaCy model for dependency parsing (if available)
        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                # Load with parser enabled for dependency parsing
                self.nlp = spacy.load(SPACY_MODEL_SMALL)
                logger.info("Loaded spaCy model for dependency parsing")
            except Exception as e:
                logger.warning(f"Could not load spaCy model for dependency parsing: {e}. Falling back to regex-based extraction.")
                self.nlp = None
        
        # Causal verbs for dependency parsing (ROOT verbs that indicate causality)
        self.causal_verbs = {
            'cause', 'lead', 'result', 'trigger', 'prompt', 'bring', 'create',
            'generate', 'produce', 'induce', 'provoke', 'spark', 'initiate',
            'stimulate', 'motivate', 'drive', 'force', 'compel', 'make'
        }
        
        # Temporal keywords indicating causality (fallback for regex-based approach)
        self.causal_indicators = [
            'because', 'due to', 'as a result', 'led to', 'caused',
            'resulted in', 'following', 'after', 'consequently',
            'therefore', 'thus', 'hence', 'prompted', 'triggered',
            'stemmed from', 'originated from', 'brought about'
        ]
        
        # Temporal ordering keywords
        self.temporal_before = ['before', 'prior to', 'earlier', 'previously']
        self.temporal_after = ['after', 'following', 'subsequently', 'later', 'then']
        
        # Event extraction patterns (rule-based)
        self.event_patterns = [
            r'(\w+(?:\s+\w+){0,5})\s+(?:announced|declared|revealed|unveiled|launched|introduced)',
            r'(\w+(?:\s+\w+){0,5})\s+(?:happened|occurred|took place|transpired)',
            r'(\w+(?:\s+\w+){0,5})\s+(?:approved|passed|rejected|denied|accepted)',
            r'(\w+(?:\s+\w+){0,5})\s+(?:started|began|commenced|initiated)',
            r'(\w+(?:\s+\w+){0,5})\s+(?:completed|finished|ended|concluded)',
            r'(\w+(?:\s+\w+){0,5})\s+(?:increased|decreased|rose|fell|dropped)',
            r'(\w+(?:\s+\w+){0,5})\s+(?:opened|closed|shut down|inaugurated)',
            r'(\w+(?:\s+\w+){0,5})\s+(?:proposed|suggested|recommended|urged)',
        ]
    
    def _normalize_event_date(self, date_value: Any) -> Optional[datetime]:
        """
        Normalize event date from various formats to UTC datetime.
        
        Handles:
        - Reddit: created_utc (timestamp or datetime)
        - News: published (string or datetime)
        - published_utc (datetime)
        
        Args:
            date_value: Date in various formats
        
        Returns:
            Normalized datetime in UTC or None
        """
        if date_value is None or pd.isna(date_value):
            return None
        
        try:
            # If already a datetime object
            if isinstance(date_value, datetime):
                # Ensure timezone-aware (default to UTC if naive)
                if date_value.tzinfo is None:
                    date_value = date_value.replace(tzinfo=timezone.utc)
                return date_value.astimezone(timezone.utc)
            
            # If timestamp (float/int)
            if isinstance(date_value, (int, float)):
                return datetime.fromtimestamp(date_value, tz=timezone.utc)
            
            # If string, try parsing
            if isinstance(date_value, str):
                # Try dateutil parser first (handles many formats)
                if date_parser:
                    parsed = date_parser.parse(date_value)
                    if parsed.tzinfo is None:
                        parsed = parsed.replace(tzinfo=timezone.utc)
                    return parsed.astimezone(timezone.utc)
                else:
                    # Fallback to pandas
                    parsed = pd.to_datetime(date_value, errors='coerce')
                    if pd.notna(parsed):
                        if isinstance(parsed, pd.Timestamp):
                            parsed = parsed.to_pydatetime()
                        if parsed.tzinfo is None:
                            parsed = parsed.replace(tzinfo=timezone.utc)
                        return parsed.astimezone(timezone.utc)
            
            # Try pandas conversion as last resort
            parsed = pd.to_datetime(date_value, errors='coerce')
            if pd.notna(parsed):
                if isinstance(parsed, pd.Timestamp):
                    parsed = parsed.to_pydatetime()
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                return parsed.astimezone(timezone.utc)
            
        except Exception as e:
            logger.debug(f"Failed to normalize date {date_value}: {e}")
            return None
        
        return None
    
    def _extract_events_from_text(self, text: str, title: str = "") -> List[Dict[str, str]]:
        """
        Extract actual events from text using rule-based patterns.
        
        Args:
            text: Article text
            title: Article title (often contains event descriptions)
        
        Returns:
            List of extracted events with text and type
        """
        events = []
        combined_text = f"{title} {text}".lower()
        
        for pattern in self.event_patterns:
            matches = re.finditer(pattern, combined_text, re.IGNORECASE)
            for match in matches:
                event_text = match.group(1).strip()
                # Filter out very short or generic events
                if len(event_text) > 5 and event_text not in ['the', 'this', 'that', 'it']:
                    # Determine event type from pattern
                    if 'announced' in pattern or 'declared' in pattern or 'revealed' in pattern:
                        event_type = 'announcement'
                    elif 'happened' in pattern or 'occurred' in pattern:
                        event_type = 'occurrence'
                    elif 'approved' in pattern or 'passed' in pattern:
                        event_type = 'decision'
                    elif 'started' in pattern or 'began' in pattern:
                        event_type = 'initiation'
                    elif 'completed' in pattern or 'finished' in pattern:
                        event_type = 'completion'
                    elif 'increased' in pattern or 'decreased' in pattern:
                        event_type = 'change'
                    elif 'opened' in pattern or 'closed' in pattern:
                        event_type = 'status_change'
                    else:
                        event_type = 'general'
                    
                    events.append({
                        'text': event_text,
                        'type': event_type
                    })
        
        # Deduplicate events
        seen = set()
        unique_events = []
        for event in events:
            event_key = event['text'].lower()
            if event_key not in seen:
                seen.add(event_key)
                unique_events.append(event)
        
        return unique_events[:5]  # Limit to top 5 events per article
    
    def extract_events(
        self,
        articles: pd.DataFrame,
        city: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract events from articles with improved event extraction.
        
        Args:
            articles: DataFrame with articles
            city: Filter by city
        
        Returns:
            List of extracted events
        """
        events = []
        
        # Filter by city if specified
        if city:
            articles = articles[articles['city'].str.contains(city, case=False, na=False)]
        
        for idx, row in articles.iterrows():
            title = str(row.get('title', 'No title'))
            text = str(row.get('text', row.get('clean_text', '')))
            
            # Extract actual events from text
            extracted_events = self._extract_events_from_text(text, title)
            
            # Normalize date
            date_value = row.get('created_utc') or row.get('published_utc') or row.get('published')
            normalized_date = self._normalize_event_date(date_value)
            
            # If we extracted specific events, create one event per extraction
            if extracted_events:
                for i, evt in enumerate(extracted_events):
                    event = {
                        'id': f"{idx}_event_{i}",
                        'title': f"{evt['text']} ({title[:50]})",
                        'description': text[:500],
                        'date': normalized_date,
                        'city': row.get('city'),
                        'url': row.get('url', ''),
                        'sentiment': row.get('sentiment', 0),
                        'topic': row.get('bertopic') or row.get('lda_topic'),
                        'platform': row.get('platform', 'unknown'),
                        'event_type': evt['type'],
                        'source_article_id': str(idx)
                    }
                    events.append(event)
            else:
                # Fallback: use article as event (original behavior)
                event = {
                    'id': str(idx),
                    'title': title,
                    'description': text[:500],
                    'date': normalized_date,
                    'city': row.get('city'),
                    'url': row.get('url', ''),
                    'sentiment': row.get('sentiment', 0),
                    'topic': row.get('bertopic') or row.get('lda_topic'),
                    'platform': row.get('platform', 'unknown'),
                    'event_type': 'article',
                    'source_article_id': str(idx)
                }
                events.append(event)
        
        # Sort by date if available
        events.sort(key=lambda x: x.get('date') or datetime.min.replace(tzinfo=timezone.utc))
        
        logger.info(f"Extracted {len(events)} events from {len(articles)} articles")
        return events
    
    def _is_temporally_ordered(self, event1: Dict, event2: Dict) -> bool:
        """Check if event1 occurs before event2."""
        date1 = event1.get('date')
        date2 = event2.get('date')
        
        if not date1 or not date2:
            return True  # Can't determine, assume possible
        
        try:
            # Both should be datetime objects after normalization
            if isinstance(date1, datetime) and isinstance(date2, datetime):
                return date1 < date2
            else:
                # Fallback to string comparison
                return str(date1) < str(date2)
        except Exception:
            return True
    
    def _extract_causal_relations_with_dep_parsing(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Extract causal relations using spaCy dependency parsing.
        
        Looks for:
        - ROOT verbs that are causal verbs (cause, lead, trigger, etc.)
        - nsubj (subject/cause) connected to the ROOT
        - dobj (object/effect) connected to the ROOT
        
        Args:
            text: Text to analyze
        
        Returns:
            List of (cause, effect, verb) tuples
        """
        if not self.nlp or not text:
            return []
        
        relations = []
        try:
            doc = self.nlp(text[:1000])  # Limit length for performance
            
            for sent in doc.sents:
                # Find ROOT verb
                root = None
                for token in sent:
                    if token.dep_ == "ROOT" and token.lemma_.lower() in self.causal_verbs:
                        root = token
                        break
                
                if not root:
                    continue
                
                # Find subject (cause) and object (effect)
                cause = None
                effect = None
                
                for token in sent:
                    # Subject (nsubj) is typically the cause
                    if token.dep_ == "nsubj" and token.head == root:
                        cause = token.text
                    # Direct object (dobj) is typically the effect
                    elif token.dep_ == "dobj" and token.head == root:
                        effect = token.text
                    # Also check for prepositional objects (pobj) with causal prepositions
                    elif token.dep_ == "pobj" and token.head.dep_ == "prep":
                        prep = token.head.text.lower()
                        if prep in ['to', 'in', 'from'] and token.head.head == root:
                            if not effect:
                                effect = token.text
                
                # If we found both cause and effect, add relation
                if cause and effect:
                    relations.append((cause, effect, root.lemma_))
        except Exception as e:
            logger.debug(f"Error in dependency parsing: {e}")
        
        return relations
    
    def _calculate_causal_confidence(
        self,
        event1: Dict,
        event2: Dict,
        dep_parse_cache: Dict[str, List[Tuple[str, str, str]]] = None
    ) -> Tuple[float, List[str]]:
        """
        Calculate confidence of causal relationship between two events.
        
        Uses dependency parsing if available, falls back to regex-based approach.
        
        Args:
            event1: First event (potential cause)
            event2: Second event (potential effect)
        
        Returns:
            Tuple of (confidence score 0-1, list of explanation reasons)
        """
        confidence = 0.0
        reasons = []
        
        # Combine text from both events
        text1 = f"{event1.get('title', '')} {event1.get('description', '')}"
        text2 = f"{event2.get('title', '')} {event2.get('description', '')}"
        
        # Try dependency parsing first (more accurate)
        if self.nlp:
            # Use cache if available to avoid re-parsing same text
            if dep_parse_cache is not None:
                if text2 not in dep_parse_cache:
                    dep_parse_cache[text2] = self._extract_causal_relations_with_dep_parsing(text2)
                causal_relations = dep_parse_cache[text2]
            else:
                causal_relations = self._extract_causal_relations_with_dep_parsing(text2)
            
            if causal_relations:
                # Check if event1 concepts match the cause in any relation
                text1_lower = text1.lower()
                for cause, effect, verb in causal_relations:
                    # Check if cause from dependency parsing relates to event1
                    cause_words = set(cause.lower().split())
                    event1_words = set(text1_lower.split())
                    
                    # Remove stopwords
                    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
                    cause_words = cause_words - stopwords
                    event1_words = event1_words - stopwords
                    
                    # Check for overlap
                    shared = cause_words & event1_words
                    meaningful_shared = {w for w in shared if len(w) > 3}
                    
                    if meaningful_shared:
                        confidence += 0.4  # Higher confidence for dependency-parsed relations
                        reasons.append(f"Dependency parsing: '{verb}' connects '{cause}' → '{effect}' (shared: {', '.join(list(meaningful_shared)[:2])})")
                        break
        
        # Fallback to regex-based approach if dependency parsing didn't find anything
        if confidence == 0.0:
            text1_lower = text1.lower()
            text2_lower = text2.lower()
            
            # Check for causal indicators in second event
            for indicator in self.causal_indicators:
                if indicator in text2_lower:
                    # Check if event2 mentions concepts from event1
                    words1 = set(text1_lower.split())
                    words2 = set(text2_lower.split())
                    
                    # Remove common stopwords
                    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
                    words1 = words1 - stopwords
                    words2 = words2 - stopwords
                    
                    # Check for shared meaningful words (length > 4)
                    shared = words1 & words2
                    meaningful_shared = {w for w in shared if len(w) > 4}
                    
                    if meaningful_shared:
                        confidence += 0.3
                        reasons.append(f"Causal indicator '{indicator}' with shared concepts: {', '.join(list(meaningful_shared)[:3])}")
                        break
        
        # Check for same city (stronger relationship)
        if event1.get('city') and event2.get('city'):
            if str(event1['city']).lower() == str(event2['city']).lower():
                confidence += 0.2
                reasons.append(f"Same city: {event1['city']}")
        
        # Check for same topic (related events)
        if event1.get('topic') is not None and event2.get('topic') is not None:
            if event1['topic'] == event2['topic']:
                confidence += 0.2
                reasons.append(f"Same topic: {event1['topic']}")
        
        # Sentiment progression (negative -> positive could indicate improvement)
        sent1 = event1.get('sentiment', 0)
        sent2 = event2.get('sentiment', 0)
        
        if isinstance(sent1, (int, float)) and isinstance(sent2, (int, float)):
            if sent1 < 0 and sent2 > 0:
                confidence += 0.2  # Problem -> solution
                reasons.append("Sentiment progression: negative → positive (problem → solution)")
            elif sent1 > 0 and sent2 < 0:
                confidence += 0.1  # Positive -> negative (regression)
                reasons.append("Sentiment progression: positive → negative (regression)")
        
        # Temporal proximity (events close in time are more likely related)
        date1 = event1.get('date')
        date2 = event2.get('date')
        if date1 and date2 and isinstance(date1, datetime) and isinstance(date2, datetime):
            time_diff = abs((date2 - date1).total_seconds())
            days_diff = time_diff / (24 * 3600)
            if days_diff < 7:
                confidence += 0.1
                reasons.append(f"Temporal proximity: {days_diff:.1f} days apart")
        
        if not reasons:
            reasons.append("Low confidence: minimal signal overlap")
        
        return min(confidence, 1.0), reasons
    
    def detect_causal_relationships(
        self,
        events: List[Dict[str, Any]],
        min_confidence: float = 0.3,
        max_events: int = 500  # Limit events to prevent O(N²) explosion
    ) -> List[Tuple[str, str, float, List[str]]]:
        """
        Detect potential causal relationships between events.
        
        Args:
            events: List of event dictionaries
            min_confidence: Minimum confidence threshold
            max_events: Maximum number of events to process (prevents O(N²) explosion)
        
        Returns:
            List of (cause_id, effect_id, confidence, reasons) tuples
        """
        relationships = []
        
        # Limit events if too many (prevents O(N²) explosion)
        if len(events) > max_events:
            logger.warning(f"Too many events ({len(events)}). Limiting to {max_events} most recent events for performance.")
            # Sort by date and take most recent
            events_with_dates = [(e, e.get('date') or datetime.min.replace(tzinfo=timezone.utc)) for e in events]
            events_with_dates.sort(key=lambda x: x[1], reverse=True)
            events = [e for e, _ in events_with_dates[:max_events]]
            logger.info(f"Processing {len(events)} events (limited from original set)")
        
        total_comparisons = len(events) * (len(events) - 1) // 2
        logger.info(f"Detecting causal relationships: {len(events)} events, ~{total_comparisons:,} comparisons")
        
        # Cache dependency parsing results to avoid re-parsing same text
        dep_parse_cache = {}
        
        processed = 0
        for i, event1 in enumerate(events):
            if i % 50 == 0 and i > 0:
                logger.info(f"Progress: {i}/{len(events)} events processed, {len(relationships)} relationships found so far...")
            
            for event2 in events[i+1:]:  # Only look at later events
                processed += 1
                
                # Check temporal ordering
                if not self._is_temporally_ordered(event1, event2):
                    continue
                
                # Calculate causal confidence with explanations
                confidence, reasons = self._calculate_causal_confidence(event1, event2, dep_parse_cache)
                
                if confidence >= min_confidence:
                    relationships.append((event1['id'], event2['id'], confidence, reasons))
        
        logger.info(f"Detected {len(relationships)} causal relationships from {processed:,} comparisons")
        return relationships
    
    def _check_dataset_guardrails(
        self,
        events: List[Dict[str, Any]],
        min_events: int = 5,
        min_edges: int = 2,
        min_time_span_days: int = 1
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if dataset meets minimum requirements for causal analysis.
        
        Args:
            events: List of events
            min_events: Minimum number of events required
            min_edges: Minimum number of edges expected (after relationship detection)
            min_time_span_days: Minimum time span in days
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if len(events) < min_events:
            return False, f"Insufficient events: {len(events)} < {min_events}. Need at least {min_events} events for meaningful causal analysis."
        
        # Check time span
        dates = [e.get('date') for e in events if e.get('date')]
        if len(dates) >= 2:
            valid_dates = [d for d in dates if isinstance(d, datetime)]
            if len(valid_dates) >= 2:
                time_span = (max(valid_dates) - min(valid_dates)).total_seconds() / (24 * 3600)
                if time_span < min_time_span_days:
                    return False, f"Insufficient time span: {time_span:.1f} days < {min_time_span_days} days. Events need to span at least {min_time_span_days} days for temporal analysis."
        
        # Note: We can't check min_edges here because relationships haven't been detected yet
        # This will be checked after relationship detection
        
        return True, None
    
    def build_graph(
        self,
        articles: pd.DataFrame,
        city: Optional[str] = None,
        min_confidence: float = 0.3,
        min_events: int = 5,
        min_edges: int = 2,
        min_time_span_days: int = 1
    ) -> Optional[nx.DiGraph]:
        """
        Build causal graph from articles with guardrails.
        
        Args:
            articles: DataFrame with articles
            city: Filter by city
            min_confidence: Minimum confidence for including relationships
            min_events: Minimum number of events required
            min_edges: Minimum number of edges required
            min_time_span_days: Minimum time span in days
        
        Returns:
            NetworkX directed graph or None if guardrails fail
        """
        if nx is None:
            logger.warning("networkx not available. Cannot build causal graph.")
            return None
        
        self.graph.clear()
        
        # Extract events
        events = self.extract_events(articles, city)
        
        # Check guardrails
        is_valid, error_msg = self._check_dataset_guardrails(events, min_events, min_time_span_days)
        if not is_valid:
            logger.warning(f"Causal graph guardrails failed: {error_msg}")
            return None
        
        # Add events as nodes
        for event in events:
            self.graph.add_node(
                event['id'],
                title=event['title'][:100],  # Truncate for display
                date=str(event.get('date', '')),
                city=event.get('city', 'Unknown'),
                sentiment=event.get('sentiment', 0),
                url=event.get('url', ''),
                event_type=event.get('event_type', 'unknown')
            )
        
        # Detect and add causal relationships as edges
        relationships = self.detect_causal_relationships(events, min_confidence)
        
        # Check edge guardrail
        if len(relationships) < min_edges:
            logger.warning(f"Insufficient relationships: {len(relationships)} < {min_edges}. Need at least {min_edges} relationships for meaningful causal graph.")
            return None
        
        for cause_id, effect_id, confidence, reasons in relationships:
            self.graph.add_edge(
                cause_id,
                effect_id,
                confidence=confidence,
                weight=confidence,
                reasons=reasons,  # Store explanation reasons
                explanation="; ".join(reasons)  # Human-readable explanation
            )
        
        logger.info(f"Built causal graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        
        return self.graph
    
    def save_graph(self, path: Path | None = None) -> Path:
        """
        Save the causal graph to disk using pickle.
        
        Args:
            path: Optional path to save graph. If None, uses default location.
        
        Returns:
            Path where graph was saved
        """
        if path is None:
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            path = MODELS_DIR / "causal_graph.pkl"
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save graph and builder metadata
        graph_data = {
            "graph": self.graph,
            "summary": self.get_graph_summary()
        }
        
        save_pickle(graph_data, path)
        logger.info(f"Causal graph saved to {path}")
        return path
    
    @staticmethod
    def load_graph(path: Path | None = None) -> Tuple[Optional[nx.DiGraph], Optional[Dict[str, Any]]]:
        """
        Load a pre-computed causal graph from disk.
        
        Args:
            path: Optional path to load graph from. If None, uses default location.
        
        Returns:
            Tuple of (graph, summary_dict) or (None, None) if not found
        """
        if path is None:
            path = MODELS_DIR / "causal_graph.pkl"
        
        path = Path(path)
        if not path.exists():
            logger.debug(f"Causal graph not found at {path}")
            return None, None
        
        try:
            graph_data = load_pickle(path)
            graph = graph_data.get("graph")
            summary = graph_data.get("summary", {})
            
            if graph is None:
                logger.warning(f"Graph data at {path} is invalid")
                return None, None
            
            logger.info(f"Loaded causal graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
            return graph, summary
        except Exception as e:
            logger.error(f"Error loading causal graph from {path}: {e}")
            return None, None
    
    def get_edge_explanation(self, source: str, target: str) -> Optional[str]:
        """Get explanation for why an edge exists."""
        if self.graph.has_edge(source, target):
            return self.graph[source][target].get('explanation', 'No explanation available')
        return None
    
    def get_graph_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the causal graph."""
        if self.graph.number_of_nodes() == 0:
            return {"message": "Graph is empty"}
        
        return {
            'num_events': self.graph.number_of_nodes(),
            'num_relationships': self.graph.number_of_edges(),
            'root_causes': len(self.get_root_causes()),
            'final_effects': len(self.get_effects()),
            'avg_degree': sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes() if self.graph.number_of_nodes() > 0 else 0,
            'is_dag': nx.is_directed_acyclic_graph(self.graph) if nx else False
        }
    
    def get_root_causes(self) -> List[str]:
        """Get root causes (nodes with no incoming edges)."""
        return [node for node in self.graph.nodes() if self.graph.in_degree(node) == 0]
    
    def get_effects(self) -> List[str]:
        """Get final effects (nodes with no outgoing edges)."""
        return [node for node in self.graph.nodes() if self.graph.out_degree(node) == 0]
    
    def get_causal_chains(
        self,
        event_id: str,
        max_depth: int = 3
    ) -> List[List[str]]:
        """
        Get causal chains starting from an event.
        
        Args:
            event_id: Starting event ID
            max_depth: Maximum chain length
        
        Returns:
            List of causal chains (lists of event IDs)
        """
        if event_id not in self.graph:
            return []
        
        chains = []
        
        def dfs(node, path, depth):
            if depth > max_depth:
                return
            
            successors = list(self.graph.successors(node))
            
            if not successors:
                chains.append(path)
                return
            
            for successor in successors:
                dfs(successor, path + [successor], depth + 1)
        
        dfs(event_id, [event_id], 1)
        
        return chains
    
    def export_graph(self, filename: str = "data/causal_graph.gexf"):
        """Export graph to GEXF file for visualization."""
        if nx is None:
            logger.warning("networkx not available. Cannot export graph.")
            return
        
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        nx.write_gexf(self.graph, str(output_path))
        logger.info(f"Exported causal graph to {output_path}")
    
    def create_interactive_html(
        self,
        max_nodes: int = 100,
        height: str = "600px",
        width: str = "100%"
    ) -> str:
        """
        Create interactive HTML visualization of the causal graph with confidence explanations.
        
        Args:
            max_nodes: Maximum number of nodes to display (for performance)
            height: Height of the visualization
            width: Width of the visualization
        
        Returns:
            HTML string for the visualization
        """
        try:
            from pyvis.network import Network
        except ImportError:
            logger.warning("pyvis not installed. Install via `pip install pyvis`")
            return "<p>pyvis not installed. Install via `pip install pyvis`</p>"
        
        if self.graph.number_of_nodes() == 0:
            return "<p>Graph is empty</p>"
        
        # Create pyvis network
        height_px = height if 'px' in height else f"{height.replace('vh', '').replace('%', '')}px"
        width_px = width if 'px' in width or '%' in width else f"{width}"
        
        net = Network(
            height=height_px,
            width=width_px,
            directed=True,
            notebook=False,
            cdn_resources='remote',
            bgcolor="#ffffff",
            font_color="#000000"
        )
        
        # Limit nodes for performance
        nodes_to_show = list(self.graph.nodes())[:max_nodes]
        subgraph = self.graph.subgraph(nodes_to_show)
        
        # Add nodes with styling
        for node_id in subgraph.nodes():
            node_data = self.graph.nodes[node_id]
            title = node_data.get("title", "Unknown")
            city = node_data.get("city", "Unknown")
            sentiment = node_data.get("sentiment", 0)
            
            # Color by sentiment
            if isinstance(sentiment, (int, float)):
                if sentiment > 0.1:
                    color = "#90EE90"  # Light green (positive)
                elif sentiment < -0.1:
                    color = "#FFB6C1"  # Light pink (negative)
                else:
                    color = "#D3D3D3"  # Light gray (neutral)
            else:
                color = "#D3D3D3"
            
            # Size by degree (more connections = larger)
            degree = self.graph.degree(node_id)
            size = 15 + min(degree * 1.5, 25)
            
            # Create tooltip
            tooltip = f"<b>{title[:100]}</b><br>City: {city}<br>Connections: {degree}"
            if node_data.get("url"):
                tooltip += f"<br><a href='{node_data['url']}' target='_blank'>View Article</a>"
            
            net.add_node(
                str(node_id),
                label=title[:50] + "..." if len(title) > 50 else title,
                title=tooltip,
                color=color,
                size=size
            )
        
        # Add edges with confidence as weight and explanation in tooltip
        for source, target, edge_data in subgraph.edges(data=True):
            confidence = edge_data.get("confidence", 0.5)
            weight = confidence * 5  # Scale for visibility
            explanation = edge_data.get("explanation", "No explanation available")
            
            # Color edge by confidence
            if confidence > 0.7:
                edge_color = "#FF0000"  # Red (high confidence)
            elif confidence > 0.5:
                edge_color = "#FFA500"  # Orange (medium confidence)
            else:
                edge_color = "#808080"  # Gray (low confidence)
            
            # Enhanced tooltip with explanation
            edge_tooltip = f"<b>Confidence: {confidence:.2f}</b><br>{explanation}"
            
            net.add_edge(
                str(source),
                str(target),
                value=weight,
                color=edge_color,
                title=edge_tooltip
            )
        
        # Configure physics for better layout
        net.set_options("""
        {
          "physics": {
            "enabled": true,
            "forceAtlas2Based": {
              "gravitationalConstant": -50,
              "centralGravity": 0.01,
              "springLength": 200,
              "springConstant": 0.08,
              "damping": 0.4,
              "avoidOverlap": 0.5
            },
            "solver": "forceAtlas2Based",
            "stabilization": {
              "enabled": true,
              "iterations": 200,
              "updateInterval": 25
            }
          },
          "interaction": {
            "zoomView": true,
            "dragView": true,
            "dragNodes": true,
            "selectConnectedEdges": true,
            "hover": true,
            "tooltipDelay": 100
          },
          "layout": {
            "improvedLayout": true,
            "hierarchical": {
              "enabled": false
            }
          },
          "nodes": {
            "scaling": {
              "min": 10,
              "max": 30
            },
            "font": {
              "size": 12
            }
          },
          "edges": {
            "smooth": {
              "type": "continuous",
              "roundness": 0.5
            },
            "scaling": {
              "min": 1,
              "max": 5
            }
          }
        }
        """)
        
        # Generate HTML
        html = net.generate_html()
        
        # Add custom CSS to ensure graph fills entire container
        height_value = height_px.replace('px', '')
        try:
            height_num = int(height_value)
        except:
            height_num = 900
        
        custom_css = f"""
        <style>
            * {{
                box-sizing: border-box;
            }}
            body, html {{
                margin: 0 !important;
                padding: 0 !important;
                width: 100% !important;
                height: 100% !important;
                overflow: hidden !important;
            }}
            #mynetwork, #mynetworkid {{
                width: 100% !important;
                height: {height_num}px !important;
                min-height: {height_num}px !important;
                max-height: {height_num}px !important;
                position: relative !important;
                display: block !important;
            }}
            #mynetwork canvas, #mynetworkid canvas {{
                width: 100% !important;
                height: 100% !important;
                display: block !important;
            }}
            .card {{
                width: 100% !important;
                height: {height_num}px !important;
            }}
            .card-body {{
                width: 100% !important;
                height: 100% !important;
                padding: 0 !important;
            }}
            .vis-network {{
                width: 100% !important;
                height: 100% !important;
                position: absolute !important;
                top: 0 !important;
                left: 0 !important;
                bottom: 0 !important;
                right: 0 !important;
            }}
        </style>
        """
        
        # Extract body content for Streamlit compatibility
        if '<body>' in html:
            start = html.find('<body>') + 6
            end = html.find('</body>')
            if start > 5 and end > start:
                body_content = html[start:end]
                scripts = []
                styles = []
                
                # Extract script tags
                script_start = html.find('<script')
                while script_start != -1:
                    script_end = html.find('</script>', script_start) + 9
                    if script_end > script_start:
                        scripts.append(html[script_start:script_end])
                    script_start = html.find('<script', script_end)
                
                # Extract style/link tags
                link_start = html.find('<link')
                while link_start != -1:
                    link_end = html.find('>', link_start) + 1
                    if link_end > link_start:
                        styles.append(html[link_start:link_end])
                    link_start = html.find('<link', link_end)
                
                # Modify body_content to ensure mynetwork div has proper sizing
                import re
                body_content = re.sub(
                    r'id="mynetwork"[^>]*style="[^"]*"',
                    'id="mynetwork"',
                    body_content
                )
                body_content = re.sub(
                    r'id="mynetworkid"[^>]*style="[^"]*"',
                    'id="mynetworkid"',
                    body_content
                )
                if 'id="mynetwork"' in body_content:
                    body_content = body_content.replace(
                        'id="mynetwork"',
                        'id="mynetwork" style="width:100%; height:100%;"'
                    )
                
                # Combine into a complete HTML document with custom CSS
                combined = '<html><head>'
                combined += ''.join(styles)
                combined += custom_css
                combined += '</head><body style="margin:0; padding:0; width:100%; height:100%; overflow:hidden;">'
                combined += ''.join(scripts)
                combined += body_content
                combined += '</body></html>'
                return combined
        
        return html


def build_causal_graph_for_city(
    city: str,
    min_confidence: float = 0.3
) -> Optional[nx.DiGraph]:
    """
    Build causal graph for a specific city.
    
    Args:
        city: City name
        min_confidence: Minimum confidence threshold
    
    Returns:
        NetworkX directed graph or None
    """
    # Load corpus
    corpus_path = PROC_DIR / "corpus_bertopic.parquet"
    if not corpus_path.exists():
        corpus_path = PROC_DIR / "corpus.parquet"
    
    if not corpus_path.exists():
        logger.warning("Corpus not found. Run pipeline first.")
        return None
    
    articles = load_parquet(corpus_path)
    
    builder = CausalGraphBuilder()
    graph = builder.build_graph(articles, city=city, min_confidence=min_confidence)
    
    return graph


if __name__ == "__main__":
    # Test causal graph builder
    from ..utils import setup_logging
    
    setup_logging()
    
    # Load sample data
    corpus_path = PROC_DIR / "corpus_bertopic.parquet"
    if corpus_path.exists():
        articles = load_parquet(corpus_path)
        builder = CausalGraphBuilder()
        graph = builder.build_graph(articles, min_confidence=0.3)
        
        if graph:
            summary = builder.get_graph_summary()
            print("Graph Summary:", summary)
            print("Root Causes:", builder.get_root_causes()[:5])
            print("Final Effects:", builder.get_effects()[:5])
    else:
        print("Corpus not found. Run pipeline first.")
