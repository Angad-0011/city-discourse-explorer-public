"""Streamlit UI for semantic search."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

# Add project root to path for imports when running directly with Streamlit
# This handles the case where Streamlit runs the script directly
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import pandas as pd
import streamlit as st
from loguru import logger

from src.config import INDEX_DIR, MODELS_DIR, PROC_DIR, SBERT_MODEL_NAME
from src.utils_io import log
from src.utils_topic_labels import get_topic_labels, get_all_topic_labels, load_bertopic_model

try:
    import faiss
except ImportError:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


@st.cache_resource(show_spinner=False)
def load_index():
    """Load FAISS index and document ID mapping."""
    try:
        from src.search.index_search import load_index as _load_index
        index, doc_id_mapping = _load_index()
        return index, doc_id_mapping
    except Exception as e:
        logger.warning(f"Could not load index: {e}")
        return None, {}


@st.cache_resource(show_spinner=False)
def load_model():
    """Load SentenceTransformer model."""
    if SentenceTransformer is None:
        return None
    return SentenceTransformer(SBERT_MODEL_NAME)


@st.cache_data(show_spinner=False)
def load_corpus() -> pd.DataFrame:
    """Load processed corpus with all enrichments."""
    # Try to load the most enriched version (BERTopic > LDA > base)
    paths = [
        PROC_DIR / "corpus_bertopic.parquet",  # Has BERTopic topics (best)
        PROC_DIR / "corpus_lda.parquet",  # Has LDA topics
        PROC_DIR / "corpus_ner_sent.parquet",  # Has sentiment
        PROC_DIR / "corpus.parquet",  # Base corpus
    ]
    
    df = None
    for path in paths:
        if path.exists():
            df = pd.read_parquet(path)
            break
    
    if df is None or df.empty:
        return pd.DataFrame()
    
    # Try to merge additional data if available
    # Merge BERTopic if not in current df
    if "bertopic" not in df.columns:
        bertopic_path = PROC_DIR / "corpus_bertopic.parquet"
        if bertopic_path.exists():
            df_bertopic = pd.read_parquet(bertopic_path)
            if "bertopic" in df_bertopic.columns:
                df = df.merge(df_bertopic[["bertopic"]], left_index=True, right_index=True, how="left")
    
    # Merge LDA if not in current df
    if "lda_topic" not in df.columns:
        lda_path = PROC_DIR / "corpus_lda.parquet"
        if lda_path.exists():
            df_lda = pd.read_parquet(lda_path)
            if "lda_topic" in df_lda.columns:
                df = df.merge(df_lda[["lda_topic"]], left_index=True, right_index=True, how="left")
    
    # Merge sentiment if not in current df
    if "sentiment" not in df.columns:
        sent_path = PROC_DIR / "corpus_ner_sent.parquet"
        if sent_path.exists():
            df_sent = pd.read_parquet(sent_path)
            if "sentiment" in df_sent.columns:
                df = df.merge(df_sent[["sentiment"]], left_index=True, right_index=True, how="left")
    
    return df


def get_sentiment_label(score: float) -> str:
    """Convert sentiment score to label."""
    if pd.isna(score):
        return "Neutral"
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"


def encode_query(model, query: str) -> Optional[np.ndarray]:
    """Encode query to embedding."""
    if model is None:
        return None
    return model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)


st.set_page_config(page_title="City Discourse Explorer", layout="wide")
st.title("🏙️ City Discourse Explorer")
st.markdown("Semantic search and analysis of city-specific news and discussions")

corpus = load_corpus()
model = load_model()
index, doc_id_mapping = load_index()

# Load topic labels for better display
@st.cache_resource(show_spinner=False)
def load_topic_labels():
    """Load BERTopic labels mapping."""
    return get_all_topic_labels()

topic_labels = load_topic_labels()

if corpus.empty:
    st.warning("Processed corpus not found. Run the pipeline first.")
    st.stop()
else:
    # Show corpus statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Documents", len(corpus))
    with col2:
        cities_count = corpus["city"].nunique() if "city" in corpus.columns else 0
        st.metric("Cities", cities_count)
    with col3:
        if "bertopic" in corpus.columns:
            topics_count = len([t for t in corpus["bertopic"].dropna().unique() if t != -1])
            topic_type = "BERTopic"
        elif "lda_topic" in corpus.columns:
            topics_count = corpus["lda_topic"].nunique()
            topic_type = "LDA"
        else:
            topics_count = 0
            topic_type = "None"
        st.metric(f"Topics ({topic_type})", topics_count)
    with col4:
        has_sentiment = "sentiment" in corpus.columns
        st.metric("Sentiment", "✅" if has_sentiment else "❌")

# Sidebar for filters
with st.sidebar:
    st.header("🔍 Filters")
    
    # Topic filter - prefer BERTopic, fallback to LDA
    topic_type = "BERTopic" if "bertopic" in corpus.columns else "LDA" if "lda_topic" in corpus.columns else None
    topic_filter = "All"
    
    if topic_type == "BERTopic":
        topics = sorted([t for t in corpus["bertopic"].dropna().unique() if t != -1])  # Exclude outliers
        topic_options = ["All"] + [f"BERTopic {int(t)}" for t in topics] + ["Outliers (-1)"]
        topic_filter = st.selectbox("Topic Filter (BERTopic)", topic_options, index=0)
    elif topic_type == "LDA":
        topics = sorted(corpus["lda_topic"].dropna().unique())
        topic_options = ["All"] + [f"LDA Topic {int(t)}" for t in topics]
        topic_filter = st.selectbox("Topic Filter (LDA)", topic_options, index=0)
    
    # Sentiment filter
    if "sentiment" in corpus.columns:
        sentiment_options = ["All", "Positive", "Neutral", "Negative"]
        sentiment_filter = st.selectbox("Sentiment Filter", sentiment_options)
    else:
        sentiment_filter = "All"
    
    # Platform filter
    if "platform" in corpus.columns:
        platforms = sorted(corpus["platform"].dropna().unique())
        platform_filter = st.selectbox("Platform Filter", ["All"] + list(platforms))
    else:
        platform_filter = "All"
    
    st.markdown("---")
    st.header("📊 Statistics")
    
    # Topic distribution - prefer BERTopic
    if "bertopic" in corpus.columns:
        st.subheader("BERTopic Distribution")
        topic_counts = corpus["bertopic"].value_counts().sort_index()
        # Rename -1 to "Outliers" for display
        topic_counts.index = topic_counts.index.map(lambda x: "Outliers" if x == -1 else f"Topic {int(x)}")
        st.bar_chart(topic_counts)
    elif "lda_topic" in corpus.columns:
        st.subheader("LDA Topic Distribution")
        topic_counts = corpus["lda_topic"].value_counts().sort_index()
        topic_counts.index = topic_counts.index.map(lambda x: f"Topic {int(x)}")
        st.bar_chart(topic_counts)
    
    # Sentiment distribution
    if "sentiment" in corpus.columns:
        st.subheader("Sentiment Distribution")
        sentiment_labels = corpus["sentiment"].apply(get_sentiment_label)
        sentiment_counts = sentiment_labels.value_counts()
        st.bar_chart(sentiment_counts)

# Tabs for different views
tab1, tab2, tab3 = st.tabs(["🔎 Single Search", "🏙️ Cross-City Comparison", "🔗 Causal Analysis"])

# Tab 1: Single Search
with tab1:
    st.header("🔎 Semantic Search")
    query = st.text_input("Search query", "waste management infrastructure", key="single_search_query")

    # Get unique cities and normalize them
    if not corpus.empty and "city" in corpus.columns:
        cities = set()
        for c in corpus["city"].dropna():
            if c:
                from src.utils import normalize_city_name
                normalized = normalize_city_name(str(c))
                if normalized:
                    cities.add(normalized)
        city_options = ["All"] + sorted(cities)
    else:
        city_options = ["All"]

    city_filter = st.selectbox("City filter", city_options, index=0)
    top_k = st.slider("Top K", min_value=5, max_value=50, value=10, step=5)

    if st.button("Search", key="single_search_button"):
        if index is None or model is None:
            st.error("Missing FAISS index or SBERT model. Run the pipeline and build the index.")
        else:
            query_vec = encode_query(model, query)
            if query_vec is None:
                st.error("Unable to encode query. Install sentence-transformers.")
            else:
                # Fetch a large k to ensure we have enough results after filtering
                # If filtering by city/topic, we need many more candidates than top_k
                search_k = max(500, top_k * 10)  # Fetch at least 500, or 10x top_k, whichever is larger
                max_search = min(search_k, index.ntotal)  # Don't exceed index size
                scores, indices = index.search(query_vec, max_search)
                rows = []
                seen_doc_ids = set()
                
                for score, idx in zip(scores[0], indices[0]):
                    if idx == -1:
                        continue
                    
                    # Use doc_id mapping if available, otherwise fall back to iloc
                    row = None
                    if doc_id_mapping and "doc_id" in corpus.columns:
                        # Find doc_id by reverse lookup
                        for doc_id, pos in doc_id_mapping.items():
                            if pos == int(idx):
                                matching_rows = corpus[corpus["doc_id"] == doc_id]
                                if len(matching_rows) > 0:
                                    row = matching_rows.iloc[0]
                                    # Deduplicate
                                    if doc_id in seen_doc_ids:
                                        continue
                                    seen_doc_ids.add(doc_id)
                                break
                    
                    # Fallback to iloc if mapping not available
                    if row is None:
                        if int(idx) < len(corpus):
                            row = corpus.iloc[int(idx)]
                        else:
                            continue
                    
                    # Normalize city name
                    city = row.get("city")
                    if city:
                        from src.utils import normalize_city_name
                        city = normalize_city_name(str(city))
                    
                    # Apply city filter
                    if city_filter != "All" and city:
                        if city_filter.lower() not in str(city).lower():
                            continue
                    
                    # Apply topic filter
                    if topic_filter != "All":
                        if "Outliers" in topic_filter or "(-1)" in topic_filter:
                            if row.get("bertopic") != -1:
                                continue
                        else:
                            # Extract topic number from filter (could be "Label (0)" or "BERTopic 0")
                            try:
                                if "(" in topic_filter and ")" in topic_filter:
                                    # Format: "Label (0)"
                                    topic_num = int(topic_filter.split("(")[1].split(")")[0])
                                else:
                                    # Format: "BERTopic 0"
                                    topic_num = int(topic_filter.split()[-1])
                            except (ValueError, IndexError):
                                continue
                            
                            # Check BERTopic first, then LDA
                            if "bertopic" in row and pd.notna(row.get("bertopic")):
                                if row.get("bertopic") != topic_num:
                                    continue
                            elif "lda_topic" in row and pd.notna(row.get("lda_topic")):
                                if row.get("lda_topic") != topic_num:
                                    continue
                    
                    # Apply sentiment filter
                    if sentiment_filter != "All" and "sentiment" in row:
                        sentiment_score = row.get("sentiment", 0)
                        sentiment_label = get_sentiment_label(sentiment_score)
                        if sentiment_label != sentiment_filter:
                            continue
                    
                    # Apply platform filter
                    if platform_filter != "All":
                        if str(row.get("platform", "")).lower() != platform_filter.lower():
                            continue
                    
                    # Get text snippet for preview
                    text_col = row.get("text", row.get("clean_text", row.get("lemma_text", "")))
                    preview = ""
                    if text_col and len(str(text_col)) > 0:
                        preview = str(text_col)[:200] + "..." if len(str(text_col)) > 200 else str(text_col)
                    
                    # Get topic - prefer BERTopic, fallback to LDA
                    topic_str = "N/A"
                    if "bertopic" in row and pd.notna(row.get("bertopic")):
                        bertopic = int(row.get("bertopic"))
                        # Use topic labels if available
                        if bertopic in topic_labels:
                            topic_str = topic_labels[bertopic]
                        elif bertopic == -1:
                            topic_str = "Outliers"
                        else:
                            topic_str = f"BERTopic {bertopic}"
                    elif "lda_topic" in row and pd.notna(row.get("lda_topic")):
                        topic = row.get("lda_topic")
                        topic_str = f"LDA Topic {int(topic)}"
                    
                    # Get sentiment
                    sentiment_score = row.get("sentiment", None)
                    sentiment_label = get_sentiment_label(sentiment_score) if pd.notna(sentiment_score) else "N/A"
                    sentiment_emoji = {
                        "Positive": "😊",
                        "Negative": "😞",
                        "Neutral": "😐",
                        "N/A": "❓"
                    }.get(sentiment_label, "❓")
                    
                    rows.append({
                        "Score": float(score),
                        "City": city or "Unknown",
                        "Topic": topic_str,
                        "Sentiment": f"{sentiment_emoji} {sentiment_label}",
                        "Title": row.get("title", "No title"),
                        "Preview": preview,
                        "URL": row.get("url", ""),
                    })
                    
                    # Stop once we have enough results
                    if len(rows) >= top_k:
                        break
                
                # Limit to top_k results after filtering
                rows = rows[:top_k]
                
                if not rows:
                    st.info("No results found. Try adjusting your filters or search query.")
                else:
                    # Display results with better formatting
                    df_results = pd.DataFrame(rows)
                    
                    # Show summary with statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Results Found", len(rows))
                    with col2:
                        avg_score = df_results["Score"].mean()
                        st.metric("Avg Relevance", f"{avg_score:.3f}")
                    with col3:
                        unique_cities = df_results["City"].nunique()
                        st.metric("Cities", unique_cities)
                    
                    # Show topic and sentiment breakdown
                    if len(rows) > 0:
                        col1, col2 = st.columns(2)
                        with col1:
                            if "Topic" in df_results.columns:
                                st.subheader("📊 Topic Distribution")
                                topic_counts = df_results["Topic"].value_counts()
                                st.bar_chart(topic_counts)
                        with col2:
                            if "Sentiment" in df_results.columns:
                                st.subheader("😊 Sentiment Distribution")
                                sentiment_counts = df_results["Sentiment"].value_counts()
                                st.bar_chart(sentiment_counts)
                    
                    st.markdown("---")
                    st.subheader("📄 Search Results")
                    
                    # Display results table
                    st.dataframe(
                        df_results,
                        width='stretch',
                        hide_index=True,
                        column_config={
                            "Score": st.column_config.NumberColumn(
                                "Relevance",
                                format="%.4f",
                                help="Semantic similarity score (higher is better)"
                            ),
                            "City": st.column_config.TextColumn("City"),
                            "Topic": st.column_config.TextColumn("Topic"),
                            "Sentiment": st.column_config.TextColumn("Sentiment"),
                            "Title": st.column_config.TextColumn("Title", width="large"),
                            "Preview": st.column_config.TextColumn("Preview", width="large"),
                            "URL": st.column_config.LinkColumn("Link"),
                        }
                    )
                    
                    # Export button
                    csv = df_results.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name=f"search_results_{query[:20].replace(' ', '_')}.csv",
                        mime="text/csv"
                    )

# Tab 2: Cross-City Comparison
with tab2:
    st.header("🏙️ Cross-City Comparison")
    st.markdown("Compare how the same topic is discussed across different cities")
    
    comparison_query = st.text_input("Search query", "waste management infrastructure", key="comparison_query")
    
    # Get available cities
    if not corpus.empty and "city" in corpus.columns:
        available_cities = set()
        for c in corpus["city"].dropna():
            if c:
                from src.utils import normalize_city_name
                normalized = normalize_city_name(str(c))
                if normalized:
                    available_cities.add(normalized)
        available_cities = sorted(available_cities)
    else:
        available_cities = []
    
    if not available_cities:
        st.warning("No cities found in corpus. Run the pipeline to collect data.")
    else:
        # Multi-select for cities
        selected_cities = st.multiselect(
            "Select cities to compare",
            available_cities,
            default=available_cities[:3] if len(available_cities) >= 3 else available_cities,
            key="comparison_cities"
        )
        
        comparison_top_k = st.slider("Results per city", min_value=5, max_value=30, value=10, step=5, key="comparison_top_k")
        
        if st.button("Compare Cities", key="compare_button"):
            if not selected_cities:
                st.warning("Please select at least one city to compare.")
            elif index is None or model is None:
                st.error("Missing FAISS index or SBERT model. Run the pipeline and build the index.")
            else:
                query_vec = encode_query(model, comparison_query)
                if query_vec is None:
                    st.error("Unable to encode query. Install sentence-transformers.")
                else:
                    # Search across all cities, then filter by selected cities
                    scores, indices = index.search(query_vec, comparison_top_k * len(selected_cities) * 2)  # Get more to filter
                    
                    # Organize results by city
                    city_results = {city: [] for city in selected_cities}
                    seen_doc_ids = set()
                    
                    for score, idx in zip(scores[0], indices[0]):
                        if idx == -1:
                            continue
                        
                        # Use doc_id mapping if available, otherwise fall back to iloc
                        row = None
                        if doc_id_mapping and "doc_id" in corpus.columns:
                            # Find doc_id by reverse lookup
                            for doc_id, pos in doc_id_mapping.items():
                                if pos == int(idx):
                                    matching_rows = corpus[corpus["doc_id"] == doc_id]
                                    if len(matching_rows) > 0:
                                        row = matching_rows.iloc[0]
                                        # Deduplicate
                                        if doc_id in seen_doc_ids:
                                            continue
                                        seen_doc_ids.add(doc_id)
                                    break
                        
                        # Fallback to iloc if mapping not available
                        if row is None:
                            if int(idx) < len(corpus):
                                row = corpus.iloc[int(idx)]
                            else:
                                continue
                        
                        # Normalize city name
                        city = row.get("city")
                        if city:
                            from src.utils import normalize_city_name
                            city = normalize_city_name(str(city))
                        
                        # Check if this city is selected
                        if city and city in selected_cities:
                            if len(city_results[city]) >= comparison_top_k:
                                continue  # Already have enough for this city
                            
                            # Get text snippet
                            text_col = row.get("text", row.get("clean_text", row.get("lemma_text", "")))
                            preview = ""
                            if text_col and len(str(text_col)) > 0:
                                preview = str(text_col)[:200] + "..." if len(str(text_col)) > 200 else str(text_col)
                            
                            # Get topic - use labels if available
                            topic_str = "N/A"
                            if "bertopic" in row and pd.notna(row.get("bertopic")):
                                bertopic = int(row.get("bertopic"))
                                if bertopic in topic_labels:
                                    topic_str = topic_labels[bertopic]
                                elif bertopic == -1:
                                    topic_str = "Outliers"
                                else:
                                    topic_str = f"BERTopic {bertopic}"
                            elif "lda_topic" in row and pd.notna(row.get("lda_topic")):
                                topic_str = f"LDA Topic {int(row.get('lda_topic'))}"
                            
                            # Get sentiment
                            sentiment_score = row.get("sentiment", None)
                            sentiment_label = get_sentiment_label(sentiment_score) if pd.notna(sentiment_score) else "N/A"
                            
                            city_results[city].append({
                                "score": float(score),
                                "title": row.get("title", "No title"),
                                "preview": preview,
                                "topic": topic_str,
                                "sentiment": sentiment_label,
                                "url": row.get("url", ""),
                            })
                    
                    # Display comparison results
                    if not any(city_results.values()):
                        st.info("No results found for the selected cities. Try a different query or select different cities.")
                    else:
                        # Summary statistics
                        st.subheader("📊 Comparison Summary")
                        summary_cols = st.columns(len(selected_cities) + 1)
                        with summary_cols[0]:
                            st.metric("Total Results", sum(len(results) for results in city_results.values()))
                        
                        for i, city in enumerate(selected_cities, 1):
                            with summary_cols[i]:
                                count = len(city_results[city])
                                avg_score = np.mean([r["score"] for r in city_results[city]]) if city_results[city] else 0
                                st.metric(city, f"{count} results", f"Avg: {avg_score:.3f}")
                        
                        # Topic distribution comparison
                        st.markdown("---")
                        st.subheader("📚 Topic Distribution by City")
                        
                        # Collect all topics by city
                        city_topics = {}
                        for city in selected_cities:
                            topics = [r["topic"] for r in city_results[city]]
                            topic_counts = pd.Series(topics).value_counts()
                            city_topics[city] = topic_counts
                        
                        # Create comparison chart
                        if city_topics:
                            topic_df = pd.DataFrame(city_topics).fillna(0)
                            st.bar_chart(topic_df)
                        
                        # Sentiment comparison
                        st.markdown("---")
                        st.subheader("😊 Sentiment Distribution by City")
                        
                        city_sentiments = {}
                        for city in selected_cities:
                            sentiments = [r["sentiment"] for r in city_results[city]]
                            sentiment_counts = pd.Series(sentiments).value_counts()
                            city_sentiments[city] = sentiment_counts
                        
                        if city_sentiments:
                            sentiment_df = pd.DataFrame(city_sentiments).fillna(0)
                            st.bar_chart(sentiment_df)
                        
                        # Side-by-side results per city
                        st.markdown("---")
                        st.subheader("📄 Top Results by City")
                        
                        # Create columns for each city
                        city_cols = st.columns(len(selected_cities))
                        
                        for col_idx, city in enumerate(selected_cities):
                            with city_cols[col_idx]:
                                st.markdown(f"### {city}")
                                results = city_results[city]
                                
                                if not results:
                                    st.info(f"No results found for {city}")
                                else:
                                    for i, result in enumerate(results[:comparison_top_k], 1):
                                        with st.expander(f"#{i}: {result['title'][:60]}..." if len(result['title']) > 60 else f"#{i}: {result['title']}"):
                                            st.markdown(f"**Relevance:** {result['score']:.4f}")
                                            st.markdown(f"**Topic:** {result['topic']}")
                                            st.markdown(f"**Sentiment:** {result['sentiment']}")
                                            st.markdown(f"**Preview:** {result['preview']}")
                                            if result['url']:
                                                st.markdown(f"[🔗 View Source]({result['url']})")

# Tab 3: Causal Analysis
with tab3:
    st.header("🔗 Causal Graph Analysis")
    st.markdown("Explore causal relationships between events and articles")
    
    # City selection for causal analysis
    if not corpus.empty and "city" in corpus.columns:
        causal_cities = set()
        for c in corpus["city"].dropna():
            if c:
                from src.utils import normalize_city_name
                normalized = normalize_city_name(str(c))
                if normalized:
                    causal_cities.add(normalized)
        causal_city_options = ["All"] + sorted(causal_cities)
    else:
        causal_city_options = ["All"]
    
    selected_causal_city = st.selectbox("Select city for causal analysis", causal_city_options, key="causal_city")
    min_confidence = st.slider("Minimum confidence threshold", min_value=0.1, max_value=0.9, value=0.3, step=0.1, key="min_confidence")
    
    # Initialize session state for graph caching
    if 'causal_graph' not in st.session_state:
        st.session_state.causal_graph = None
    if 'causal_graph_city' not in st.session_state:
        st.session_state.causal_graph_city = None
    if 'causal_graph_confidence' not in st.session_state:
        st.session_state.causal_graph_confidence = None
    
    # Try to load pre-computed graph first
    from src.causal.causal_graph import CausalGraphBuilder
    
    graph, summary = CausalGraphBuilder.load_graph()
    
    if graph is None:
        st.info("💡 **Tip:** No pre-computed causal graph found. Run the full pipeline to pre-compute the graph for faster loading.")
        st.markdown("---")
        
        # Option to build on-demand (slower, but available)
        build_button = st.button("Build Causal Graph (On-Demand)", key="build_causal_graph")
        
        if build_button:
            try:
                # Filter corpus by city if selected
                analysis_corpus = corpus.copy()
                if selected_causal_city != "All":
                    analysis_corpus = analysis_corpus[
                        analysis_corpus["city"].str.contains(selected_causal_city, case=False, na=False)
                    ]
                
                if len(analysis_corpus) < 2:
                    st.warning(f"Not enough articles ({len(analysis_corpus)}) for causal analysis. Need at least 2 articles.")
                else:
                    with st.spinner("Building causal graph (this may take a while for large datasets)..."):
                        builder = CausalGraphBuilder()
                        graph = builder.build_graph(analysis_corpus, city=None, min_confidence=min_confidence)
                        
                        if graph and graph.number_of_nodes() > 0:
                            summary = builder.get_graph_summary()
                            st.session_state.causal_graph = graph
                            st.session_state.causal_graph_builder = builder
                        else:
                            st.warning("No causal relationships found. Try lowering the confidence threshold or selecting a different city.")
                            graph = None
            except Exception as e:
                st.error(f"Error building causal graph: {e}")
                import traceback
                st.code(traceback.format_exc())
                graph = None
    else:
        # Use pre-computed graph
        st.success("✅ Loaded pre-computed causal graph (fast!)")
        
        # Create a builder instance for visualization methods
        builder = CausalGraphBuilder()
        builder.graph = graph
        
        st.session_state.causal_graph = graph
        st.session_state.causal_graph_builder = builder
        
        # Display summary
        if summary:
            st.subheader("📊 Graph Summary")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Events", summary.get("num_events", 0))
            with col2:
                st.metric("Relationships", summary.get("num_relationships", 0))
            with col3:
                st.metric("Root Causes", summary.get("root_causes", 0))
            with col4:
                st.metric("Final Effects", summary.get("final_effects", 0))
    
    # Display visualization (using cached graph, won't rebuild on slider/checkbox changes)
    if st.session_state.causal_graph is not None and st.session_state.causal_graph.number_of_nodes() > 0:
        # Import here to ensure it's available
        from src.causal.causal_graph import CausalGraphBuilder
        import networkx as nx
        
        graph = st.session_state.causal_graph
        builder = st.session_state.causal_graph_builder
        
        st.markdown("---")
        st.subheader("🕸️ Interactive Graph Visualization")
        
        # Options for visualization
        viz_col1, viz_col2 = st.columns(2)
        with viz_col1:
            max_nodes = st.slider(
                "Max nodes to display",
                min_value=10,
                max_value=min(200, graph.number_of_nodes()),
                value=min(100, graph.number_of_nodes()),
                key="max_nodes_viz"
            )
        with viz_col2:
            show_high_confidence_only = st.checkbox(
                "Show high confidence only (>0.5)",
                value=False,
                key="high_conf_only"
            )
        
        # Generate visualization (this won't rebuild the graph, just filter it)
        try:
            # Create a copy to avoid modifying the cached graph
            # Use NetworkX copy method (faster and safer than deepcopy)
            viz_graph = graph.copy()
            
            # Filter graph if needed
            if show_high_confidence_only:
                edges_to_remove = [
                    (u, v) for u, v, d in viz_graph.edges(data=True)
                    if d.get("confidence", 0.5) <= 0.5
                ]
                if edges_to_remove:
                    viz_graph.remove_edges_from(edges_to_remove)
                # Remove isolated nodes
                isolated = list(nx.isolates(viz_graph))
                if isolated:
                    viz_graph.remove_nodes_from(isolated)
            
            # Limit nodes for display (select top nodes by degree)
            if viz_graph.number_of_nodes() > max_nodes:
                node_degrees = dict(viz_graph.degree())
                top_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
                nodes_to_keep = [node_id for node_id, _ in top_nodes]
                viz_graph = viz_graph.subgraph(nodes_to_keep).copy()
            
            if viz_graph.number_of_nodes() == 0:
                st.warning("No nodes remaining after filtering. Try adjusting the filters.")
            else:
                # Create temporary builder with filtered graph
                temp_builder = CausalGraphBuilder()
                temp_builder.graph = viz_graph
                
                html_viz = temp_builder.create_interactive_html(
                    max_nodes=max_nodes,
                    height="1000px",  # Increased height for better visibility
                    width="100%"
                )
                
                if html_viz and len(html_viz) > 1000:
                    # Display the HTML directly with larger height
                    # Match the HTML height exactly
                    st.components.v1.html(
                        html_viz,
                        height=1050,  # Slightly larger than HTML to ensure full display
                        scrolling=False  # Disable scrolling for better interaction
                    )
                    
                    st.caption("💡 **Tip:** Hover over nodes to see details. Click and drag to explore. Node size indicates number of connections. Edge color indicates confidence (red=high, orange=medium, gray=low).")
                    st.info(f"📊 Displaying {viz_graph.number_of_nodes()} nodes with {viz_graph.number_of_edges()} relationships.")
                else:
                    st.warning(f"Could not generate visualization HTML (length: {len(html_viz) if html_viz else 0}).")
                    st.info("You can still view the graph summary above and export it as GEXF.")
        except Exception as e:
            st.error(f"Error generating visualization: {str(e)}")
            import traceback
            with st.expander("Error details"):
                st.code(traceback.format_exc())
            st.info("You can still view the graph summary above and export it as GEXF.")
        
        except Exception as e:
            st.error(f"Error building causal graph: {e}")
            import traceback
            st.code(traceback.format_exc())
    
    # Show root causes and effects (only if graph exists)
    if st.session_state.causal_graph is not None and st.session_state.causal_graph.number_of_nodes() > 0:
        graph = st.session_state.causal_graph
        builder = st.session_state.causal_graph_builder
        
        # Show root causes
        root_causes = builder.get_root_causes()
        if root_causes:
            st.markdown("---")
            st.subheader("🌱 Root Causes")
            for i, cause_id in enumerate(root_causes[:10], 1):
                node_data = graph.nodes[cause_id]
                title = node_data.get("title", "Unknown")
                city = node_data.get("city", "Unknown")
                with st.expander(f"{i}. {title[:80]}..."):
                    st.write(f"**City:** {city}")
                    st.write(f"**Date:** {node_data.get('date', 'N/A')}")
                    if node_data.get('url'):
                        st.write(f"**URL:** {node_data['url']}")
                    
                    # Show outgoing edges with explanations
                    outgoing = list(graph.successors(cause_id))
                    if outgoing:
                        st.write(f"**Causes {len(outgoing)} effect(s):**")
                        for effect_id in outgoing[:5]:  # Show top 5
                            edge_data = graph[cause_id][effect_id]
                            confidence = edge_data.get('confidence', 0)
                            explanation = edge_data.get('explanation', 'No explanation')
                            effect_title = graph.nodes[effect_id].get('title', 'Unknown')[:60]
                            st.write(f"- → **{effect_title}...** (confidence: {confidence:.2f})")
                            st.caption(f"  *{explanation}*")
        
        # Show final effects
        effects = builder.get_effects()
        if effects:
            st.markdown("---")
            st.subheader("🎯 Final Effects")
            for i, effect_id in enumerate(effects[:10], 1):
                node_data = graph.nodes[effect_id]
                title = node_data.get("title", "Unknown")
                city = node_data.get("city", "Unknown")
                with st.expander(f"{i}. {title[:80]}..."):
                    st.write(f"**City:** {city}")
                    st.write(f"**Date:** {node_data.get('date', 'N/A')}")
                    if node_data.get('url'):
                        st.write(f"**URL:** {node_data['url']}")
                    
                    # Show incoming edges with explanations
                    incoming = list(graph.predecessors(effect_id))
                    if incoming:
                        st.write(f"**Caused by {len(incoming)} event(s):**")
                        for cause_id in incoming[:5]:  # Show top 5
                            edge_data = graph[cause_id][effect_id]
                            confidence = edge_data.get('confidence', 0)
                            explanation = edge_data.get('explanation', 'No explanation')
                            cause_title = graph.nodes[cause_id].get('title', 'Unknown')[:60]
                            st.write(f"- ← **{cause_title}...** (confidence: {confidence:.2f})")
                            st.caption(f"  *{explanation}*")
        
        # Show edge explanations section
        st.markdown("---")
        st.subheader("🔍 Edge Explanations")
        st.info("💡 **Tip:** Hover over edges in the graph above to see confidence explanations. Below are detailed explanations for all relationships.")
        
        # Get all edges with explanations
        all_edges = []
        for source, target, edge_data in graph.edges(data=True):
            confidence = edge_data.get('confidence', 0)
            explanation = edge_data.get('explanation', 'No explanation available')
            source_title = graph.nodes[source].get('title', 'Unknown')[:60]
            target_title = graph.nodes[target].get('title', 'Unknown')[:60]
            all_edges.append({
                'source': source_title,
                'target': target_title,
                'confidence': confidence,
                'explanation': explanation
            })
        
        # Sort by confidence
        all_edges.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Display in expandable sections
        for i, edge in enumerate(all_edges[:20], 1):  # Show top 20
            with st.expander(f"{i}. {edge['source']} → {edge['target']} (confidence: {edge['confidence']:.2f})"):
                st.write(f"**Confidence:** {edge['confidence']:.2f}")
                st.write(f"**Explanation:** {edge['explanation']}")
        
        # Export option
        st.markdown("---")
        if st.button("Export Graph (GEXF)", key="export_graph"):
            from pathlib import Path
            output_path = Path("data/causal_graph.gexf")
            builder.export_graph(str(output_path))
            st.success(f"Graph exported to {output_path}")
            st.info("You can open GEXF files in Gephi or other graph visualization tools")

st.markdown(
    "---\n"
    "Need to (re)build artifacts? Run `make run` for sample pipeline and `python -m src.search.index_search --build` to rebuild the index."
)

