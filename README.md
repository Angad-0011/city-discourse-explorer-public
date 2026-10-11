# City Discourse Explorer

City Discourse Explorer is an NLP and retrieval pipeline for collecting, analyzing, and exploring city-specific public discourse across Reddit and news sources. It combines scraping, preprocessing, topic modeling, sentiment analysis, vector search, and lightweight interfaces for exploration.

> **Source-only public snapshot.** This repository deliberately excludes scraped posts and articles, sample corpora of uncertain provenance, derived model and index artifacts, logs, and the PDF from the private project. It has a fresh Git history. Bring data that you are authorized to collect and use; no dataset or API credentials are distributed here.

## What It Does

- Collects city-related data from Reddit and curated news sources
- Normalizes city names and handles common variants such as `Bombay` to `Mumbai` and `NCR` to `Delhi`
- Builds embeddings and FAISS indexes for semantic search
- Runs NER, sentiment analysis, and topic modeling over the corpus
- Supports cross-city comparison and causal graph exploration in a Streamlit UI
- Includes a small FastAPI stub for API-oriented extension

## Main Components

- `src/scrapers/`: Reddit and news collection
- `src/processors/`: preprocessing, embeddings, NER, sentiment, and topic modeling
- `src/search/`: FAISS-based search index build and query logic
- `src/causal/`: event extraction and causal graph utilities
- `src/ui/`: Streamlit application
- `src/api/`: FastAPI placeholder app
- `config/`: scrape settings, city registry, feature flags
- `tests/`: city normalization and detection tests

## Quickstart

### Requirements

- Python 3.8+
- `pip`
- internet access for live scraping and model downloads

### Setup

```bash
git clone https://github.com/Angad-0011/city-discourse-explorer-public.git
cd city-discourse-explorer-public
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Or use the bundled target:

```bash
make bootstrap
```

## Common Commands

```bash
make test   # run tests
make run    # requires your own compatible data/sample/*.parquet files
make ui     # launch Streamlit UI
make api    # start FastAPI app
make full   # run full scraping and processing pipeline
```

## Running The Project

### Fast mode

The code's fast mode reads `data/sample/sample_news.parquet` and `data/sample/sample_reddit.parquet`. Those files are **not included** in this public copy. Supply your own compatible, publishable sample data before running:

```bash
make run
```

### Full pipeline

Run live scraping, preprocessing, embeddings, NLP passes, indexing, and causal graph generation only after configuring your own credentials and checking the terms and rights applicable to your data sources:

```bash
make full
```

You can also run the entrypoint directly:

```bash
FAST_MODE=0 python -m src.run_pipeline
```

## Configuration

The main configuration lives in `config/config.yaml`.

Key areas you will probably change first:

- scrape targets and limits
- allowed news domains
- embedding and sentiment models
- topic modeling method
- FAISS indexing mode
- feature flags for BERTopic, causal analysis, Streamlit, and FastAPI

For live Reddit scraping, add credentials in a local `.env` file:

```env
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=your_app_name/1.0
```

## Interfaces

### Streamlit UI

```bash
make ui
```

The Streamlit app supports:

- semantic search over the processed corpus
- sentiment and topic filters
- cross-city comparison
- topic and corpus-level summary views
- causal graph exploration from precomputed artifacts

### FastAPI

```bash
make api
```

The FastAPI app currently provides a basic skeleton with root and health endpoints. It is a starting point for turning the project into a service, not yet a full search API.

## Data And Outputs

Generated directories (created locally by the pipeline and ignored by Git):

- `data/raw/`: raw scraped data and fetch logs
- `data/processed/`: processed corpora and derived parquet files
- `models/`: BERTopic artifacts, LDA artifacts, and document mappings
- `index/`: FAISS index metadata and index files
- `logs/`: pipeline and scraping logs

## Testing

The current test suite focuses on city normalization and detection behavior:

```bash
make test
```

## Notes

- `FAST_MODE=1` is the quickest way to validate the pipeline locally.
- The public repository contains no scraped or processed datasets, trained models, vector indexes, logs, or PDF companion material.
- The Streamlit app expects processed corpus files to exist; run the pipeline first if the UI opens without data.

## Project Status

This is a source snapshot for local experimentation after supplying your own data. The documentation and UI are ahead of the API layer, which is still intentionally minimal. No open-source license has been granted; contact the maintainer about reuse rights.
