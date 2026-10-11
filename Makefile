.PHONY: bootstrap test run ui api full clean help

help:
	@echo "Available commands:"
	@echo "  make bootstrap  - Set up virtual environment and install dependencies"
	@echo "  make test       - Run test suite"
	@echo "  make run        - Run fast pipeline on sample data"
	@echo "  make ui         - Launch Streamlit UI"
	@echo "  make api        - Start FastAPI server"
	@echo "  make full       - Run full pipeline (may take time)"
	@echo "  make clean      - Clean generated files"

bootstrap:
	@echo "🔧 Setting up environment..."
	python -m venv .venv
	. .venv/bin/activate && pip install -U pip wheel
	. .venv/bin/activate && pip install -r requirements.txt
	python -m spacy download en_core_web_sm
	@echo "✅ Setup complete! Activate with: source .venv/bin/activate"

test:
	@echo "🧪 Running test suite..."
	pytest -q

run:
	@echo "⚡ Running fast pipeline on sample data..."
	FAST_MODE=1 python -m src.run_pipeline

ui:
	@echo "🌐 Launching Streamlit UI..."
	streamlit run src/ui/app_streamlit.py

api:
	@echo "🚀 Starting FastAPI server..."
	python -m src.api.main

full:
	@echo "🚀 Running full pipeline (may take time)..."
	FAST_MODE=0 python -m src.run_pipeline

clean:
	@echo "🧹 Cleaning generated files..."
	rm -rf data/processed/*.parquet
	rm -rf models/*.npy
	rm -rf index/*.faiss
	rm -rf logs/*.log
	@echo "✅ Clean complete"

