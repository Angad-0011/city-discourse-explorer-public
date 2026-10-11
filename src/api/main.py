"""FastAPI application - placeholder."""
from fastapi import FastAPI
import uvicorn

from ..config import settings

app = FastAPI(
    title="City News Search Engine API",
    description="Search and analyze city-wise news and discussions",
    version="1.0.0"
)


@app.get("/")
def root():
    """Root endpoint."""
    return {
        "message": "City News Search Engine API",
        "version": "1.0.0",
        "status": "API endpoints coming soon"
    }


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


def start_server(host: str = None, port: int = None):
    """Start the API server."""
    if host is None:
        host = settings.api_host
    if port is None:
        port = settings.api_port
    
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start_server()

