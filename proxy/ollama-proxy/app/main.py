import logging
import os
import uvicorn
from fastapi import FastAPI
from routes import register_routes
from contextlib import asynccontextmanager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Ollama Proxy...")
    logger.info(f"Ollama host: {app.state.config['ollama_host']}")
    os.makedirs(app.state.config['log_dir'], exist_ok=True)
    yield
    logger.info("Shutting down Ollama Proxy...")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Ollama Proxy",
        description="Proxy for Ollama with Prometheus metrics and logging",
        version="1.0.0",
        lifespan=lifespan
    )
    
    app.state.config = {
        'ollama_host': os.getenv('OLLAMA_HOST', 'http://ollama:11434'),
        'log_dir': '/app/logs'
    }
    
    register_routes(app)
    
    return app


app = create_app()


if __name__ == '__main__':
    uvicorn.run(
        "main:app",
        host='0.0.0.0',
        port=8010,
        log_level='info'
    )