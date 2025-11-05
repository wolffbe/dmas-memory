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
    logger.info("Starting OpenAI Proxy...")
    os.makedirs(app.state.config['log_dir'], exist_ok=True)
    yield
    logger.info("Shutting down OpenAI Proxy...")


def create_app() -> FastAPI:
    app = FastAPI(
        title="OpenAI Proxy",
        description="Proxy for OpenAI with Prometheus metrics and logging",
        version="1.0.0",
        lifespan=lifespan
    )
    
    app.state.config = {
        'openai_base_url': 'https://api.openai.com',
        'log_dir': '/app/logs'
    }
    
    register_routes(app)
    
    return app


app = create_app()


if __name__ == '__main__':
    uvicorn.run(
        "main:app",
        host='0.0.0.0',
        port=8011,
        log_level='info'
    )