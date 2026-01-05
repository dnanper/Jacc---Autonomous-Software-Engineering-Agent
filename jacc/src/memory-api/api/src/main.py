"""
Memory API - Main Entry Point

This module provides the FastAPI application and server startup logic.
It handles:
- Application initialization
- Database migrations (on startup)
- Health check endpoints
- MCP server integration (optional)
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .config import get_config, MemoryAPIConfig
from .migrations import run_migrations

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================
# Application State
# ============================================
class AppState:
    """Application state container."""
    
    config: MemoryAPIConfig | None = None
    db_pool = None  # asyncpg pool will be stored here
    embeddings_model = None
    reranker_model = None


app_state = AppState()


# ============================================
# Lifespan Management
# ============================================
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.
    
    Handles startup and shutdown tasks:
    - Load configuration
    - Run database migrations
    - Initialize models (embeddings, reranker)
    - Create database connection pool
    """
    logger.info("=" * 60)
    logger.info("Starting Memory API...")
    logger.info("=" * 60)
    
    # Load configuration
    config = get_config()
    app_state.config = config
    config.configure_logging()
    config.log_config()
    
    # Run database migrations (if enabled)
    if config.server.run_migrations:
        logger.info("Running database migrations...")
        try:
            run_migrations(config.database_url)
            logger.info("✓ Database migrations completed")
        except Exception as e:
            logger.error(f"✗ Database migration failed: {e}")
            raise
    else:
        logger.info("Skipping database migrations (disabled)")
    
    # Initialize embeddings model (if using local provider)
    if config.embeddings_provider == "local" and not config.lazy_reranker:
        logger.info(f"Loading embeddings model: {config.embeddings_local_model}...")
        try:
            from sentence_transformers import SentenceTransformer
            app_state.embeddings_model = SentenceTransformer(config.embeddings_local_model)
            logger.info("✓ Embeddings model loaded")
        except Exception as e:
            logger.warning(f"Failed to load embeddings model: {e}")
    
    # Initialize reranker model (if using local provider and not lazy loading)
    if config.reranker_provider == "local" and not config.lazy_reranker:
        logger.info(f"Loading reranker model: {config.reranker_local_model}...")
        try:
            from sentence_transformers import CrossEncoder
            app_state.reranker_model = CrossEncoder(config.reranker_local_model)
            logger.info("✓ Reranker model loaded")
        except Exception as e:
            logger.warning(f"Failed to load reranker model: {e}")
    
    # Create database connection pool
    logger.info("Creating database connection pool...")
    try:
        import asyncpg
        app_state.db_pool = await asyncpg.create_pool(
            dsn=config.database_url,
            min_size=config.database.pool_min_size,
            max_size=config.database.pool_max_size,
            command_timeout=config.database.command_timeout,
        )
        logger.info("✓ Database connection pool created")
    except Exception as e:
        logger.error(f"✗ Failed to create database pool: {e}")
        raise
    
    logger.info("=" * 60)
    logger.info(f"Memory API is ready at http://{config.host}:{config.port}")
    logger.info("=" * 60)
    
    yield  # Application runs here
    
    # Shutdown
    logger.info("Shutting down Memory API...")
    
    if app_state.db_pool:
        await app_state.db_pool.close()
        logger.info("✓ Database pool closed")
    
    logger.info("Memory API shutdown complete")


# ============================================
# FastAPI Application
# ============================================
app = FastAPI(
    title="Memory API",
    description="Memory System for AI Agents - Temporal, Semantic, and Entity Memory Architecture",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# Health Check Endpoints
# ============================================
@app.get("/health", tags=["Health"])
async def health_check():
    """
    Basic health check endpoint.
    
    Returns 200 if the server is running.
    Used by Docker health checks and load balancers.
    """
    return {"status": "healthy", "service": "memory-api"}


@app.get("/health/ready", tags=["Health"])
async def readiness_check():
    """
    Readiness check endpoint.
    
    Verifies that all critical services are ready:
    - Database connection pool is available
    - Embeddings model is loaded (if using local)
    
    Returns 503 if not ready.
    """
    issues = []
    
    # Check database pool
    if not app_state.db_pool:
        issues.append("Database pool not initialized")
    else:
        try:
            async with app_state.db_pool.acquire() as conn:
                await conn.execute("SELECT 1")
        except Exception as e:
            issues.append(f"Database connection failed: {e}")
    
    # Check embeddings model (if local)
    config = app_state.config
    if config and config.embeddings_provider == "local":
        if not app_state.embeddings_model and not config.lazy_reranker:
            issues.append("Embeddings model not loaded")
    
    if issues:
        raise HTTPException(status_code=503, detail={"status": "not_ready", "issues": issues})
    
    return {"status": "ready", "service": "memory-api"}


@app.get("/health/live", tags=["Health"])
async def liveness_check():
    """
    Liveness check endpoint.
    
    Simple check to verify the process is alive.
    Used by Kubernetes liveness probes.
    """
    return {"status": "alive", "service": "memory-api"}


# ============================================
# Info Endpoints
# ============================================
@app.get("/", tags=["Info"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Memory API",
        "version": "0.1.0",
        "description": "Memory System for AI Agents",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/info", tags=["Info"])
async def info():
    """
    Get API configuration info (non-sensitive).
    """
    config = app_state.config
    if not config:
        raise HTTPException(status_code=503, detail="Configuration not loaded")
    
    return {
        "version": "0.1.0",
        "llm_provider": config.llm_provider,
        "llm_model": config.llm_model,
        "embeddings_provider": config.embeddings_provider,
        "embeddings_model": config.embeddings_local_model,
        "reranker_provider": config.reranker_provider,
        "graph_retriever": config.graph_retriever,
        "mcp_enabled": config.mcp_enabled,
    }


# ============================================
# Bank Endpoints (Placeholder)
# ============================================
@app.get("/banks", tags=["Banks"])
async def list_banks():
    """List all memory banks."""
    # TODO: Implement actual bank listing
    return {"banks": [], "message": "Bank endpoints coming soon"}


@app.post("/banks/{bank_id}", tags=["Banks"])
async def create_bank(bank_id: str):
    """Create a new memory bank."""
    # TODO: Implement bank creation
    return {"bank_id": bank_id, "message": "Bank creation coming soon"}


# ============================================
# Memory Endpoints (Placeholder)
# ============================================
@app.post("/banks/{bank_id}/retain", tags=["Memory"])
async def retain(bank_id: str, text: str):
    """
    Store information in long-term memory.
    
    Extracts facts from text and stores them with embeddings.
    """
    # TODO: Implement retain functionality
    return {"bank_id": bank_id, "message": "Retain coming soon", "text_length": len(text)}


@app.post("/banks/{bank_id}/recall", tags=["Memory"])
async def recall(bank_id: str, query: str, limit: int = 10):
    """
    Search memories to retrieve relevant information.
    
    Uses semantic search and graph traversal.
    """
    # TODO: Implement recall functionality
    return {"bank_id": bank_id, "query": query, "results": [], "message": "Recall coming soon"}


# ============================================
# Main Entry Point
# ============================================
def main():
    """Main entry point for the Memory API server."""
    config = get_config()
    
    uvicorn.run(
        "src.main:app",
        host=config.host,
        port=config.port,
        log_level=config.log_level,
        reload=False,
    )


if __name__ == "__main__":
    main()
