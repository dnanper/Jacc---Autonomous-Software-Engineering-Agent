"""
Centralized configuration for Memory API.

All environment variables and their defaults are defined here.
Uses Pydantic for validation and type safety.

Based on hindsight-api/config.py but adapted for 2-container architecture
(separate PostgreSQL container instead of embedded pg0).
"""

import logging
import os
from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

# ============================================
# Environment variable names
# ============================================
# Using MEMORY_API_ prefix for this project
# Change to HINDSIGHT_API_ if you want compatibility with original

ENV_DATABASE_URL = "MEMORY_API_DATABASE_URL"
ENV_LLM_PROVIDER = "MEMORY_API_LLM_PROVIDER"
ENV_LLM_API_KEY = "MEMORY_API_LLM_API_KEY"
ENV_LLM_MODEL = "MEMORY_API_LLM_MODEL"
ENV_LLM_BASE_URL = "MEMORY_API_LLM_BASE_URL"
ENV_LLM_MAX_CONCURRENT = "MEMORY_API_LLM_MAX_CONCURRENT"
ENV_LLM_TIMEOUT = "MEMORY_API_LLM_TIMEOUT"

ENV_EMBEDDINGS_PROVIDER = "MEMORY_API_EMBEDDINGS_PROVIDER"
ENV_EMBEDDINGS_LOCAL_MODEL = "MEMORY_API_EMBEDDINGS_LOCAL_MODEL"
ENV_EMBEDDINGS_TEI_URL = "MEMORY_API_EMBEDDINGS_TEI_URL"

ENV_RERANKER_PROVIDER = "MEMORY_API_RERANKER_PROVIDER"
ENV_RERANKER_LOCAL_MODEL = "MEMORY_API_RERANKER_LOCAL_MODEL"
ENV_RERANKER_TEI_URL = "MEMORY_API_RERANKER_TEI_URL"

ENV_HOST = "MEMORY_API_HOST"
ENV_PORT = "MEMORY_API_PORT"
ENV_LOG_LEVEL = "MEMORY_API_LOG_LEVEL"
ENV_MCP_ENABLED = "MEMORY_API_MCP_ENABLED"
ENV_GRAPH_RETRIEVER = "MEMORY_API_GRAPH_RETRIEVER"

# Database pool settings
ENV_DB_POOL_MIN = "MEMORY_API_DB_POOL_MIN"
ENV_DB_POOL_MAX = "MEMORY_API_DB_POOL_MAX"

# Optimization flags
ENV_SKIP_LLM_VERIFICATION = "MEMORY_API_SKIP_LLM_VERIFICATION"
ENV_LAZY_RERANKER = "MEMORY_API_LAZY_RERANKER"
ENV_RUN_MIGRATIONS = "MEMORY_API_RUN_MIGRATIONS"

# ============================================
# Default values
# ============================================
# Note: Changed from "pg0" to actual PostgreSQL URL for 2-container setup
DEFAULT_DATABASE_URL = "postgresql://memory:memory_secret@localhost:5432/memory_db"
DEFAULT_LLM_PROVIDER = "openai"
DEFAULT_LLM_MODEL = "gemini-2.5-flash"
DEFAULT_LLM_MAX_CONCURRENT = 32
DEFAULT_LLM_TIMEOUT = 120.0

DEFAULT_EMBEDDINGS_PROVIDER = "local"
DEFAULT_EMBEDDINGS_LOCAL_MODEL = "BAAI/bge-small-en-v1.5"

DEFAULT_RERANKER_PROVIDER = "local"
DEFAULT_RERANKER_LOCAL_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8888
DEFAULT_LOG_LEVEL = "info"
DEFAULT_MCP_ENABLED = True
DEFAULT_GRAPH_RETRIEVER = "bfs"  # Options: "bfs", "mpfp"

DEFAULT_DB_POOL_MIN = 5
DEFAULT_DB_POOL_MAX = 20

# Required embedding dimension for database schema
EMBEDDING_DIMENSION = 384

# ============================================
# Pydantic Configuration Models
# ============================================

class DatabaseConfig(BaseModel):
    """Database configuration."""
    
    url: str = Field(
        default_factory=lambda: os.getenv(ENV_DATABASE_URL, DEFAULT_DATABASE_URL),
        description="PostgreSQL connection URL"
    )
    pool_min_size: int = Field(
        default_factory=lambda: int(os.getenv(ENV_DB_POOL_MIN, str(DEFAULT_DB_POOL_MIN))),
        ge=1,
        le=100,
        description="Minimum connection pool size"
    )
    pool_max_size: int = Field(
        default_factory=lambda: int(os.getenv(ENV_DB_POOL_MAX, str(DEFAULT_DB_POOL_MAX))),
        ge=1,
        le=200,
        description="Maximum connection pool size"
    )
    command_timeout: int = Field(default=60, description="SQL command timeout in seconds")
    
    @field_validator("pool_max_size")
    @classmethod
    def validate_pool_size(cls, v, info):
        """Ensure pool_max_size >= pool_min_size."""
        if hasattr(info, 'data') and 'pool_min_size' in info.data:
            if v < info.data['pool_min_size']:
                raise ValueError("pool_max_size must be >= pool_min_size")
        return v


class LLMConfig(BaseModel):
    """LLM provider configuration."""
    
    provider: str = Field(
        default_factory=lambda: os.getenv(ENV_LLM_PROVIDER, DEFAULT_LLM_PROVIDER),
        description="LLM provider: openai, groq, ollama, gemini"
    )
    api_key: str | None = Field(
        default_factory=lambda: os.getenv(ENV_LLM_API_KEY),
        description="API key for the LLM provider"
    )
    model: str = Field(
        default_factory=lambda: os.getenv(ENV_LLM_MODEL, DEFAULT_LLM_MODEL),
        description="Model name to use"
    )
    base_url: str | None = Field(
        default_factory=lambda: os.getenv(ENV_LLM_BASE_URL) or None,
        description="Custom base URL for the LLM API"
    )
    
    def get_base_url(self) -> str:
        """Get the LLM base URL, with provider-specific defaults."""
        if self.base_url:
            return self.base_url
        
        provider = self.provider.lower()
        if provider == "groq":
            return "https://api.groq.com/openai/v1"
        elif provider == "ollama":
            return "http://localhost:11434/v1"
        else:
            return ""
    
    def validate_api_key(self) -> None:
        """Validate that API key is set for providers that require it."""
        if self.provider.lower() not in ("ollama",) and not self.api_key:
            raise ValueError(f"API key is required for LLM provider: {self.provider}")


class EmbeddingsConfig(BaseModel):
    """Embeddings configuration."""
    
    provider: Literal["local", "tei"] = Field(
        default_factory=lambda: os.getenv(ENV_EMBEDDINGS_PROVIDER, DEFAULT_EMBEDDINGS_PROVIDER),
        description="Embeddings provider: local (SentenceTransformers) or tei (TEI server)"
    )
    local_model: str = Field(
        default_factory=lambda: os.getenv(ENV_EMBEDDINGS_LOCAL_MODEL, DEFAULT_EMBEDDINGS_LOCAL_MODEL),
        description="Model name for local embeddings"
    )
    tei_url: str | None = Field(
        default_factory=lambda: os.getenv(ENV_EMBEDDINGS_TEI_URL),
        description="TEI server URL (required if provider is 'tei')"
    )
    dimension: int = Field(default=EMBEDDING_DIMENSION, description="Embedding dimension (must match DB schema)")
    
    def validate_tei_url(self) -> None:
        """Validate TEI URL is set when provider is 'tei'."""
        if self.provider == "tei" and not self.tei_url:
            raise ValueError(f"{ENV_EMBEDDINGS_TEI_URL} is required when provider is 'tei'")


class RerankerConfig(BaseModel):
    """Reranker/Cross-encoder configuration."""
    
    provider: Literal["local", "tei"] = Field(
        default_factory=lambda: os.getenv(ENV_RERANKER_PROVIDER, DEFAULT_RERANKER_PROVIDER),
        description="Reranker provider: local or tei"
    )
    local_model: str = Field(
        default_factory=lambda: os.getenv(ENV_RERANKER_LOCAL_MODEL, DEFAULT_RERANKER_LOCAL_MODEL),
        description="Model name for local reranker"
    )
    tei_url: str | None = Field(
        default_factory=lambda: os.getenv(ENV_RERANKER_TEI_URL),
        description="TEI server URL for reranking"
    )
    
    def validate_tei_url(self) -> None:
        """Validate TEI URL is set when provider is 'tei'."""
        if self.provider == "tei" and not self.tei_url:
            raise ValueError(f"{ENV_RERANKER_TEI_URL} is required when provider is 'tei'")


class ServerConfig(BaseModel):
    """Server configuration."""
    
    host: str = Field(
        default_factory=lambda: os.getenv(ENV_HOST, DEFAULT_HOST),
        description="Host to bind to"
    )
    port: int = Field(
        default_factory=lambda: int(os.getenv(ENV_PORT, str(DEFAULT_PORT))),
        ge=1,
        le=65535,
        description="Port to bind to"
    )
    log_level: str = Field(
        default_factory=lambda: os.getenv(ENV_LOG_LEVEL, DEFAULT_LOG_LEVEL),
        description="Logging level: debug, info, warning, error, critical"
    )
    mcp_enabled: bool = Field(
        default_factory=lambda: os.getenv(ENV_MCP_ENABLED, str(DEFAULT_MCP_ENABLED)).lower() == "true",
        description="Enable MCP server"
    )
    run_migrations: bool = Field(
        default_factory=lambda: os.getenv(ENV_RUN_MIGRATIONS, "true").lower() == "true",
        description="Run database migrations on startup"
    )


class RecallConfig(BaseModel):
    """Recall/search configuration."""
    
    graph_retriever: Literal["bfs", "mpfp"] = Field(
        default_factory=lambda: os.getenv(ENV_GRAPH_RETRIEVER, DEFAULT_GRAPH_RETRIEVER),
        description="Graph retrieval algorithm: bfs or mpfp"
    )

# New
class OptimizationConfig(BaseModel):
    """Optimization flags."""
    
    skip_llm_verification: bool = Field(
        default_factory=lambda: os.getenv(ENV_SKIP_LLM_VERIFICATION, "false").lower() == "true",
        description="Skip LLM connection verification during startup"
    )
    lazy_reranker: bool = Field(
        default_factory=lambda: os.getenv(ENV_LAZY_RERANKER, "false").lower() == "true",
        description="Delay reranker initialization until first use"
    )


class MemoryAPIConfig(BaseModel):
    """
    Main configuration container for Memory API.
    
    Combines all sub-configurations and provides helper methods.
    This is the Pydantic equivalent of HindsightConfig from hindsight-api.
    """
    
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    recall: RecallConfig = Field(default_factory=RecallConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    
    # Convenience aliases matching hindsight-api style
    @property
    def database_url(self) -> str:
        return self.database.url
    
    @property
    def llm_provider(self) -> str:
        return self.llm.provider
    
    @property
    def llm_api_key(self) -> str | None:
        return self.llm.api_key
    
    @property
    def llm_model(self) -> str:
        return self.llm.model
    
    @property
    def llm_base_url(self) -> str | None:
        return self.llm.base_url
    
    @property
    def embeddings_provider(self) -> str:
        return self.embeddings.provider
    
    @property
    def embeddings_local_model(self) -> str:
        return self.embeddings.local_model
    
    @property
    def embeddings_tei_url(self) -> str | None:
        return self.embeddings.tei_url
    
    @property
    def reranker_provider(self) -> str:
        return self.reranker.provider
    
    @property
    def reranker_local_model(self) -> str:
        return self.reranker.local_model
    
    @property
    def reranker_tei_url(self) -> str | None:
        return self.reranker.tei_url
    
    @property
    def host(self) -> str:
        return self.server.host
    
    @property
    def port(self) -> int:
        return self.server.port
    
    @property
    def log_level(self) -> str:
        return self.server.log_level
    
    @property
    def mcp_enabled(self) -> bool:
        return self.server.mcp_enabled
    
    @property
    def graph_retriever(self) -> str:
        return self.recall.graph_retriever
    
    @property
    def skip_llm_verification(self) -> bool:
        return self.optimization.skip_llm_verification
    
    @property
    def lazy_reranker(self) -> bool:
        return self.optimization.lazy_reranker
    
    def get_llm_base_url(self) -> str:
        """Get the LLM base URL, with provider-specific defaults."""
        return self.llm.get_base_url()
    
    def get_python_log_level(self) -> int:
        """Get the Python logging level from the configured log level string."""
        log_level_map = {
            "critical": logging.CRITICAL,
            "error": logging.ERROR,
            "warning": logging.WARNING,
            "info": logging.INFO,
            "debug": logging.DEBUG,
            "trace": logging.DEBUG,  # Python doesn't have TRACE, use DEBUG
        }
        return log_level_map.get(self.log_level.lower(), logging.INFO)
    
    def configure_logging(self) -> None:
        """Configure Python logging based on the log level."""
        logging.basicConfig(
            level=self.get_python_log_level(),
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            force=True,  # Override any existing configuration
        )
    
    def log_config(self) -> None:
        """Log the current configuration (without sensitive values)."""
        # Hide password in database URL
        db_display = self.database_url
        if "@" in db_display:
            # postgresql://user:pass@host:port/db -> postgresql://...@host:port/db
            parts = db_display.split("@")
            db_display = f"{parts[0].split(':')[0]}:***@{parts[1]}"
        
        logger.info(f"Database: {db_display}")
        logger.info(f"LLM: provider={self.llm_provider}, model={self.llm_model}")
        logger.info(f"Embeddings: provider={self.embeddings_provider}, model={self.embeddings_local_model}")
        logger.info(f"Reranker: provider={self.reranker_provider}")
        logger.info(f"Graph retriever: {self.graph_retriever}")
        logger.info(f"Server: {self.host}:{self.port}")
    
    def validate_all(self) -> None:
        """Validate all configuration that requires cross-field checks."""
        self.llm.validate_api_key()
        self.embeddings.validate_tei_url()
        self.reranker.validate_tei_url()
    
    @classmethod
    def from_env(cls) -> "MemoryAPIConfig":
        """Create configuration from environment variables."""
        return cls()


# ============================================
# Global configuration accessor
# ============================================

@lru_cache()
def get_config() -> MemoryAPIConfig:
    """
    Get the current configuration from environment variables.
    
    Uses LRU cache to avoid re-parsing environment on every call.
    Call get_config.cache_clear() if you need to reload config.
    """
    return MemoryAPIConfig.from_env()


# Backward compatibility alias (matches hindsight-api naming)
HindsightConfig = MemoryAPIConfig
