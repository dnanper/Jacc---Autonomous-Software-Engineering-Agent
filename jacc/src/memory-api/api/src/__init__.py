"""
Memory API - Memory System for AI Agents.

This package provides a memory system with:
- Temporal + Semantic + Entity Memory Architecture
- PostgreSQL with pgvector for vector storage
- Multi-strategy retrieval (TEMPR)
- LLM-powered fact extraction and reasoning
"""

from .config import (
    MemoryAPIConfig,
    get_config,
    EMBEDDING_DIMENSION,
    # Sub-configs
    DatabaseConfig,
    LLMConfig,
    EmbeddingsConfig,
    RerankerConfig,
    ServerConfig,
    RecallConfig,
    OptimizationConfig,
    # Backward compatibility
    HindsightConfig,
)
from .models import (
    Base,
    Bank,
    Document,
    MemoryUnit,
    Entity,
    UnitEntity,
    EntityCooccurrence,
    MemoryLink,
    RequestContext,
)

__all__ = [
    # Config
    "MemoryAPIConfig",
    "HindsightConfig",
    "get_config",
    "EMBEDDING_DIMENSION",
    "DatabaseConfig",
    "LLMConfig",
    "EmbeddingsConfig",
    "RerankerConfig",
    "ServerConfig",
    "RecallConfig",
    "OptimizationConfig",
    # Models
    "Base",
    "Bank",
    "Document",
    "MemoryUnit",
    "Entity",
    "UnitEntity",
    "EntityCooccurrence",
    "MemoryLink",
    "RequestContext",
]

__version__ = "0.1.0"
