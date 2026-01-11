# Memory API

Memory System for AI Agents - Temporal, Semantic, and Entity Memory Architecture.

## Overview

This API provides a comprehensive memory system designed for AI agents, featuring:

- **Temporal Memory**: Time-aware memory storage and retrieval
- **Semantic Memory**: Vector-based semantic search using embeddings
- **Entity Memory**: Structured entity extraction and relationship tracking

## Features

- PostgreSQL with pgvector for vector similarity search
- FastAPI-based REST and MCP interfaces
- Support for multiple LLM providers (OpenAI, Google Gemini, Ollama)
- Local embeddings with sentence-transformers

## Requirements

- Python 3.11+
- PostgreSQL with pgvector extension
- Docker (recommended)

## Quick Start

```bash
# Using Docker Compose
docker-compose up --build

# Or run locally
pip install -e .
uvicorn src.main:app --reload
```

## License

MIT License
