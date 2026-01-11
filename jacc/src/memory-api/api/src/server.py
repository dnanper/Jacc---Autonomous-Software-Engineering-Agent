"""
FastAPI server for Memory API.

This module provides the ASGI app for uvicorn import string usage:
    uvicorn src.server:app
"""

import os
import warnings

# Filter deprecation warnings from third-party libraries
warnings.filterwarnings("ignore", message="websockets.legacy is deprecated")
warnings.filterwarnings("ignore", message="websockets.server.WebSocketServerProtocol is deprecated")

from src import MemoryEngine
from src.api import create_app
from src.config import get_config

# Disable tokenizers parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load configuration and configure logging
config = get_config()
config.configure_logging()

# Create app at module level (required for uvicorn import string)
# MemoryEngine reads configuration from environment variables automatically
_memory = MemoryEngine()

# Create unified app with both HTTP and optionally MCP
app = create_app(memory=_memory, http_api_enabled=True, mcp_api_enabled=config.mcp_enabled, mcp_mount_path="/mcp")

if __name__ == "__main__":
    # When run directly, start uvicorn
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=config.port)

