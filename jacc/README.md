# JACC - Memory-Augmented Software Engineering Agent

> **Version 1.0** - Basic Hindsight Memory Integration

JACC (Just Another Code Companion) is an autonomous software engineering agent that leverages episodic memory to learn from past experiences. Built on top of [Hindsight](https://github.com/anthropics/hindsight) memory system, JACC can recall relevant experiences when solving new tasks and retain learnings from successful problem-solving sessions.

## 🎯 Key Features

- **LangGraph-based Agent Architecture**: Modular think-act-observe-decide loop
- **Memory Integration**: Recall past experiences before reasoning, retain learnings after actions
- **SWE-Bench Compatible**: Run evaluations on SWE-Bench Lite/Verified/Full datasets
- **Docker Environment**: Isolated execution environment for each task
- **Multi-Model Support**: Works with Gemini, OpenAI, Anthropic, and other LLM providers via LiteLLM

## 📁 Project Structure

```
jacc/
├── src/
│   ├── agent/                    # Main agent implementation
│   │   ├── config/               # Agent configuration
│   │   ├── environments/         # Execution environments (Docker, Local, Mock)
│   │   ├── experiments/          # Experiment scripts for evaluation
│   │   ├── memory/               # Memory integration layer
│   │   │   ├── base.py           # Abstract memory client interface
│   │   │   ├── direct.py         # Direct MemoryEngine integration
│   │   │   ├── config.py         # Memory configuration
│   │   │   ├── utils.py          # Recall/retain utilities
│   │   │   └── async_utils.py    # Async execution helpers
│   │   ├── models/               # LLM providers
│   │   ├── nodes/                # LangGraph nodes
│   │   │   ├── think.py          # LLM reasoning + memory recall
│   │   │   ├── act.py            # Command execution
│   │   │   ├── observe.py        # Output processing + memory retain
│   │   │   └── decide.py         # Workflow control
│   │   ├── graph.py              # LangGraph workflow definition
│   │   ├── state.py              # Agent state definition
│   │   └── run_swebench.py       # SWE-Bench runner
│   │
│   └── memory-api/               # Hindsight memory engine (submodule)
│       ├── api/src/engine/       # Core memory engine
│       └── docker-compose.dev.yml
│
└── config/                       # Project-level configuration
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker Desktop
- API key for your preferred LLM provider

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/jaccque.git
cd jaccque/jacc

# Install dependencies
pip install -e .
```

### Start Memory Database

```bash
cd src/memory-api
docker-compose -f docker-compose.dev.yml up -d
```

This starts a PostgreSQL database with pgvector for memory storage.

### Set API Keys

```powershell
# PowerShell
$env:GEMINI_API_KEY = "your-api-key"
$env:MEMORY_API_LLM_API_KEY = "your-api-key"  # For memory fact extraction
```

```bash
# Bash
export GEMINI_API_KEY="your-api-key"
export MEMORY_API_LLM_API_KEY="your-api-key"
```

### Run on SWE-Bench

```bash
cd src

# Run with memory enabled
python -m agent.run_swebench \
    --memory \
    --subset lite \
    --split dev \
    --slice "0:5" \
    -o ./output

# Run without memory (baseline)
python -m agent.run_swebench \
    --subset lite \
    --split dev \
    --slice "0:5" \
    -o ./output_baseline
```

## 🧠 Memory System

### How It Works

JACC integrates with the Hindsight memory system to:

1. **Recall** - Before each reasoning step, query memory for relevant past experiences
2. **Retain** - After significant actions (errors, fixes, task completion), store learnings

### Recall Flow

```
ThinkNode.__call__()
  └→ _recall_experience(state)
       ├→ build_recall_query(state)    # Build query from current context
       ├→ memory.recall(query)         # Semantic + BM25 + Graph retrieval
       └→ format_facts_for_prompt()    # Inject into LLM context
```

### Retain Triggers

| Context | Trigger Condition |
|---------|-------------------|
| `task_complete` | Agent submits solution |
| `error_encountered` | Command fails with error |
| `error_discovery` | File not found, permission denied |
| `test_success` | Tests pass |
| `git_operation` | Git commands executed |

### Memory Banks

Memories are organized into banks:
- `swe_agent` - Default bank for production runs
- `exp_{repo}` - Per-repo banks for experiments

## ⚙️ Configuration

### Agent Configuration

Edit `src/agent/config/default.yaml`:

```yaml
system_prompt: |
  You are a skilled software engineer...

step_limit: 30
cost_limit: 1.0
```

### Memory Configuration

```python
from agent.memory import MemoryConfig

config = MemoryConfig(
    bank_id="swe_agent",           # Memory bank ID
    database_url="...",            # PostgreSQL connection
    recall_budget=300,             # Max facts to consider
    recall_max_tokens=2048,        # Max tokens in recall result
    retain_enabled=True,           # Enable memory retention
)
```

## 📊 Experiments

### Analyze Dataset by Repository

```bash
python -m agent.experiments.repo_memory_exp --analyze --subset lite
```

### Run Memory vs No-Memory Comparison

```bash
# With memory
python -m agent.experiments.repo_memory_exp \
    --run \
    --memory \
    --repo django__django \
    --limit 10 \
    -o ./exp_memory

# Without memory (baseline)
python -m agent.experiments.repo_memory_exp \
    --run \
    --repo django__django \
    --limit 10 \
    -o ./exp_baseline
```

## 🔧 Development

### View Memory Database

```bash
cd src/agent
python db_explore.py
```

### Run Tests

```bash
cd src
pytest tests/
```

### Direct Memory Test

```bash
cd src/memory-api
python api/test_direct.py
```

## 📈 Roadmap

- [x] **v0.1** - Basic SWE Agent with LangGraph
- [x] **v1.0** - Basic Hindsight integration (recall + retain)
- [ ] **v1.1** - Reflect operation for experience consolidation
- [ ] **v2.0** - Building In-Repo Code Map
- [ ] **v2.1** - Update In-Repo Experience
- [ ] **v3.0** - Building Multi-Repo Memory

## 📝 License

MIT License

## 🙏 Acknowledgments

- [Hindsight](https://github.com/anthropics/hindsight) - Memory system
- [SWE-Bench](https://www.swebench.com/) - Evaluation benchmark
- [LangGraph](https://github.com/langchain-ai/langgraph) - Agent framework
- [mini-swe-agent](https://github.com/princeton-nlp/SWE-bench) - Inspiration for agent design
