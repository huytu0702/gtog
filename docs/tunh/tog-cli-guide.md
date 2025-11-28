# GraphRAG ToG Search CLI Guide

This guide provides step-by-step instructions for running GraphRAG with the Tree-of-Graph (ToG) search method.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Configuration](#configuration)
- [Indexing Your Data](#indexing-your-data)
- [Running Queries](#running-queries)
- [Troubleshooting](#troubleshooting)

## Prerequisites

1. **Python 3.10-3.12** installed
2. **GraphRAG** installed with ToG modifications
3. **API Keys** configured:
   - Google Gemini API key (or OpenAI/Azure OpenAI)

## Configuration

### 1. Initialize Your Project

Create a project directory and initialize GraphRAG:

```bash
mkdir -p ./my-project/input
cd my-project
graphrag init --root .
```

This creates two files:
- `.env` - Environment variables for API keys
- `settings.yaml` - Pipeline configuration

### 2. Configure API Keys

Edit `.env` and add your API key:

```bash
GOOGLE_API_KEY=your_api_key_here
```

### 3. Configure Settings

Edit `settings.yaml` to configure models and ToG search parameters:

```yaml
models:
  default_chat_model:
    type: chat
    auth_type: api_key
    api_key: ${GOOGLE_API_KEY}
    model_provider: gemini
    model: gemini-2.5-flash-lite
    model_supports_json: true

  default_embedding_model:
    type: embedding
    auth_type: api_key
    api_key: ${GOOGLE_API_KEY}
    model_provider: gemini
    model: embedding-001

# Input configuration
input:
  type: file
  file_type: text
  base_dir: "input"
  encoding: utf-8

# Output configuration
output:
  type: file
  base_dir: "output"

# Cache configuration
cache:
  type: file
  base_dir: "cache"

# Graph extraction workflow
extract_graph:
  model_id: default_chat_model
  prompt: "prompts/index/extract_graph.py"
  entity_types: [organization, person, geo, event]
  max_gleanings: 1

# Community reports
community_reports:
  model_id: default_chat_model
  prompt: "prompts/index/community_report.py"
  max_length: 2000
  max_input_length: 8000

# Summarization
summarize_descriptions:
  model_id: default_chat_model
  prompt: "prompts/index/summarize_descriptions.py"
  max_length: 500
  max_input_length: 3000

# ToG (Tree-of-Graph) search configuration
tog_search:
  chat_model_id: default_chat_model
  embedding_model_id: default_embedding_model
  width: 3                      # Beam width for exploration
  depth: 3                      # Maximum exploration depth
  prune_strategy: llm          # Pruning strategy: llm or embedding
  num_retain_entity: 5         # Number of entities to retain per level
  temperature_exploration: 0.4  # Temperature for exploration
  temperature_reasoning: 0.0    # Temperature for final reasoning
  max_context_tokens: 8000
  max_exploration_paths: 10
```

### 4. Add Your Input Data

Place your text files in the `input` directory:

```bash
# Example: Add a sample document
echo "Your text content here..." > input/document.txt
```

## Indexing Your Data

Run the indexing pipeline to process your documents and build the knowledge graph:

```bash
graphrag index --root .
```

**Expected output:**
```
Starting pipeline with workflows: load_input_documents, create_base_text_units, ...
Starting workflow: load_input_documents
Workflow complete: load_input_documents
Starting workflow: create_base_text_units
...
Pipeline complete
```

**What this does:**
- Loads and chunks your documents
- Extracts entities and relationships
- Builds community structure
- Generates embeddings
- Creates output files in `./output/`:
  - `entities.parquet` - Extracted entities
  - `relationships.parquet` - Entity relationships
  - `communities.parquet` - Community structure
  - `community_reports.parquet` - Community summaries
  - `text_units.parquet` - Document chunks
  - `documents.parquet` - Document metadata

**Indexing time:** Depends on:
- Input data size
- Model speed
- Chunk size configuration
- Network latency

## Running Queries

GraphRAG supports multiple search methods. The ToG (Tree-of-Graph) method provides deep, reasoning-based exploration of the knowledge graph.

### ToG Search

Tree-of-Graph search uses iterative exploration to find relevant information:

```bash
graphrag query \
  --root . \
  --method tog \
  --query "What are the main topics in this dataset?"
```

**ToG Search Parameters:**

The ToG method uses these parameters from `settings.yaml`:
- `width`: Number of paths to explore in parallel (beam width)
- `depth`: Maximum exploration depth
- `prune_strategy`: How to score/filter paths (`llm` or `embedding`)
- `num_retain_entity`: Entities to keep at each level
- `temperature_exploration`: LLM creativity during exploration
- `temperature_reasoning`: LLM creativity for final answer

### Other Search Methods

**Global Search** - High-level questions about the entire dataset:
```bash
graphrag query \
  --root . \
  --method global \
  --query "What are the top themes in this dataset?"
```

**Local Search** - Specific questions about particular entities:
```bash
graphrag query \
  --root . \
  --method local \
  --query "Who is Alice and what are her relationships?"
```

**Drift Search** - Multi-hop reasoning across the graph:
```bash
graphrag query \
  --root . \
  --method drift \
  --query "How are organization A and person B connected?"
```

## Query Examples

### ToG Search Examples

1. **Topic Discovery:**
   ```bash
   graphrag query --root . --method tog \
     --query "What are the main topics discussed in these documents?"
   ```

2. **Relationship Analysis:**
   ```bash
   graphrag query --root . --method tog \
     --query "How do the key organizations interact with each other?"
   ```

3. **Deep Investigation:**
   ```bash
   graphrag query --root . --method tog \
     --query "What are the connections between project X and its stakeholders?"
   ```

4. **Comparative Analysis:**
   ```bash
   graphrag query --root . --method tog \
     --query "Compare the roles of Alice and Bob in the project"
   ```

## Troubleshooting

### Common Issues

#### 1. Missing Output Files

**Error:** `Could not find entities.parquet in storage!`

**Solution:** Run the indexing pipeline:
```bash
graphrag index --root .
```

#### 2. Missing Prompt Files

**Error:** `FileNotFoundError: prompts/extract_graph.txt`

**Solution:** Update `settings.yaml` to use correct prompt paths:
```yaml
extract_graph:
  prompt: "prompts/index/extract_graph.py"  # Not .txt
```

#### 3. API Key Issues

**Error:** `Authentication failed` or `Invalid API key`

**Solution:**
- Check `.env` file has correct API key
- Verify environment variable is set: `echo $GOOGLE_API_KEY`
- For Google Gemini, get API key from: https://aistudio.google.com/apikey

#### 4. Module Import Errors

**Error:** `module 'graphrag.api' has no attribute 'tog_search'`

**Solution:** Ensure `graphrag/api/__init__.py` exports ToG functions:
```python
from graphrag.api.query import (
    ...
    tog_search,
    tog_search_streaming,
)
```

#### 5. Empty or Poor Results

**Possible causes:**
- Input data too small or low quality
- Wrong search method for question type
- Need to adjust ToG parameters

**Solutions:**
- Verify input data is properly indexed
- Try different search methods
- Adjust ToG `width` and `depth` in settings
- Increase `max_gleanings` for better entity extraction

### Verification Steps

1. **Check indexing succeeded:**
   ```bash
   ls -lh output/*.parquet
   ```
   Should show: communities, community_reports, documents, entities, relationships, text_units

2. **Check configuration:**
   ```bash
   cat settings.yaml | grep -A5 tog_search
   ```

3. **Test with simple query:**
   ```bash
   graphrag query --root . --method global --query "Summarize the dataset"
   ```

## Performance Tips

1. **Use caching:** The `cache` directory stores LLM responses to avoid re-processing
2. **Start small:** Test with a small dataset first
3. **Choose appropriate models:**
   - Fast/cheap: `gemini-2.5-flash-lite`
   - Balanced: `gemini-2.0-flash`
   - High quality: `gemini-1.5-pro`
4. **Adjust chunk size:** Smaller chunks = more chunks = more API calls
5. **Tune ToG parameters:**
   - Lower `width` and `depth` = faster but less thorough
   - Higher values = more comprehensive but slower

## Advanced Usage

### Custom Prompts

Modify prompts in `graphrag/prompts/index/` to customize entity extraction.

### Multi-Index Search

Query across multiple indexed datasets simultaneously.

### Streaming Responses

Get real-time streaming output:
```bash
graphrag query --root . --method tog --streaming --query "Your question"
```

### Verbose Output

Debug with detailed logging:
```bash
graphrag query --root . --method tog --verbose --query "Your question"
```

## Next Steps

- Explore the [GraphRAG documentation](../get_started.md)
- Learn about [configuration options](../config/overview.md)
- Try different search methods and compare results
- Tune ToG parameters for your use case

## References

- [Gemini API Setup Guide](./gemini-setup.md)
- [GraphRAG Getting Started](../get_started.md)
- [Query Engine Overview](../query/overview.md)
