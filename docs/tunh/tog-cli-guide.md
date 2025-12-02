# GraphRAG ToG Search CLI Guide

This guide provides step-by-step instructions for running GraphRAG with the Tree-of-Graph (ToG) search method.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Detailed Configuration](#detailed-configuration)
- [Indexing Your Data](#indexing-your-data)
- [Running Queries](#running-queries)
- [Search Method Comparison](#search-method-comparison)
- [Troubleshooting](#troubleshooting)

## Prerequisites

1. **Python 3.10-3.12** installed
2. **GraphRAG** installed with ToG modifications
3. **API Keys** configured:
   - OpenAI API key (or Azure OpenAI)

## Quick Start

Follow these steps to get ToG search working in minutes:

### 1. Create and Initialize Project

```bash
# Create new project directory
mkdir my-graphrag-project
cd my-graphrag-project

# Create input directory
mkdir input

# Initialize GraphRAG
graphrag init --root .
```

### 2. Add Sample Data

```bash
# Create a sample document
echo "This is a sample document for testing GraphRAG search methods. It contains information about different search techniques including global search, local search, drift search, and ToG search." > input/sample.txt
```

### 3. Configure API Key

Edit the generated `.env` file to include your API key:

```bash
GRAPHRAG_API_KEY=your_openai_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

### 4. Update Settings for ToG

Replace the contents of `settings.yaml` with a working configuration:

```yaml
models:
  default_chat_model:
    type: chat
    auth_type: api_key
    api_key: ${GRAPHRAG_API_KEY}
    model_provider: openai
    model: gpt-4o-mini
    model_supports_json: true

  default_embedding_model:
    type: embedding
    auth_type: api_key
    api_key: ${GRAPHRAG_API_KEY}
    model_provider: openai
    model: text-embedding-3-small

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

# Chunking configuration
chunks:
  size: 1200
  overlap: 100
  group_by_columns: []

# Graph extraction workflow
extract_graph:
  model_id: default_chat_model
  prompt: "prompts/extract_graph.txt"
  entity_types: [organization, person, geo, event]
  max_gleanings: 1

# Text embedding workflow
embed_text:
  model_id: default_embedding_model
  batch_size: 16
  batch_max_tokens: 8000

# Community reports
community_reports:
  model_id: default_chat_model
  graph_prompt: "prompts/community_report_graph.txt"
  text_prompt: "prompts/community_report_text.txt"
  max_length: 2000
  max_input_length: 8000

# Summarization
summarize_descriptions:
  model_id: default_chat_model
  prompt: "prompts/summarize_descriptions.txt"
  max_length: 500

# Query configurations
local_search:
  chat_model_id: default_chat_model
  embedding_model_id: default_embedding_model
  prompt: "prompts/local_search_system_prompt.txt"

global_search:
  chat_model_id: default_chat_model
  map_prompt: "prompts/global_search_map_system_prompt.txt"
  reduce_prompt: "prompts/global_search_reduce_system_prompt.txt"
  knowledge_prompt: "prompts/global_search_knowledge_system_prompt.txt"

# ToG (Think-on-Graph) search configuration
tog_search:
  chat_model_id: default_chat_model
  embedding_model_id: default_embedding_model
  width: 3
  depth: 3
  prune_strategy: llm
  num_retain_entity: 5
  temperature_exploration: 0.4
  temperature_reasoning: 0.0
  max_context_tokens: 8000
  max_exploration_paths: 10

drift_search:
  chat_model_id: default_chat_model
  embedding_model_id: default_embedding_model
  prompt: "prompts/drift_search_system_prompt.txt"
  reduce_prompt: "prompts/drift_search_reduce_prompt.txt"

basic_search:
  chat_model_id: default_chat_model
  embedding_model_id: default_embedding_model
  prompt: "prompts/basic_search_system_prompt.txt"
```

### 5. Index Your Data

```bash
graphrag index --root .
```

**Expected output:**
```
Starting pipeline with workflows: load_input_documents, create_base_text_units, create_final_documents, extract_graph, finalize_graph, extract_covariates, create_communities, create_final_text_units, create_community_reports, generate_text_embeddings
Starting workflow: load_input_documents
Workflow complete: load_input_documents
...
Pipeline complete
```

### 6. Run All Search Methods

```bash
# ToG Search - Deep, reasoning-based exploration
graphrag query --root . --method tog --query "How are the different search methods connected?"

# Global Search - High-level overview
graphrag query --root . --method global --query "What are the main topics in this dataset?"

# Local Search - Entity-specific information
graphrag query --root . --method local --query "What search techniques are mentioned in the documents?"

# Drift Search - Multi-hop reasoning
graphrag query --root . --method drift --query "What are the relationships between Global Search and Local Search?"
```

## Detailed Configuration

### ToG Search Parameters

The ToG method uses these key parameters in `settings.yaml`:

- `width`: Number of paths to explore in parallel (beam width) - **Recommended: 3**
- `depth`: Maximum exploration depth - **Recommended: 3**
- `prune_strategy`: How to score/filter paths (`llm` or `embedding`) - **Recommended: llm**
- `num_retain_entity`: Entities to keep at each level - **Recommended: 5**
- `temperature_exploration`: LLM creativity during exploration - **Recommended: 0.4**
- `temperature_reasoning`: LLM creativity for final answer - **Recommended: 0.0**
- `max_exploration_paths`: Maximum number of paths to explore - **Recommended: 10**

### Model Configuration

For best results with ToG search:

```yaml
models:
  default_chat_model:
    model: gpt-4o-mini  # Best balance of quality and speed
    # Alternative: gpt-4-turbo-preview for higher quality
  default_embedding_model:
    model: text-embedding-3-small
```

## Indexing Your Data

### What Indexing Does

The indexing pipeline processes your documents and builds a knowledge graph:

1. **Document Loading**: Loads and chunks your documents
2. **Entity Extraction**: Identifies entities (organizations, people, locations, events)
3. **Relationship Building**: Creates relationships between entities
4. **Community Detection**: Groups related entities into communities
5. **Embedding Generation**: Creates vector embeddings for semantic search
6. **Report Generation**: Creates summaries for each community

### Output Files

Successful indexing creates these files in `./output/`:
- `entities.parquet` - Extracted entities
- `relationships.parquet` - Entity relationships
- `communities.parquet` - Community structure
- `community_reports.parquet` - Community summaries
- `text_units.parquet` - Document chunks
- `documents.parquet` - Document metadata
- `stats.json` - Indexing statistics

### Indexing Time

Depends on:
- Input data size and complexity
- Model speed and API rate limits
- Chunk size configuration (smaller = more chunks = slower)
- Network latency

**Typical times**:
- Small document (< 1K words): 1-2 minutes
- Medium document (1K-10K words): 2-5 minutes
- Large document (> 10K words): 5+ minutes

## Running Queries

### ToG Search

**Best for**: Complex reasoning, relationship analysis, deep exploration

```bash
graphrag query \
  --root . \
  --method tog \
  --query "How do different entities influence each other in this dataset?"
```

**Characteristics**:
- Iterative graph exploration
- Evidence-based reasoning
- Relationship-focused analysis
- Detailed, structured answers

### Global Search

**Best for**: High-level questions, topic discovery, dataset overview

```bash
graphrag query \
  --root . \
  --method global \
  --query "What are the main themes in this dataset?"
```

**Characteristics**:
- Broad overview across entire dataset
- Community-level analysis
- Topic summarization
- Comprehensive but less detailed

### Local Search

**Best for**: Specific entities, detailed information, targeted questions

```bash
graphrag query \
  --root . \
  --method local \
  --query "What do we know about [specific entity]?"
```

**Characteristics**:
- Entity-focused results
- Direct evidence from source documents
- Precise, factual information
- Fast and efficient

### Drift Search

**Best for**: Multi-hop reasoning, entity relationships, adaptive analysis

```bash
graphrag query \
  --root . \
  --method drift \
  --query "How are [entity A] and [entity B] connected?"
```

**Characteristics**:
- Adaptive exploration
- Relationship traversal
- Context-aware results
- Progressive refinement

## Search Method Comparison

| Method | Best For | Response Style | Speed | Detail Level |
|--------|----------|----------------|-------|--------------|
| **ToG** | Complex reasoning, relationships | Structured, evidence-based | Medium | Very High |
| **Global** | Dataset overview, topics | Comprehensive summary | Medium | High |
| **Local** | Specific entities, facts | Direct, factual | Fast | Medium |
| **Drift** | Entity connections, multi-hop | Adaptive, progressive | Slow | High |

### When to Use Each Method

- **Use ToG Search** when you need:
  - Deep analysis of entity relationships
  - Evidence-based reasoning
  - Complex, multi-step questions
  - Detailed explanations

- **Use Global Search** when you need:
  - Dataset overview
  - Main themes and topics
  - High-level summaries
  - Broad understanding

- **Use Local Search** when you need:
  - Information about specific entities
  - Direct facts and evidence
  - Quick, targeted answers
  - Document-level details

- **Use Drift Search** when you need:
  - Entity relationship analysis
  - Multi-hop reasoning
  - Adaptive exploration
  - Contextual understanding

## Troubleshooting

### Common Issues and Solutions

#### 1. Indexing Fails

**Error**: `API rate limit exceeded` or `Authentication failed`

**Solutions**:
- Verify API key in `.env` file: `cat .env`
- Check network connectivity
- Reduce `concurrent_requests` in settings
- Use a faster model for testing

#### 2. Search Method Not Found

**Error**: `module 'graphrag.api' has no attribute 'tog_search'`

**Solutions**:
- Ensure ToG functions are exported in `graphrag/api/__init__.py`
- Verify you're using the correct GraphRAG version
- Check that ToG configuration is present in `settings.yaml`

#### 3. Empty or Poor Results

**Possible causes**:
- Input data too small or simple
- Entities not properly extracted
- Wrong search method for question type

**Solutions**:
- Increase data size and complexity
- Adjust `max_gleanings` in `extract_graph` configuration
- Try different search methods
- Increase chunk size for better context

#### 4. ToG Search Takes Too Long

**Solutions**:
- Reduce `width` and `depth` parameters
- Decrease `max_exploration_paths`
- Use a faster model (`gpt-4o-mini` instead of `gpt-4-turbo`)
- Enable caching to avoid repeated processing

#### 5. Memory Issues

**Solutions**:
- Reduce `batch_size` in `embed_text` configuration
- Increase chunk size to reduce number of chunks
- Use smaller context windows
- Clear cache directory if needed

### Verification Steps

1. **Check indexing succeeded**:
   ```bash
   ls -lh output/*.parquet
   # Should see: entities, relationships, communities, community_reports, text_units, documents
   ```

2. **Verify ToG configuration**:
   ```bash
   grep -A 10 "tog_search:" settings.yaml
   ```

3. **Test with simple queries first**:
   ```bash
   graphrag query --root . --method global --query "Summarize this dataset"
   ```

4. **Check API key is working**:
   ```bash
   echo $GRAPHRAG_API_KEY
   # Should display your API key
   ```

### Performance Optimization

1. **Use caching**: The `cache` directory stores LLM responses to avoid re-processing
2. **Start small**: Test with a small dataset first, then scale up
3. **Optimize chunk size**:
   - Smaller chunks = more precise but slower
   - Larger chunks = faster but less precise
4. **Choose appropriate models**:
   - Fast/cheap: `gpt-4o-mini` (recommended for most cases)
   - High quality: `gpt-4-turbo-preview` (for critical applications)
5. **Tune ToG parameters**:
   - Lower `width` (2-3) = faster
   - Lower `depth` (2-3) = faster
   - Lower `max_exploration_paths` (5-10) = faster

### Query Examples

#### ToG Search Examples

```bash
# Topic relationships
graphrag query --root . --method tog \
  --query "How do the main topics relate to each other?"

# Entity influence analysis
graphrag query --root . --method tog \
  --query "What entities have the most influence on others?"

# Causal relationships
graphrag query --root . --method tog \
  --query "What are the cause-and-effect relationships in this data?"

# Comparative analysis
graphrag query --root . --method tog \
  --query "Compare the roles of different entities in the project"
```

#### Method Selection Guide

- **Start with Global Search** to understand your dataset
- **Use Local Search** for specific entity information
- **Apply ToG Search** for complex relationship analysis
- **Try Drift Search** for adaptive exploration

## Advanced Usage

### Custom Entity Types

```yaml
extract_graph:
  entity_types: [organization, person, geo, event, product, technology]
```

### Streaming Responses

```bash
graphrag query --root . --method tog --streaming --query "Your complex question"
```

### Batch Queries

Create a script to run multiple queries efficiently:

```python
import os
from graphrag.api import tog_search

queries = [
    "What are the main topics?",
    "How are entities connected?",
    "What are the key relationships?"
]

for query in queries:
    result = tog_search(
        query=query,
        config_path="settings.yaml",
        data_path="output"
    )
    print(f"Query: {query}")
    print(f"Result: {result}\n")
```

### Performance Monitoring

Enable verbose output to debug issues:

```bash
graphrag query --root . --method tog --verbose --query "Your question"
```

## Next Steps

- Experiment with different ToG parameters for your use case
- Try all search methods to understand their differences
- Scale up to larger datasets once basics are working
- Customize prompts for your domain-specific needs
- Explore the [GraphRAG documentation](../get_started.md) for advanced features

## References

- [GraphRAG Getting Started](../get_started.md)
- [Configuration Options](../config/overview.md)
- [Query Engine Overview](../query/overview.md)
- [API Reference](../api/reference.md)