# How to Run GraphRAG: Complete Guide

This guide will walk you through the complete process of using GraphRAG, from indexing your documents to querying the knowledge graph.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Project Setup](#project-setup)
4. [Configuration](#configuration)
5. [Indexing Documents](#indexing-documents)
6. [Querying the Knowledge Graph](#querying-the-knowledge-graph)
7. [Advanced Usage](#advanced-usage)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

- **Python**: Version 3.10-3.12
- **API Key**: 
  - Gemini (as configured in your current setup)
- **System Resources**: GraphRAG can consume significant LLM resources. Start with small datasets first.

---

## Installation

```bash
cd graphrag
pip install -e .
```

---

## Project Setup

### Step 1: Create Your Project Directory

Create a directory structure for your GraphRAG project:

```bash
mkdir -p ./my-project/input
cd ./my-project
```

### Step 2: Add Your Documents

Place your text documents in the `input` folder:

```bash
# Example: Download a sample text
curl https://www.gutenberg.org/cache/epub/24022/pg24022.txt -o ./input/book.txt

# Or copy your own documents
cp /path/to/your/documents/*.txt ./input/
```

Supported input formats:
- `.txt` - Plain text files
- `.pdf` - PDF documents
- `.docx` - Word documents
- `.csv` - CSV files

### Step 3: Initialize the Workspace

Run the initialization command:

```bash
graphrag init --root .
```

This creates two important files:
- `.env` - Contains environment variables (API keys)
- `settings.yaml` - Contains pipeline configuration

---

## Configuration

#### For Gemini (Current Configuration)

Your `settings.yaml` is already configured for Gemini:

```yaml
models:
  default_chat_model:
    type: chat
    auth_type: api_key
    api_key: ${GEMINI_API_KEY}
    model_provider: gemini
    model: gemini-2.5-flash-lite
    model_supports_json: true

  default_embedding_model:
    type: embedding
    auth_type: api_key
    api_key: ${GEMINI_API_KEY}
    model_provider: gemini
    model: text-embedding-001
```

Make sure your `.env` file contains:
```bash
GEMINI_API_KEY=your-gemini-api-key-here
```

### Key Configuration Parameters

Edit `settings.yaml` to customize:

#### Chunking Settings
```yaml
chunks:
  size: 1200          # Size of text chunks
  overlap: 100        # Overlap between chunks
```

#### Entity Extraction
```yaml
extract_graph:
  entity_types: [organization, person, geo, event]
  max_gleanings: 1    # Number of extraction passes
```

#### Community Reports
```yaml
community_reports:
  max_length: 2000           # Max length of reports
  max_input_length: 8000     # Max input context
```

---

## Indexing Documents

### Run the Indexing Pipeline

Execute the indexing command:

```bash
graphrag index --root .
```

**What happens during indexing:**

1. **Text Chunking**: Documents are split into overlapping chunks
2. **Entity Extraction**: LLM identifies entities (people, organizations, locations, events)
3. **Relationship Extraction**: LLM identifies relationships between entities
4. **Graph Construction**: Entities and relationships form a knowledge graph
5. **Community Detection**: Graph clustering identifies semantic communities
6. **Community Summarization**: Each community gets an AI-generated summary
7. **Embedding Generation**: Text chunks and entities are embedded into vector space

### Expected Output

After completion, you'll find these folders:

```
my-project/
├── input/              # Your source documents
├── output/             # Parquet files with extracted data
│   ├── create_final_entities.parquet
│   ├── create_final_relationships.parquet
│   ├── create_final_communities.parquet
│   ├── create_final_community_reports.parquet
│   ├── create_final_text_units.parquet
│   └── ...
├── cache/              # Cached LLM responses (speeds up re-runs)
├── .env
└── settings.yaml
```

### Indexing Time and Cost

- **Time**: Depends on dataset size, model speed, and chunk size
  - Small dataset (1 book): 5-15 minutes
  - Medium dataset (100 documents): 1-3 hours
  - Large dataset (1000+ documents): Several hours

- **Cost**: Proportional to number of LLM calls
  - Start with cheaper/faster models (e.g., gemini-2.5-flash-lite)
  - Monitor API usage during first run

---

## Querying the Knowledge Graph

After indexing completes, you can query your knowledge graph using different search methods.

### 1. Global Search

**Best for**: High-level questions about the entire dataset, themes, patterns, and aggregated insights.

**Examples:**
- "What are the top themes in this dataset?"
- "What are the main topics discussed?"
- "Summarize the key findings across all documents"

**Command:**
```bash
graphrag query \
  --root . \
  --method global \
  --query "What are the top themes in this story?"
```

**How it works:**
- Uses community reports from the knowledge graph
- Map-reduce approach for comprehensive answers
- Analyzes the dataset as a whole

### 2. Local Search

**Best for**: Specific questions about particular entities, relationships, or detailed information.

**Examples:**
- "Who is Scrooge and what are his main relationships?"
- "What are the healing properties of chamomile?"
- "Describe the relationship between X and Y"

**Command:**
```bash
graphrag query \
  --root . \
  --method local \
  --query "Who is Scrooge and what are his main relationships?"
```

**How it works:**
- Identifies relevant entities from the query
- Retrieves connected entities, relationships, and text chunks
- Combines structured graph data with raw text for detailed answers

### 3. DRIFT Search

**Best for**: Local search with broader community context.

**Command:**
```bash
graphrag query \
  --root . \
  --method drift \
  --query "Your question here"
```

**How it works:**
- Enhanced local search that includes community information
- Provides more comprehensive answers than basic local search

### 4. ToG (Think-on-Graph) Search

**Best for**: Complex multi-hop reasoning questions that require deep traversal of the knowledge graph, path-finding queries, and causal chain analysis.

**Examples:**
- "What is the connection between entity A and entity B?"
- "How does X influence Y through intermediary factors?"
- "What are the paths from X to Y?"
- "What factors led to outcome Z?"
- "Trace the chain of events from A to B"

**Command:**
```bash
graphrag query \
  --root . \
  --method tog \
  --query "What connects entity A to entity B?"
```

**How it works:**
- Uses iterative graph exploration with beam search
- LLM-guided pruning of exploration paths based on relevance
- Multi-hop reasoning over discovered paths
- Returns traceable reasoning paths showing how answers were derived

#### ToG Configuration

Customize ToG search behavior in `settings.yaml`:

```yaml
tog_search:
  # Model Configuration
  chat_model_id: default_chat_model
  embedding_model_id: default_embedding_model

  # Exploration Parameters
  width: 3                    # Beam width - number of paths to explore
  depth: 3                    # Maximum depth of graph traversal

  # Pruning Configuration
  prune_strategy: llm         # Options: "llm", "semantic"
  num_retain_entity: 5        # Entities to keep at each step

  # Temperature Settings
  temperature_exploration: 0.4  # Randomness during exploration
  temperature_reasoning: 0.0    # Deterministic final reasoning

  # Resource Limits
  max_context_tokens: 8000
  max_exploration_paths: 10
```

**ToG Performance Considerations:**
- **Speed vs. Quality Trade-offs**:
  - Lower `width` (1-2): Focused, faster, may miss alternative paths
  - Higher `width` (5+): Comprehensive, slower, more complete exploration
  - Lower `depth` (1-2): Quick, surface-level connections
  - Higher `depth` (4+): Deep, multi-hop reasoning
  - `prune_strategy`: "semantic" is faster, "llm" is more accurate

### Query Configuration

Customize query behavior in `settings.yaml`:

#### Local Search Settings
```yaml
local_search:
  max_context_tokens: 8000      # Context window size
  text_unit_prop: 0.5           # Proportion of text chunks
  community_prop: 0.25          # Proportion of community info
  top_k_entities: 10            # Number of top entities
  top_k_relationships: 10       # Number of top relationships
```

#### Global Search Settings
```yaml
global_search:
  map_max_length: 1000          # Max length for map phase
  reduce_max_length: 2000       # Max length for reduce phase
  max_context_tokens: 8000      # Total context budget
```

---

## Advanced Usage

### Using Python API

For programmatic access, use the Python API:

```python
from graphrag.api import index, query

# Index documents
index(root="./my-project")

# Query using global search
result = query(
    root="./my-project",
    method="global",
    query="What are the main themes?"
)
print(result)

# Query using local search
result = query(
    root="./my-project",
    method="local",
    query="Who is the main character?"
)
print(result)
```

### Re-running Indexing

If you modify your documents or configuration:

```bash
# Full re-index (uses cache for unchanged parts)
graphrag index --root .

# Clear cache and re-index from scratch
rm -rf ./cache
graphrag index --root .
```

### Optimizing Performance

**For faster indexing:**
- Use faster/cheaper models (e.g., `gemini-2.5-flash-lite`)
- Increase chunk size to reduce number of chunks
- Reduce `max_gleanings` in entity extraction
- Use caching effectively

**For better quality:**
- Use more powerful models (e.g., `gpt-4`, `gemini-2.5-flash`)
- Decrease chunk size for finer granularity
- Increase `max_gleanings` for better entity extraction
- Customize entity types for your domain

---

## Troubleshooting

### Common Issues

#### 1. API Key Errors

```
Error: Invalid API key
```

**Solution:** Check your `.env` file and ensure the API key is correct:
```bash
cat .env
```

#### 2. Out of Memory

```
Error: Out of memory
```

**Solution:**
- Reduce batch size in `settings.yaml`
- Process documents in smaller batches
- Increase system memory

#### 3. Rate Limiting

```
Error: Rate limit exceeded
```

**Solution:**
- Add retry logic (GraphRAG handles this automatically)
- Reduce concurrent requests
- Upgrade API tier for higher limits

#### 4. Poor Query Results

**Solutions:**
- Try different search methods (global vs. local)
- Adjust query phrasing to be more specific
- Check if indexing completed successfully
- Review entity extraction quality in output files

#### 5. Indexing Takes Too Long

**Solutions:**
- Use faster models (e.g., gemini-2.5-flash-lite)
- Increase chunk size
- Reduce max_gleanings
- Process smaller dataset first to test

### Checking Output Quality

Inspect the parquet files to verify extraction quality:

```python
import pandas as pd

# Check extracted entities
entities = pd.read_parquet("./output/create_final_entities.parquet")
print(entities.head())

# Check relationships
relationships = pd.read_parquet("./output/create_final_relationships.parquet")
print(relationships.head())

# Check community reports
reports = pd.read_parquet("./output/create_final_community_reports.parquet")
print(reports.head())
```

---

## Quick Reference

### Essential Commands

```bash
# Initialize project
graphrag init --root ./my-project

# Index documents
graphrag index --root ./my-project

# Global search query
graphrag query --root ./my-project --method global --query "Your question"

# Local search query
graphrag query --root ./my-project --method local --query "Your question"

# DRIFT search query
graphrag query --root ./my-project --method drift --query "Your question"

# ToG (Think-on-Graph) search query
graphrag query --root ./my-project --method tog --query "Your question"
```

### Directory Structure

```
my-project/
├── input/              # Place your documents here
├── output/             # Generated knowledge graph (parquet files)
├── cache/              # Cached LLM responses
├── prompts/            # Custom prompt templates (optional)
├── .env                # API keys and environment variables
└── settings.yaml       # Pipeline configuration
```

---

## Additional Resources

- [Configuration Documentation](../config/overview.md)
- [Local Search Details](../query/local_search.md)
- [Global Search Details](../query/global_search.md)
- [DRIFT Search Details](../query/drift_search.md)
- [ToG Search Details](../tunh/tog_search_guide.md)
- [Architecture Overview](../index/architecture.md)
- [Visualization Guide](../visualization_guide.md)

---

## Tips for Success

1. **Start Small**: Test with a small dataset first (1-10 documents)
2. **Monitor Costs**: Use cheaper models initially to understand resource usage
3. **Customize Prompts**: Adjust prompts in `settings.yaml` for your domain
4. **Iterate Configuration**: Fine-tune chunk size, entity types, and other parameters
5. **Combine Search Methods**: Use global for overview, local for details
6. **Use Caching**: Keep the cache folder to speed up re-runs
7. **Validate Output**: Check parquet files to ensure good entity extraction

Happy GraphRAG-ing!
