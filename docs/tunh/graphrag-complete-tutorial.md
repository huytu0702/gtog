# GraphRAG Complete Tutorial: Indexing to Querying

A step-by-step guide to run GraphRAG on your documents and query the knowledge graph.

## Prerequisites

- Python 3.10-3.12
- Google Gemini API key (or other LLM provider)
- GraphRAG installed: `pip install -e .` (from repository root)

## Step 1: Create Project Directory

```bash
mkdir -p my-project/input
cd my-project
```

## Step 2: Prepare Your Documents

Place your text files in the `input` folder:

```bash
# Copy your documents
cp /path/to/your/documents/*.txt ./input/
```

Supported formats:
- `.txt` - Plain text files
- `.pdf` - PDF documents  
- `.docx` - Word documents
- `.csv` - CSV files

Example: Create a sample file
```bash
# Create sample_story.txt with your content
echo "Your document text here..." > input/sample_story.txt
```

## Step 3: Initialize the Project

```bash
graphrag init --root .
```

This creates:
- `.env` - Environment variables (API keys)
- `settings.yaml` - Pipeline configuration
- `prompts/` - Prompt templates

## Step 4: Configure API Key

Edit `.env` and add your Gemini API key:

```bash
GOOGLE_API_KEY=your-gemini-api-key-here
```

Or if using OpenAI:
```bash
GRAPHRAG_API_KEY=your-openai-api-key-here
```

## Step 5: Configure Settings (if needed)

Edit `settings.yaml` to match your LLM provider.

### For Google Gemini:

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
```

### For OpenAI:

```yaml
models:
  default_chat_model:
    type: chat
    model_provider: openai
    auth_type: api_key
    api_key: ${GRAPHRAG_API_KEY}
    model: gpt-4-turbo-preview
    model_supports_json: true

  default_embedding_model:
    type: embedding
    model_provider: openai
    auth_type: api_key
    api_key: ${GRAPHRAG_API_KEY}
    model: text-embedding-3-small
```

### Customize Indexing Settings:

```yaml
# Text chunking
chunks:
  size: 1200          # Size of text chunks
  overlap: 100        # Overlap between chunks

# Entity extraction
extract_graph:
  entity_types: [organization, person, geo, event]
  max_gleanings: 1    # Number of extraction passes

# Community reports
community_reports:
  max_length: 2000           # Max report length
  max_input_length: 8000     # Max input context
```

## Step 6: Run Indexing

```bash
graphrag index --root .
```

**What happens:**
1. Loads documents from `input/` folder
2. Chunks text into overlapping segments
3. Extracts entities (people, organizations, locations, events)
4. Identifies relationships between entities
5. Detects communities using graph clustering
6. Generates summaries for each community
7. Creates embeddings for semantic search

**Output files created:**
```
output/
├── entities.parquet              # Extracted entities
├── relationships.parquet         # Entity relationships
├── communities.parquet           # Community clusters
├── community_reports.parquet     # AI-generated summaries
├── documents.parquet             # Processed documents
├── text_units.parquet            # Text chunks
├── lancedb/                      # Vector embeddings database
├── context.json                  # Pipeline context
└── stats.json                    # Pipeline statistics
```

**Indexing time:**
- Small dataset (1 book): 5-15 minutes
- Medium dataset (100 documents): 1-3 hours
- Large dataset (1000+ documents): Several hours

## Step 7: Query the Knowledge Graph

### Local Search (Entity-focused)

Best for: Specific questions about entities, relationships, or detailed information

```bash
graphrag query \
  --root . \
  --method local \
  --query "Who is [entity_name] and what are their relationships?"
```

**Example:**
```bash
graphrag query --root . --method local --query "Who is Scrooge and what are his main relationships?"
```

**Output:** Detailed entity analysis with relationships, citing specific data sources

### Global Search (Thematic)

Best for: High-level questions about themes, patterns, and overall insights

```bash
graphrag query \
  --root . \
  --method global \
  --query "What are the main themes in this dataset?"
```

**Example:**
```bash
graphrag query --root . --method global --query "What are the main themes in this story?"
```

**Output:** Comprehensive analysis synthesizing patterns across the entire knowledge graph

### DRIFT Search (Local with context)

Best for: Local search with broader community context

```bash
graphrag query \
  --root . \
  --method drift \
  --query "Your question here"
```

## Step 8: Verify Output Quality

Inspect extracted data:

```python
import pandas as pd

# Check extracted entities
entities = pd.read_parquet("./output/entities.parquet")
print(entities.head())

# Check relationships
relationships = pd.read_parquet("./output/relationships.parquet")
print(relationships.head())

# Check community reports
reports = pd.read_parquet("./output/community_reports.parquet")
print(reports.head())
```

## Common Issues & Solutions

### Issue: `KeyError: 'GEMINI_API_KEY'`

**Solution:** Ensure `.env` file exists in project root with:
```bash
GOOGLE_API_KEY=your-actual-api-key
```

### Issue: Embedding Model Not Found

**Error:** `models/text-embedding-001 is not found`

**Solution:** Use correct model name:
- Gemini: `embedding-001` (not `text-embedding-001`)
- OpenAI: `text-embedding-3-small`

Update in `settings.yaml`:
```yaml
model: embedding-001  # Gemini
# or
model: text-embedding-3-small  # OpenAI
```

### Issue: Out of Memory

**Solution:**
1. Reduce chunk size in `settings.yaml`
2. Process smaller document batches
3. Reduce `max_gleanings` for faster processing

### Issue: Rate Limiting

**Solution:**
- GraphRAG handles retries automatically
- Reduce concurrent requests if needed
- Upgrade your API tier for higher limits

### Issue: Poor Query Results

**Solutions:**
- Ensure indexing completed successfully (check `output/` folder)
- Try different search methods (local vs. global)
- Rephrase queries to be more specific
- Check entity extraction quality in parquet files

## Advanced: Re-index Documents

If you modify documents or configuration:

```bash
# Re-index (uses cache for unchanged parts)
graphrag index --root .

# Clear cache and re-index from scratch
rm -rf ./cache
graphrag index --root .
```

## Performance Optimization

### For Faster Indexing:
- Use cheaper/faster models (e.g., `gemini-2.5-flash-lite`)
- Increase chunk size (fewer chunks to process)
- Reduce `max_gleanings` to 0 (skip re-extraction)
- Keep cache folder for subsequent runs

### For Better Quality:
- Use more powerful models (e.g., `gpt-4`, `gemini-2.5-flash`)
- Decrease chunk size for finer granularity
- Increase `max_gleanings` for thorough extraction
- Customize entity types for your domain

## Complete Example: A Christmas Carol

```bash
# 1. Create project
mkdir -p christmas-carol/input
cd christmas-carol

# 2. Add document
curl https://www.gutenberg.org/cache/epub/24022/pg24022.txt \
  -o ./input/carol.txt

# 3. Initialize
graphrag init --root .

# 4. Configure API (edit .env with your key)

# 5. Run indexing
graphrag index --root .

# 6. Query
graphrag query --root . --method local \
  --query "Who is Scrooge?"

graphrag query --root . --method global \
  --query "What are the main themes?"
```

## Directory Structure

```
my-project/
├── input/                  # Place your documents here
│   ├── document1.txt
│   ├── document2.pdf
│   └── ...
├── output/                 # Generated by GraphRAG
│   ├── entities.parquet
│   ├── relationships.parquet
│   ├── communities.parquet
│   ├── community_reports.parquet
│   ├── documents.parquet
│   ├── text_units.parquet
│   ├── lancedb/            # Vector embeddings
│   ├── context.json
│   └── stats.json
├── cache/                  # LLM response cache
├── logs/                   # Indexing logs
├── prompts/                # Prompt templates (auto-generated)
├── .env                    # API keys
└── settings.yaml           # Configuration
```

## Quick Reference

```bash
# Initialize project
graphrag init --root ./my-project

# Index documents
graphrag index --root ./my-project

# Local search (specific queries)
graphrag query --root ./my-project --method local \
  --query "Your specific question"

# Global search (thematic analysis)
graphrag query --root ./my-project --method global \
  --query "What are the main patterns?"

# DRIFT search (local + context)
graphrag query --root ./my-project --method drift \
  --query "Your question"
```

## Tips for Success

1. **Start Small:** Test with 1-10 small documents first
2. **Monitor Costs:** Use cheaper models initially to understand resource usage
3. **Verify Extraction:** Check parquet files to ensure good entity extraction
4. **Customize Prompts:** Adjust prompts in `settings.yaml` for your domain
5. **Iterate Configuration:** Fine-tune chunk size, entity types, and parameters
6. **Use Caching:** Keep cache folder to speed up re-runs
7. **Combine Search Methods:** Use global for overview, local for details

## Resources

- [Configuration Guide](./how-to-run-graphrag.md)
- [Architecture Overview](./architecture.md)
- [Entity Extraction Details](./entity-extraction.md)
- [Query Methods Explained](./query-methods.md)

## Support

If you encounter issues:
1. Check the logs in `logs/indexing-engine.log`
2. Verify `.env` file contains valid API key
3. Ensure documents are in correct format in `input/` folder
4. Try with a smaller test document first
5. Check that `settings.yaml` matches your LLM provider

Happy GraphRAG-ing!
