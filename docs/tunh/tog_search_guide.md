# ToG (Think-on-Graph) Search Guide

## Overview

ToG (Think-on-Graph) is an advanced search method in GraphRAG that enables deep, multi-hop reasoning over knowledge graphs. Unlike other search methods, ToG iteratively explores the graph using beam search guided by LLM scoring, making it ideal for complex questions requiring multi-step inference.

## When to Use ToG Search

### Best Use Cases

ToG search excels at:

1. **Multi-hop reasoning questions**
   - "What is the connection between entity A and entity B?"
   - "How does X influence Y through intermediary factors?"

2. **Path-finding queries**
   - "What are the paths from X to Y?"
   - "How are these two concepts related?"

3. **Complex causal chains**
   - "What factors led to outcome Z?"
   - "Trace the chain of events from A to B"

4. **Exploratory analysis**
   - "What are the indirect relationships between X and Y?"
   - "Find unexpected connections in the data"

### When NOT to Use ToG

- **Simple factual lookups**: Use Local search instead
- **High-level summaries**: Use Global search instead
- **Quick overviews**: Use Basic search instead
- **Specific entity details**: Use Local search instead

## Configuration

Add ToG configuration to your `settings.yaml`:

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

### Configuration Parameters Explained

**width**: Controls the breadth of exploration
- Lower (1-2): Focused, faster, may miss alternative paths
- Higher (5+): Comprehensive, slower, more complete exploration
- Recommended: 3 for balanced exploration

**depth**: Controls how far to traverse the graph
- Lower (1-2): Quick, surface-level connections
- Higher (4+): Deep, multi-hop reasoning
- Recommended: 3 for most questions

**prune_strategy**:
- `llm`: Uses language model to score relevance (slower, more accurate)
- `semantic`: Uses embedding similarity (faster, good quality)

**num_retain_entity**: How many candidate entities to evaluate at each step
- Higher values = more thorough but slower
- Recommended: 5 for balanced performance

## Usage

### Command Line

```bash
# Basic ToG search
graphrag query --root ./my-project --method tog --query "What connects entity A to entity B?"

# With streaming output
graphrag query --root ./my-project --method tog --streaming --query "Your question here"
```

### Python API

```python
import asyncio
import pandas as pd
from graphrag.config.load_config import load_config
from graphrag.api import tog_search

# Load configuration and data
config = load_config(root_dir="./my-project")
entities = pd.read_parquet("./my-project/output/create_final_entities.parquet")
relationships = pd.read_parquet("./my-project/output/create_final_relationships.parquet")

# Run ToG search
response, context = asyncio.run(
    tog_search(
        config=config,
        entities=entities,
        relationships=relationships,
        query="What is the connection between X and Y?"
    )
)

print(response)
print(f"Explored {context['num_explored_entities']} entities")
print(f"Reasoning paths: {context['reasoning_paths']}")
```

## Understanding ToG Results

ToG returns both an answer and exploration metadata:

```python
{
    "reasoning_paths": [
        "EntityA -> relationX -> EntityB -> relationY -> EntityC",
        "EntityA -> relationZ -> EntityD -> relationY -> EntityC"
    ],
    "num_explored_entities": 45,
    "max_depth_reached": 3
}
```

**reasoning_paths**: The paths through the graph that led to the answer
**num_explored_entities**: Total entities considered during exploration
**max_depth_reached**: How deep the search went

## Performance Considerations

### Speed vs. Quality Trade-offs

| Setting | Fast | Balanced | Thorough |
|---------|------|----------|----------|
| width | 1-2 | 3 | 5+ |
| depth | 1-2 | 3 | 4+ |
| prune_strategy | semantic | semantic | llm |
| num_retain_entity | 3 | 5 | 10 |

### Cost Optimization

ToG makes multiple LLM calls during exploration:
- Relation scoring: ~(width × depth × num_retain_entity) calls
- Entity scoring: Similar magnitude
- Final reasoning: 1-2 calls

**Tips to reduce cost:**
1. Use faster/cheaper models for exploration
2. Use semantic pruning instead of LLM pruning
3. Reduce width and depth for simpler questions
4. Cache exploration results for similar queries

## Advanced Usage

### Custom Prompts

Override default prompts in `settings.yaml`:

```yaml
tog_search:
  relation_scoring_prompt: "./prompts/my_relation_prompt.txt"
  entity_scoring_prompt: "./prompts/my_entity_prompt.txt"
  reasoning_prompt: "./prompts/my_reasoning_prompt.txt"
```

### Programmatic Control

```python
from graphrag.query.structured_search.tog_search import ToGSearch
from graphrag.query.factory import get_tog_search_engine

# Create custom search engine
search_engine = get_tog_search_engine(
    config=config,
    entities=entities_list,
    relationships=relationships_list,
    response_type="detailed",
)

# Run with custom callbacks
class MyCallback:
    def on_context(self, context):
        print(f"Depth {context.get('depth')}: exploring {context.get('frontier_size')} entities")

result = await search_engine.search(
    query="Your question",
)
```

## Troubleshooting

### Issue: Search takes too long

**Solutions:**
- Reduce `width` and `depth` parameters
- Switch to `semantic` pruning strategy
- Reduce `num_retain_entity`

### Issue: Answer quality is poor

**Solutions:**
- Increase `width` to explore more paths
- Increase `depth` for deeper reasoning
- Switch to `llm` pruning strategy
- Check if entities and relationships were extracted well during indexing

### Issue: Out of memory errors

**Solutions:**
- Reduce `max_exploration_paths`
- Process smaller graph subsets
- Use semantic pruning (lighter weight)

## Comparison with Other Methods

| Aspect | ToG | Local | Global | DRIFT |
|--------|-----|-------|--------|-------|
| Question Complexity | High | Medium | Low-Medium | Medium |
| Reasoning Depth | Multi-hop | Single-hop | Aggregated | Mixed |
| Speed | Slower | Fast | Medium | Medium |
| Explainability | High (paths) | Medium | Low | Medium |
| Best For | Complex reasoning | Entity details | Themes | Balanced |