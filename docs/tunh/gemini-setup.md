# Setting Up GraphRAG with Google Gemini Models

This guide provides step-by-step instructions for configuring GraphRAG to use Google's Gemini models for inference and embedding.

## Prerequisites

- GraphRAG version 2.6.0 or higher
- Google Cloud account with Gemini API access
- Gemini API key

## Supported Models

- **Inference Model**: `gemini-2.5-flash` (or `gemini-2.0-flash-lite`)
- **Embedding Model**: `gemini-embedding-001`

## Configuration Steps

### 1. Obtain Your Gemini API Key

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Create a new API key or use an existing one
3. Copy the API key for use in configuration

### 2. Create Environment File

Create a `.env` file in your GraphRAG project root directory:

```bash
# .env
GEMINI_API_KEY=your_gemini_api_key_here
```

**Important**: Add `.env` to your `.gitignore` file to prevent accidentally committing your API key.

### 3. Configure settings.yaml

Create or update your `settings.yaml` file in the project root with the following configuration:

```yaml
models:
  default_chat_model:
    type: chat
    auth_type: api_key
    api_key: ${GEMINI_API_KEY}
    model_provider: gemini
    model: gemini-2.0-flash-lite
    model_supports_json: true

  default_embedding_model:
    type: embedding
    auth_type: api_key
    api_key: ${GEMINI_API_KEY}
    model_provider: gemini
    model: text-embedding-004

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

# Cache configuration (recommended for development)
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
  prompt: "prompts/community_report.txt"
  max_length: 2000
  max_input_length: 8000

# Summarization
summarize_descriptions:
  model_id: default_chat_model
  prompt: "prompts/summarize_descriptions.txt"
  max_length: 500
  max_input_length: 3000

# Query configurations
local_search:
  chat_model_id: default_chat_model
  embedding_model_id: default_embedding_model
  max_context_tokens: 8000
  text_unit_prop: 0.5
  community_prop: 0.25
  top_k_entities: 10
  top_k_relationships: 10

global_search:
  chat_model_id: default_chat_model
  map_max_length: 1000
  reduce_max_length: 2000
  max_context_tokens: 8000
```

## Understanding LiteLLM Integration

GraphRAG uses LiteLLM (starting from version 2.6.0) to interface with Gemini models. The configuration follows this pattern:

- `type`: Set to `chat` or `embedding`
- `model_provider`: The provider name (e.g., `gemini`, `openai`, `azure`)
- `model`: The specific model name from the provider's API

LiteLLM translates these into API calls in the format: `{model_provider}/{model}`

For example: `gemini/gemini-2.0-flash-lite`

## Alternative Configuration (JSON)

If you prefer JSON format, create `settings.json`:

```json
{
  "models": {
    "default_chat_model": {
      "type": "chat",
      "auth_type": "api_key",
      "api_key": "${GEMINI_API_KEY}",
      "model_provider": "gemini",
      "model": "gemini-2.0-flash-lite",
      "model_supports_json": true
    },
    "default_embedding_model": {
      "type": "embedding",
      "auth_type": "api_key",
      "api_key": "${GEMINI_API_KEY}",
      "model_provider": "gemini",
      "model": "text-embedding-004"
    }
  }
}
```

## Project Directory Structure

Ensure your project has the following structure:

```
your-project/
├── .env                    # API keys (DO NOT commit)
├── settings.yaml           # Configuration file
├── input/                  # Input documents
│   └── your-documents.txt
├── output/                 # Generated artifacts
├── cache/                  # LLM response cache
└── prompts/                # Custom prompts (optional)
```

## Initialize GraphRAG

If you're starting a new project:

```bash
# Initialize GraphRAG project
graphrag init --root .

# This will create the directory structure and default configuration
```

Then update the `settings.yaml` file with the Gemini configuration shown above.

## Running the Pipeline

### Indexing

To create the knowledge graph from your documents:

```bash
graphrag index --root .
```

### Querying

**Local Search** (detail-oriented questions):
```bash
graphrag query --root . --method local "Your question here"
```

**Global Search** (broad questions):
```bash
graphrag query --root . --method global "Your question here"
```

## Rate Limiting and Cost Optimization

To control API usage and costs, add these parameters to your model configuration:

```yaml
models:
  default_chat_model:
    type: chat
    auth_type: api_key
    api_key: ${GEMINI_API_KEY}
    model_provider: gemini
    model: gemini-2.0-flash-lite
    model_supports_json: true
    # Rate limiting
    tokens_per_minute: 40000
    requests_per_minute: 50
    concurrent_requests: 5
    # Retry strategy
    retry_strategy: exponential_backoff
    max_retries: 3
    max_retry_wait: 10.0
    request_timeout: 60.0
```

## Troubleshooting

### Common Issues

1. **API Key Not Found**
   - Ensure `.env` file is in the project root
   - Verify the environment variable name matches: `GEMINI_API_KEY`
   - Check that the API key is valid and active

2. **Rate Limit Errors**
   - Reduce `tokens_per_minute` and `requests_per_minute`
   - Increase `concurrent_requests` value
   - Add delays between requests

3. **JSON Mode Errors**
   - Ensure `model_supports_json: true` is set
   - Verify the model supports structured outputs
   - Check that you're using a compatible Gemini model version

4. **Embedding Dimension Mismatches**
   - Ensure you're consistently using the same embedding model
   - Clear the cache if you switch embedding models: `rm -rf cache/`

## Model Selection Notes

### Gemini 2.0 Flash Lite vs Gemini 2.5 Flash

- **Gemini 2.0 Flash Lite**: Faster, more cost-effective, suitable for most use cases
- **Gemini 2.5 Flash**: More capable, better for complex extraction tasks

### Embedding Models

- **text-embedding-004**: Latest Google embedding model with improved performance
- **gemini-embedding-001**: Legacy model, still supported

## Additional Resources

- [GraphRAG Documentation](https://microsoft.github.io/graphrag/)
- [LiteLLM Documentation](https://docs.litellm.ai/)
- [Google AI Studio](https://aistudio.google.com/)
- [GraphRAG Model Configuration](../config/models.md)
- [GraphRAG YAML Configuration](../config/yaml.md)

## Support

For issues specific to:
- **GraphRAG**: [GitHub Issues](https://github.com/microsoft/graphrag/issues)
- **Gemini API**: [Google AI Forum](https://discuss.ai.google.dev/)
- **LiteLLM**: [LiteLLM GitHub](https://github.com/BerriAI/litellm)
