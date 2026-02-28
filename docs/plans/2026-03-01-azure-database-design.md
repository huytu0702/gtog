# Azure Database Layer Design for GraphRAG (gtog)

**Date**: 2026-03-01
**Status**: Approved
**Scope**: Database/storage layer only (Phase 1 of full deployment)

## Context

The project currently uses local file storage (output/, cache/, logs/) and LanceDB for vector search. This design migrates the storage layer to Azure for production durability and scalability.

## Decisions

- **LLM Provider**: Gemini (kept as-is, no migration)
- **Storage**: Azure Blob Storage (native GraphRAG support)
- **Vector Store**: Azure AI Search (native GraphRAG support, replaces LanceDB)
- **Provisioning**: Azure CLI scripts (transparent, auditable)
- **Region**: `southeastasia`
- **Subscription**: Azure for Students (`1095803e-80bf-47e0-961f-3d74cb4c605c`)

## Azure Resources

| Resource | Name | SKU | Purpose |
|---|---|---|---|
| Resource Group | `rg-gtog-prod` | - | Container for all resources |
| Storage Account | `stgtogprod` | Standard LRS | GraphRAG index/cache/log files |
| Blob Container | `gtog-input` | - | Input documents |
| Blob Container | `gtog-output` | - | Indexed graph artifacts (parquet files) |
| Blob Container | `gtog-cache` | - | LLM call cache |
| Blob Container | `gtog-logs` | - | Pipeline logs |
| Azure AI Search | `srch-gtog-prod` | Free | Vector embeddings (replaces LanceDB) |

## Settings.yaml Changes

Replace the storage and vector_store sections in `backend/settings.yaml`:

```yaml
input:
  storage:
    type: blob
    connection_string: ${AZURE_STORAGE_CONNECTION_STRING}
    container_name: "gtog-input"
  file_type: text

output:
  type: blob
  connection_string: ${AZURE_STORAGE_CONNECTION_STRING}
  container_name: "gtog-output"

cache:
  type: blob
  connection_string: ${AZURE_STORAGE_CONNECTION_STRING}
  container_name: "gtog-cache"

reporting:
  type: blob
  connection_string: ${AZURE_STORAGE_CONNECTION_STRING}
  container_name: "gtog-logs"

vector_store:
  default_vector_store:
    type: azure_ai_search
    url: ${AZURE_SEARCH_ENDPOINT}
    api_key: ${AZURE_SEARCH_API_KEY}
    index_schema:
      vector_size: 768  # gemini-embedding-001 output dimension
```

## Environment Variables Required

```
GRAPHRAG_API_KEY=<gemini api key>
AZURE_STORAGE_CONNECTION_STRING=<from storage account>
AZURE_SEARCH_ENDPOINT=<https://srch-gtog-prod.search.windows.net>
AZURE_SEARCH_API_KEY=<from ai search>
```

## Notes

- Azure AI Search Free tier: 1 index, 50MB limit. Upgrade to Basic ($73/mo) when multiple collections needed.
- Storage Account name `stgtogprod` must be globally unique — script will verify.
- `vector_size: 768` matches `gemini-embedding-001` output dimensions.
- The backend `storage/collections/` local directory is NOT migrated in this phase — that is a backend API concern for Phase 2.
