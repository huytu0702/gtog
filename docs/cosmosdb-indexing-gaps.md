# CosmosDB indexing alignment status

## Summary

Backend runtime is now aligned with direct Cosmos pipeline output for search.

- Indexing writes GraphRAG datasets to per-version containers: `pipeline-{collection}-{version}`.
- Query loads datasets directly from the active pipeline container via `collections.activeVersion`.
- Legacy serving-container layer (`entities`, `relationships`, `textUnits`, `communities`, `communityReports`, `covariates`) is deprecated and not part of runtime search flow.

## Current authoritative flow

1. `load_graphrag_config(...)` sets `output.type=cosmosdb` and `output.container_name=pipeline-{collection}-{version}`.
2. Indexing writes parquet datasets into that pipeline container.
3. `IndexingService` verifies required datasets, writes `artifactManifest` (`artifactName=pipeline-datasets`), then flips `collections.activeVersion`.
4. `QueryService` reads required datasets directly from the active pipeline container.

## Operational policy

- Keep control-plane containers: `collections`, `documents`, `indexingJobs`, `jobEvents`, `artifactManifest`.
- Keep conversation containers: `conversationSessions`, `conversationTurns`.
- Keep active/retained `pipeline-*` containers.
- Keep vector containers used by Cosmos vector store.
- Delete legacy serving containers only after deploy + verification + backup.
