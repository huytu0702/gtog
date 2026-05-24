# Current Production Defaults

Use these values unless the user explicitly asks for a new environment.

## Azure

- Subscription: `1095803e-80bf-47e0-961f-3d74cb4c605c`
- Resource group: `rg-gtog-prod`
- Location: `southeastasia`
- ACR: `acrgtogprod22028126`
- ACA environment: `cae-gtog-prod`
- ACA infra resource group: `rg-gtog-prod-aca-infra`
- Log Analytics workspace: `law-gtog-prod`
- VNet: `vnet-gtog-prod-aca`
- Managed identity: `mi-gtog-backend`
- Managed identity client id: `f1dead4c-e004-478e-b548-3eaed7fcff93`
- Key Vault: `kvgtog22028126`
- Key Vault URL: `https://kvgtog22028126.vault.azure.net/`

## Data services

- Storage account: `stgtog22028126`
- Queue name: `indexing-jobs`
- Search service: `srch-gtog-22028126`
- Search endpoint: `https://srch-gtog-22028126.search.windows.net`
- Cosmos account: `cdb-gtog-22028126`
- Cosmos endpoint: `https://cdb-gtog-22028126.documents.azure.com:443/`
- Cosmos database: `gtog-control`

## ACA apps

- Frontend app: `ca-gtog-frontend-prod`
- API app: `ca-gtog-api-prod`
- Worker app: `ca-gtog-worker-prod`
- Tunnel app: `ca-gtog-tunnel-prod`

## Current ACA private origins

- Frontend origin: `https://ca-gtog-frontend-prod.internal.ashyisland-27b3f981.southeastasia.azurecontainerapps.io`
- API origin: `https://ca-gtog-api-prod.internal.ashyisland-27b3f981.southeastasia.azurecontainerapps.io`

## Public hostnames

- Frontend: `https://app.gtog.id.vn`
- API: `https://api.gtog.id.vn`

## Key Vault secret names in use

- `storage-connection-string`
- `storage-account-key`
- `search-api-key`
- `cosmos-connection-string`
- `cosmos-key`
- `graphrag-api-key`
- `google-api-key`
- `tavily-api-key`
- `edge-origin-secret`

## Known-good image tags from this session

- Backend initial deploy: `manual-20260317-0445`
- Backend with trusted-tunnel fix: `manual-20260317-0548`
- Frontend deploy: `manual-20260317-0445`
