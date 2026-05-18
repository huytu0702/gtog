---
name: gtog-azure-aca-deploy
description: Deploy and update the GraphRAG ToG repo at `F:\KL\gtog` to Azure Container Apps behind a Cloudflare Tunnel private-origin setup. Use this whenever the user wants to provision Azure from scratch, rebuild and push backend/frontend images, wire Storage/Cosmos/Search/Key Vault/managed identity, reconcile ACA apps, or troubleshoot `app.gtog.id.vn` / `api.gtog.id.vn` deployment and Cloudflare 403 issues.
---

# GraphRAG Azure ACA Deploy

Use this skill for the `F:\KL\gtog` repository and its Azure + Cloudflare topology.

Read `references/current-production.md` first for the current production resource names and hostnames. Read `references/command-templates.md` when you need copy-paste commands.

## What this skill covers

- Full Azure bootstrap for this repo: resource group, ACR, Storage, Cosmos, Search, Key Vault, managed identity, ACA environment, and apps.
- Code deployment from the local repo by building Docker images and pushing them to ACR.
- ACA runtime configuration for `api`, `worker`, `frontend`, and `cloudflared` tunnel apps.
- Cloudflare private-origin cutover checks and the common `403 Forbidden` failure mode.

## Default operating rules

- Treat the repo as container-based deployment only. Do not use source-zip deploy flows like `az webapp up`.
- Prefer the repo's existing scripts for Azure resource provisioning, but use direct `az containerapp` commands when Windows shell quirks make the scripts unreliable.
- Never print secrets in the final response. Read secrets from `backend/.env` only when needed to seed Azure, and do not commit them.
- Do not revert unrelated user changes in the repo.

## Windows-specific rules

- When running the repo's `.sh` Azure scripts from Windows Git Bash, set:

```bash
export AZURE_CONFIG_DIR="C:/Users/DELL/.azure"
export MSYS2_ARG_CONV_EXCL='*'
```

- This prevents two real failures seen in this repo:
  - Git Bash path-mangling of Cosmos partition keys like `/collectionId`
  - loss of Azure CLI session visibility inside shell-script executions
- Prefer PowerShell for `az containerapp create` and `az containerapp update` on Windows. Git Bash may corrupt `--user-assigned` identity resource IDs.

## Workflow

### 1. Preflight

- Confirm Azure login and active subscription with `az account show`.
- Confirm Docker is running.
- Read `backend/.env` only if you need to seed app/API secrets into Key Vault or ACA secrets.
- Use the values from `references/current-production.md` unless the user explicitly wants a different environment.

### 2. Provision Azure foundation

- Ensure the resource group exists.
- Create or reuse ACR.
- Run `scripts/provision-azure-db.sh` to create Storage, Queue, Search, Cosmos, database, and containers.
- Run `scripts/harden-azure-phase5.sh` to create managed identity, Key Vault, alerts, and baseline hardening.

If Key Vault secret writes fail with RBAC errors on this subscription shape:

- switch the vault from RBAC to access-policy mode
- grant the signed-in user secret `get/list/set/delete/recover/backup/restore/purge`
- grant the managed identity secret `get/list`
- rerun the hardening script

### 3. Build and push images

- Login to ACR.
- Build backend from repo root using `backend/Dockerfile`.
- Build frontend using `frontend/Dockerfile` and pass `NEXT_PUBLIC_API_BASE_URL=https://api.gtog.id.vn`.
- Push both images to ACR.
- Reuse the backend image for the worker app.

### 4. Provision or reconcile ACA apps

Create or update these apps in the ACA environment:

- `frontend`: internal ingress, target port `3000`
- `api`: internal ingress, target port `8000`
- `worker`: no ingress
- `tunnel`: no ingress, image `cloudflare/cloudflared:latest`

For the tunnel app:

- store the tunnel token as an ACA secret
- run `cloudflared tunnel --no-autoupdate run`

### 5. Apply runtime configuration

For `api` and `worker`, set:

- `APP_ROLE=api|worker`
- `AZURE_USE_MANAGED_IDENTITY=true`
- `AZURE_MANAGED_IDENTITY_CLIENT_ID=<mi-client-id>`
- `AZURE_KEY_VAULT_URL=<vault-url>`
- Key Vault secret-name env vars for storage/search/cosmos/app keys
- non-secret Azure endpoints and container names

Also set ACA app secrets for:

- `GRAPHRAG_API_KEY`
- `GOOGLE_API_KEY`
- `TAVILY_API_KEY`

Do not assume Key Vault alone is enough unless the repo's runtime env names are already mapped correctly.

### 6. Cloudflare cutover

- Update the published routes for `app.gtog.id.vn` and `api.gtog.id.vn` to the current ACA internal FQDNs.
- Set both `HTTP Host Header` and `Origin Server Name` to the exact internal ACA hostname.
- For the current repo state, remote-managed Cloudflare Tunnel traffic should work without a custom `X-Edge-Secret` header because the backend now trusts Cloudflare tunnel overlay requests from the private proxy network.

If the public API still returns `403` while direct requests with a manual edge secret succeed:

- verify the deployed backend image includes the trusted-tunnel proxy fix in `backend/app/main.py`
- redeploy the API image if needed

### 7. Validate

- Check ACA revision health and logs.
- Validate the public API:

```bash
curl -i "https://api.gtog.id.vn/api/collections" -H "Origin: https://app.gtog.id.vn"
```

- Expected result: `200 OK` with CORS headers for `https://app.gtog.id.vn`.
- Validate the frontend in a browser and confirm collections load.

## Troubleshooting cues

- `Cosmos partition key path became C:/Program Files/Git/...`: missing `MSYS2_ARG_CONV_EXCL='*'`
- `az login` works outside script but not inside `.sh`: set `AZURE_CONFIG_DIR="C:/Users/DELL/.azure"`
- `InvalidIdentityId` during ACA create/update on Windows: switch to PowerShell for containerapp commands
- frontend loads but API calls return `403`: Cloudflare route/origin mismatch or old backend image without trusted tunnel proxy support
- backend startup fails with missing `TAVILY_API_KEY` or similar: seed ACA secrets and redeploy/restart

## Response pattern when using this skill

- Lead with whether the task is a full bootstrap, an image-only redeploy, or a Cloudflare/API troubleshooting pass.
- Mention which commands you ran and which Azure resources were created or updated.
- Return the active internal ACA origins and any Cloudflare route changes the user still needs to make.
