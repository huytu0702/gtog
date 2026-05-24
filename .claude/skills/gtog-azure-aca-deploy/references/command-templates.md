# Command Templates

These are the copy-paste command patterns that worked for this repo on Windows.

## 1. Preflight

```bash
az account show --output json
docker version --format '{{.Server.Version}}'
```

## 2. Git Bash shell safety for repo scripts

```bash
export AZURE_CONFIG_DIR="C:/Users/DELL/.azure"
export MSYS2_ARG_CONV_EXCL='*'
```

## 3. Create ACR

```bash
az acr create \
  --resource-group "rg-gtog-prod" \
  --name "acrgtogprod22028126" \
  --sku Basic \
  --admin-enabled true \
  --location "southeastasia"
```

## 4. Provision data services

```bash
STORAGE_ACCOUNT="stgtog22028126" \
SEARCH_SERVICE="srch-gtog-22028126" \
COSMOS_ACCOUNT="cdb-gtog-22028126" \
RESOURCE_GROUP="rg-gtog-prod" \
LOCATION="southeastasia" \
SUBSCRIPTION="1095803e-80bf-47e0-961f-3d74cb4c605c" \
bash "scripts/provision-azure-db.sh"
```

## 5. Apply hardening and Key Vault bootstrap

```bash
RESOURCE_GROUP="rg-gtog-prod" \
LOCATION="southeastasia" \
SUBSCRIPTION="1095803e-80bf-47e0-961f-3d74cb4c605c" \
STORAGE_ACCOUNT="stgtog22028126" \
SEARCH_SERVICE="srch-gtog-22028126" \
COSMOS_ACCOUNT="cdb-gtog-22028126" \
KEY_VAULT_NAME="kvgtog22028126" \
MANAGED_IDENTITY_NAME="mi-gtog-backend" \
LOG_ANALYTICS_WORKSPACE="law-gtog-prod" \
bash "scripts/harden-azure-phase5.sh"
```

If Key Vault RBAC blocks secret writes, switch to access policies:

```bash
az keyvault update --name "kvgtog22028126" --resource-group "rg-gtog-prod" --enable-rbac-authorization false
az keyvault set-policy --name "kvgtog22028126" --resource-group "rg-gtog-prod" --upn "22028126@vnu.edu.vn" --secret-permissions get list set delete recover backup restore purge
az keyvault set-policy --name "kvgtog22028126" --resource-group "rg-gtog-prod" --object-id "3ef7d610-11d5-42c0-acff-8cc5703133b6" --secret-permissions get list
```

## 6. Build and push images

```bash
az acr login --name "acrgtogprod22028126"

docker build -f backend/Dockerfile \
  -t "acrgtogprod22028126.azurecr.io/gtog-backend:manual-<timestamp>" .

docker build --build-arg NEXT_PUBLIC_API_BASE_URL="https://api.gtog.id.vn" \
  -f frontend/Dockerfile \
  -t "acrgtogprod22028126.azurecr.io/gtog-frontend:manual-<timestamp>" frontend

docker push "acrgtogprod22028126.azurecr.io/gtog-backend:manual-<timestamp>"
docker push "acrgtogprod22028126.azurecr.io/gtog-frontend:manual-<timestamp>"
```

## 7. Create container apps from PowerShell

Use PowerShell for these commands on Windows to avoid identity ID mangling.

```powershell
$acrUser = az acr credential show --name "acrgtogprod22028126" --query username --output tsv
$acrPass = az acr credential show --name "acrgtogprod22028126" --query passwords[0].value --output tsv
$identityId = az identity show --name "mi-gtog-backend" --resource-group "rg-gtog-prod" --query id --output tsv

az containerapp create --resource-group "rg-gtog-prod" --name "ca-gtog-frontend-prod" --environment "cae-gtog-prod" --image "acrgtogprod22028126.azurecr.io/gtog-frontend:manual-<timestamp>" --registry-server "acrgtogprod22028126.azurecr.io" --registry-username $acrUser --registry-password $acrPass --user-assigned $identityId --ingress internal --target-port 3000 --transport auto --cpu 1.0 --memory 2.0Gi --min-replicas 1 --max-replicas 2 --env-vars "NEXT_PUBLIC_API_BASE_URL=https://api.gtog.id.vn" "CORS_ORIGINS=https://app.gtog.id.vn"

az containerapp create --resource-group "rg-gtog-prod" --name "ca-gtog-api-prod" --environment "cae-gtog-prod" --image "acrgtogprod22028126.azurecr.io/gtog-backend:manual-<timestamp>" --registry-server "acrgtogprod22028126.azurecr.io" --registry-username $acrUser --registry-password $acrPass --user-assigned $identityId --ingress internal --target-port 8000 --transport auto --cpu 1.0 --memory 2.0Gi --min-replicas 1 --max-replicas 2 --env-vars "APP_ROLE=api" "CORS_ORIGINS=https://app.gtog.id.vn" "REQUIRE_EDGE_AUTH=true"

az containerapp create --resource-group "rg-gtog-prod" --name "ca-gtog-worker-prod" --environment "cae-gtog-prod" --image "acrgtogprod22028126.azurecr.io/gtog-backend:manual-<timestamp>" --registry-server "acrgtogprod22028126.azurecr.io" --registry-username $acrUser --registry-password $acrPass --user-assigned $identityId --cpu 1.0 --memory 2.0Gi --min-replicas 1 --max-replicas 1 --env-vars "APP_ROLE=worker"

az containerapp create --resource-group "rg-gtog-prod" --name "ca-gtog-tunnel-prod" --environment "cae-gtog-prod" --image "cloudflare/cloudflared:latest" --cpu 0.5 --memory 1.0Gi --min-replicas 2 --max-replicas 2 --secrets "tunnel-token=<token>" --env-vars "TUNNEL_TOKEN=secretref:tunnel-token"
```

## 8. Update API/worker runtime env vars

```powershell
az containerapp update --resource-group "rg-gtog-prod" --name "ca-gtog-api-prod" --set-env-vars "APP_ROLE=api" "AZURE_USE_MANAGED_IDENTITY=true" "AZURE_MANAGED_IDENTITY_CLIENT_ID=f1dead4c-e004-478e-b548-3eaed7fcff93" "AZURE_KEY_VAULT_URL=https://kvgtog22028126.vault.azure.net/" "AZURE_STORAGE_ACCOUNT_NAME=stgtog22028126" "AZURE_STORAGE_QUEUE_NAME=indexing-jobs" "AZURE_SEARCH_ENDPOINT=https://srch-gtog-22028126.search.windows.net" "AZURE_COSMOS_ENDPOINT=https://cdb-gtog-22028126.documents.azure.com:443/" "AZURE_COSMOS_DATABASE_NAME=gtog-control"
```

Repeat for the worker app and include all required container-name env vars.

## 9. Public validation

```bash
curl -i "https://api.gtog.id.vn/api/collections" -H "Origin: https://app.gtog.id.vn"
az containerapp logs show --resource-group "rg-gtog-prod" --name "ca-gtog-api-prod" --tail 80
az containerapp revision list --resource-group "rg-gtog-prod" --name "ca-gtog-api-prod" --output table
```
