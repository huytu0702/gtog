```mermaid
flowchart TB
  U[User Browser] --> FD[Azure Front Door - WAF - Routing - Rate Limit]

  FD -->|All paths| FE[ACA Frontend - Next.js - Easy Auth]
  FD -->|API paths - public| BE[ACA Backend - FastAPI - Easy Auth - internal ingress only]

  subgraph EASYAUTH[ACA Managed Authentication - Easy Auth]
    EA1[Frontend: unauthenticated → redirect to Entra login]
    EA2[Backend: unauthenticated → return 401]
    EA3[Both: token validated at platform level before app receives request]
    EA4[Both: user identity injected as X-MS-CLIENT-PRINCIPAL headers]
    EA1 --- EA2 --- EA3 --- EA4
  end

  U -->|1 - visit app| FE
  FE -->|2 - Easy Auth redirects to Entra login| ENTRA[Microsoft Entra ID]
  ENTRA -->|3 - tokens returned to Easy Auth| FE
  U -->|4 - GET /.auth/me to get access token| FE
  U -->|5 - call /api/* via Front Door with Bearer token| FD

  FD -.->|Inject header X-AFD-Secret| BE
  BE --> LOCK[Origin Lock Check - deny if header missing]

  BE --> KV[Azure Key Vault - BE Managed Identity]
  BE --> COSMOS[Azure Cosmos DB]
  BE --> BLOB[Azure Blob Storage]
  BE --> SEARCH[Azure AI Search - vector store]
  BE --> LLM[Gemini or OpenAI APIs]
  BE --> TAVILY[Tavily API]

  FE --> LAW[Log Analytics and Azure Monitor Alerts]
  BE --> LAW

  subgraph CICD[Azure DevOps CI/CD]
    REPO[Git Repo] --> PIPE[Pipeline YAML]
    PIPE --> ACR[Azure Container Registry]
    ACR --> FE
    ACR --> BE
  end
```

**Notes:**
- Auth is handled entirely by **ACA Easy Auth** at the platform level — no auth code in Next.js or FastAPI.
- Frontend Easy Auth action: `RedirectToLoginPage`. Backend Easy Auth action: `Return401`.
- Browser calls `/.auth/me` (served by Easy Auth) to get the access token, then sends it as `Authorization: Bearer` to `/api/*` via Front Door.
- Backend has **internal ingress only** — only reachable via Front Door publicly. Easy Auth on the backend validates the Bearer token before FastAPI processes the request.
- `X-AFD-Secret` origin lock applies to all Front Door → backend traffic.
- Backend Managed Identity used for Key Vault access only. Frontend does not need a Managed Identity.
- Separate `AFD_ORIGIN_SECRET` values per environment (staging vs prod) in Key Vault.
