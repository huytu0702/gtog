```mermaid
flowchart TB
  U[User Browser] --> FD[Azure Front Door - WAF - Routing - Rate Limit]

  FD -->|"/* (HTML/JS)"| FE[ACA Frontend - Next.js - Easy Auth]
  FD -->|"/api/* + X-AFD-Secret"| BE[ACA Backend - FastAPI - Easy Auth]

  subgraph EASYAUTH[ACA Managed Authentication - Easy Auth]
    EA1[Frontend: unauthenticated -> redirect to Entra login]
    EA2[Backend: unauthenticated -> return 401]
    EA3[Both: token validated at platform level before app receives request]
    EA4[Both: user identity headers injected: X-MS-CLIENT-PRINCIPAL*]
    EA1 --- EA2 --- EA3 --- EA4
  end

  U -->|1 - visit app| FE
  FE -->|2 - Easy Auth redirect| ENTRA[Microsoft Entra ID]
  ENTRA -->|3 - auth session + tokens| FE
  U -->|4 - GET /.auth/me| FE
  U -->|"5a - /api/* with Bearer token"| FD
  U -->|"5b - SSE /api/* with session cookie"| FD

  FD -.->|Inject X-AFD-Secret + X-Azure-Ref| BE
  BE --> LOCK[Origin Lock Check - deny if header missing]

  subgraph DATALAYER[Data Layer - already deployed]
    KV[Azure Key Vault]
    MI[User-assigned Managed Identity for backend]
    COSMOS[Azure Cosmos DB]
    BLOB[Azure Blob Storage]
    SEARCH[Azure AI Search]
  end

  BE --> MI
  MI --> KV
  BE --> COSMOS
  BE --> BLOB
  BE --> SEARCH
  BE --> LLM[Gemini or OpenAI APIs]
  BE --> TAVILY[Tavily API]

  FE --> LAW[Log Analytics and Azure Monitor Alerts]
  BE --> LAW

  subgraph CICD[Azure DevOps CI/CD]
    REPO[Git Repo] --> PIPE[Pipeline YAML]
    PIPE --> ACR[Azure Container Registry]
    ACR -->|"frontend:<sha>-<env>"| FE
    ACR -->|"backend:<sha>"| BE
  end
```

**Notes:**
- Auth is handled by ACA Easy Auth at the platform level (no custom JWT validation in app code).
- Frontend Easy Auth action: `RedirectToLoginPage`. Backend Easy Auth action: `Return401`.
- Browser calls `/api/*` through Front Door. Front Door routes to backend origin and injects `X-AFD-Secret`.
- Backend is protected against bypass by layered controls: Front Door route + `X-AFD-Secret` check + Easy Auth on backend.
- Backend uses Managed Identity to read runtime secrets from Key Vault. Frontend does not use Managed Identity.
- `AFD_ORIGIN_SECRET` must be different per environment (staging/prod).
- Data layer (Cosmos, Blob, Search, Key Vault + MI) is already provisioned from previous phases and must be reused.
- SSE endpoints (agent stream, indexing status) use session cookie auth because `EventSource` cannot set custom headers.
- Front Door origin timeout for backend set to 240s to support long-running SSE streams.
- Frontend image is built per-environment (`frontend:<sha>-<env>`) because `NEXT_PUBLIC_` vars are inlined at build time.
- Backend image is environment-agnostic (runtime env vars only).
- Backend Easy Auth must have `allowedAudiences` including `api://<backend-app-id>` for token validation.
- Backend logs `X-Azure-Ref` header from Front Door for cross-layer request correlation.
