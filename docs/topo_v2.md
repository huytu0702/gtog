```mermaid
flowchart TB
  U[User Browser]

  U -->|Visit https://app.<domain>| CFAPP[Cloudflare Edge - app host]
  U -->|API and auth calls to https://api.<domain>| CFAPI[Cloudflare Edge - api host]

  CFAPP --> FE[ACA Frontend - Next.js - Public UI]
  CFAPI -->|"X-Edge-Secret + Cf-Ray"| BE[ACA Backend - FastAPI - Easy Auth]

  subgraph EDGE[Cloudflare Edge Controls]
    E1[Proxied DNS]
    E2[Rate limit on api host]
    E3[WAF or custom edge rules]
    E4[Cache bypass for API auth and SSE]
    E5[Request header transform injects X-Edge-Secret]
    E1 --- E2 --- E3 --- E4 --- E5
  end

  CFAPP --- EDGE
  CFAPI --- EDGE

  subgraph EASYAUTH[ACA Managed Authentication - Backend Only]
    EA1[Backend unauthenticated requests return 401]
    EA2[Easy Auth validates token before FastAPI receives request]
    EA3[User identity header injected: X-MS-CLIENT-PRINCIPAL]
    EA1 --- EA2 --- EA3
  end

  FE -->|Login redirect to api host| ENTRA[Microsoft Entra ID]
  ENTRA -->|Auth session + tokens on api host| BE
  FE -->|GET https://api.<domain>/.auth/me| BE
  FE -->|"GET /api/* with Bearer token"| CFAPI
  FE -->|"SSE /api/* with session cookie"| CFAPI

  BE --> LOCK[Origin Lock Check - deny if X-Edge-Secret missing]

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
- Cloudflare replaces Azure Front Door as the public edge because Front Door is not available in the target Azure Student setup.
- Public routing uses two hostnames, not one path-routed hostname:
  - `app.<domain>` -> frontend ACA
  - `api.<domain>` -> backend ACA
- Backend remains protected by layered controls: Cloudflare rate limiting/WAF, `X-Edge-Secret`, and Easy Auth on the backend.
- Frontend is public. Backend is the protected auth boundary.
- Frontend login/logout actions must target the backend auth host and redirect users back to `app.<domain>` after login/logout.
- Backend uses Managed Identity to read runtime secrets from Key Vault. Frontend does not use Managed Identity.
- `EDGE_ORIGIN_SECRET` must be different per environment (staging/prod).
- Data layer (Cosmos, Blob, Search, Key Vault + MI) is already provisioned from previous phases and must be reused.
- SSE endpoints (agent stream, indexing status) use session cookie auth because `EventSource` cannot set custom headers.
- SSE responses must emit heartbeat events every 25-30 seconds to avoid idle proxy timeout at the edge.
- Frontend image is built per environment (`frontend:<sha>-<env>`) because `NEXT_PUBLIC_` vars are inlined at build time.
- Backend image is environment-agnostic (runtime env vars only).
- Backend Easy Auth must have `allowedAudiences` including `api://<backend-app-id>` for token validation.
- Backend should log `Cf-Ray` for request correlation and `CF-Connecting-IP` as the original client IP when available.
- ACA custom domains should use uploaded certificates in this topology. Do not rely on ACA managed certificates behind Cloudflare proxying.
