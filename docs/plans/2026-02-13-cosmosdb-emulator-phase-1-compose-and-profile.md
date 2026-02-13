# CosmosDB Emulator Phase 1 (Compose + Profile Plumbing) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a deterministic local Cosmos emulator runtime path with one full-stack compose entrypoint and explicit backend profile selection for Cosmos mode.

**Architecture:** Introduce a single `docker-compose.dev.yaml` as the source of truth for local orchestration (Cosmos emulator + backend + frontend), plus a dedicated backend GraphRAG settings profile for Cosmos emulator. Keep existing default file-based behavior untouched by adding opt-in environment-driven profile selection in backend config/helpers.

**Tech Stack:** Docker Compose, Azure Cosmos DB Emulator, FastAPI (backend), Next.js (frontend), GraphRAG YAML config.

---

### Task 1: Add full-stack compose file for Cosmos emulator development

**Files:**
- Create: `docker-compose.dev.yaml`
- Create: `.env.cosmos-emulator.example`
- Test: `docker compose -f docker-compose.dev.yaml config`

**Step 1: Write the failing test (configuration validation command)**

```bash
docker compose -f docker-compose.dev.yaml config
```

Expected: FAIL with "no such file or directory" for `docker-compose.dev.yaml`.

**Step 2: Create `docker-compose.dev.yaml` with all services**

```yaml
services:
  cosmos-emulator:
    image: mcr.microsoft.com/cosmosdb/linux/azure-cosmos-emulator:latest
    container_name: gtog-cosmos-emulator
    ports:
      - "8081:8081"
      - "10251:10251"
      - "10252:10252"
      - "10253:10253"
      - "10254:10254"
    environment:
      AZURE_COSMOS_EMULATOR_PARTITION_COUNT: ${COSMOS_PARTITION_COUNT:-5}
      AZURE_COSMOS_EMULATOR_ENABLE_DATA_PERSISTENCE: ${COSMOS_ENABLE_PERSISTENCE:-true}
    volumes:
      - cosmos_emulator_data:/data
    healthcheck:
      test: ["CMD", "curl", "-f", "https://localhost:8081/_explorer/index.html"]
      interval: 15s
      timeout: 10s
      retries: 20
      start_period: 60s

  backend:
    build:
      context: .
      dockerfile: backend/Dockerfile
    container_name: gtog-backend
    env_file:
      - .env
      - .env.cosmos-emulator
    environment:
      STORAGE_MODE: cosmos
      GRAPHRAG_SETTINGS_FILE: /app/backend/settings.cosmos-emulator.yaml
      STORAGE_ROOT_DIR: /app/backend/storage
    volumes:
      - ./:/app
    working_dir: /app/backend
    command: ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
    ports:
      - "8000:8000"
    depends_on:
      cosmos-emulator:
        condition: service_healthy

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    container_name: gtog-frontend
    environment:
      NEXT_PUBLIC_API_BASE_URL: http://localhost:8000/api
    ports:
      - "3000:3000"
    depends_on:
      - backend

volumes:
  cosmos_emulator_data:
```

**Step 3: Create `.env.cosmos-emulator.example`**

```env
# Cosmos emulator endpoint + key
COSMOS_ENDPOINT=https://cosmos-emulator:8081
COSMOS_KEY=C2y6yDjf5/R+ob0N8A7Cgv30VRDJIWEHLM+ZLw==
COSMOS_DATABASE=gtog
COSMOS_CONTAINER=graphrag

# Optional emulator tuning
COSMOS_PARTITION_COUNT=5
COSMOS_ENABLE_PERSISTENCE=true

# Backend mode selection
STORAGE_MODE=cosmos
GRAPHRAG_SETTINGS_FILE=./settings.cosmos-emulator.yaml

# Frontend
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000/api
```

**Step 4: Run validation command to verify pass**

Run:

```bash
docker compose -f docker-compose.dev.yaml config
```

Expected: PASS and renders merged service definitions.

**Step 5: Commit**

```bash
git add docker-compose.dev.yaml .env.cosmos-emulator.example
git commit -m "feat: add full-stack compose for cosmos emulator development"
```

---

### Task 2: Add backend Dockerfile for compose backend service

**Files:**
- Create: `backend/Dockerfile`
- Test: `docker compose -f docker-compose.dev.yaml build backend`

**Step 1: Write the failing test (build backend image)**

Run:

```bash
docker compose -f docker-compose.dev.yaml build backend
```

Expected: FAIL with "Cannot locate Dockerfile: backend/Dockerfile".

**Step 2: Write minimal backend Dockerfile**

```dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml /app/pyproject.toml
COPY backend/requirements.txt /app/backend/requirements.txt

RUN pip install --no-cache-dir -r /app/backend/requirements.txt \
    && pip install --no-cache-dir -e /app

COPY . /app

WORKDIR /app/backend
```

**Step 3: Run build test to verify it passes**

Run:

```bash
docker compose -f docker-compose.dev.yaml build backend
```

Expected: PASS, backend image builds successfully.

**Step 4: Commit**

```bash
git add backend/Dockerfile
git commit -m "chore: add backend container image for local compose stack"
```

---

### Task 3: Add dedicated Cosmos GraphRAG settings profile

**Files:**
- Create: `backend/settings.cosmos-emulator.yaml`
- Modify: `backend/settings.yaml` (optional comment-only pointer to new profile)
- Test: `python -c "from graphrag.config.load_config import load_config; load_config(root_dir='backend', config_filepath='backend/settings.cosmos-emulator.yaml')"`

**Step 1: Write the failing test (load Cosmos profile)**

Run:

```bash
python -c "from graphrag.config.load_config import load_config; load_config(root_dir='backend', config_filepath='backend/settings.cosmos-emulator.yaml')"
```

Expected: FAIL because file does not exist.

**Step 2: Create `backend/settings.cosmos-emulator.yaml`**

Use current `backend/settings.yaml` as base, but set Cosmos-enabled sections:

```yaml
output:
  type: cosmosdb
  base_dir: ${COSMOS_DATABASE}
  connection_string: "AccountEndpoint=${COSMOS_ENDPOINT};AccountKey=${COSMOS_KEY};"
  container_name: ${COSMOS_CONTAINER}

cache:
  type: cosmosdb
  base_dir: ${COSMOS_DATABASE}
  connection_string: "AccountEndpoint=${COSMOS_ENDPOINT};AccountKey=${COSMOS_KEY};"
  container_name: ${COSMOS_CONTAINER}

vector_store:
  default_vector_store:
    type: cosmosdb
    url: ${COSMOS_ENDPOINT}
    database_name: ${COSMOS_DATABASE}
    container_name: ${COSMOS_CONTAINER}
    overwrite: true
    embeddings_schema:
      id:
        id_field: id
        vector_field: vector
        text_field: text
        metadata_fields: [title]
```

Keep model/query sections aligned with `backend/settings.yaml`.

**Step 3: Run config load test**

Run:

```bash
python -c "from graphrag.config.load_config import load_config; load_config(root_dir='backend', config_filepath='backend/settings.cosmos-emulator.yaml')"
```

Expected: PASS.

**Step 4: Commit**

```bash
git add backend/settings.cosmos-emulator.yaml backend/settings.yaml
git commit -m "feat: add cosmos emulator graphrag settings profile"
```

---

### Task 4: Add explicit backend settings profile selection

**Files:**
- Modify: `backend/app/config.py`
- Test: `backend/tests/unit/test_config.py` (new)
- Create: `backend/tests/unit/test_config.py`

**Step 1: Write failing tests for profile path behavior**

```python
from backend.app.config import Settings


def test_settings_yaml_path_defaults_to_settings_yaml(monkeypatch):
    monkeypatch.delenv("GRAPHRAG_SETTINGS_FILE", raising=False)
    s = Settings()
    assert s.settings_yaml_path.name == "settings.yaml"


def test_settings_yaml_path_uses_env_override(monkeypatch):
    monkeypatch.setenv("GRAPHRAG_SETTINGS_FILE", "./settings.cosmos-emulator.yaml")
    s = Settings()
    assert s.settings_yaml_path.name == "settings.cosmos-emulator.yaml"
```

**Step 2: Run tests and verify fail**

Run:

```bash
pytest backend/tests/unit/test_config.py -v
```

Expected: FAIL (new field/property not implemented).

**Step 3: Implement minimal config support**

Update `backend/app/config.py`:

```python
# add field
settings_file: str = "settings.yaml"

@property
def settings_yaml_path(self) -> Path:
    path = Path(self.settings_file)
    if path.is_absolute():
        return path
    return Path(__file__).parent.parent / path
```

Ensure env name `GRAPHRAG_SETTINGS_FILE` maps to `settings_file` automatically.

**Step 4: Run tests again**

Run:

```bash
pytest backend/tests/unit/test_config.py -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add backend/app/config.py backend/tests/unit/test_config.py
git commit -m "feat: support explicit backend graphrag settings profile selection"
```

---

### Task 5: Gate file-only overrides behind storage mode

**Files:**
- Modify: `backend/app/utils/helpers.py`
- Create: `backend/tests/unit/test_helpers_config_overrides.py`
- Test: `backend/tests/unit/test_helpers_config_overrides.py`

**Step 1: Write failing tests for storage mode behavior**

```python
from backend.app.utils.helpers import load_graphrag_config


def test_load_graphrag_config_file_mode_applies_file_overrides(monkeypatch):
    monkeypatch.setenv("STORAGE_MODE", "file")
    cfg = load_graphrag_config("demo")
    assert cfg.output.type == "file"
    assert cfg.cache.type == "file"


def test_load_graphrag_config_cosmos_mode_does_not_force_file_overrides(monkeypatch):
    monkeypatch.setenv("STORAGE_MODE", "cosmos")
    cfg = load_graphrag_config("demo")
    assert cfg.output.type != "file" or cfg.cache.type != "file"
```

(Use mocking around `load_config` if needed to avoid external dependencies.)

**Step 2: Run tests to confirm failure**

Run:

```bash
pytest backend/tests/unit/test_helpers_config_overrides.py -v
```

Expected: FAIL.

**Step 3: Implement minimal mode-gated overrides**

In `backend/app/utils/helpers.py`:

- Add helper:

```python
def _is_cosmos_mode() -> bool:
    return (settings.storage_mode or "file").strip().lower() == "cosmos"
```

- In `load_graphrag_config`, only pass current file overrides when NOT cosmos mode.
- In cosmos mode, pass only root/config path and retain profile values.

**Step 4: Re-run tests**

Run:

```bash
pytest backend/tests/unit/test_helpers_config_overrides.py -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add backend/app/utils/helpers.py backend/tests/unit/test_helpers_config_overrides.py
git commit -m "refactor: make graphrag file overrides conditional on storage mode"
```

---

### Task 6: Add env template entries and startup docs for profile switching

**Files:**
- Modify: `backend/.env.example`
- Modify: `backend/README.md`
- Test: manual doc sanity check

**Step 1: Write failing check (missing env keys in template)**

Run:

```bash
python -c "from pathlib import Path; t=Path('backend/.env.example').read_text(); assert 'GRAPHRAG_SETTINGS_FILE' in t and 'STORAGE_MODE' in t"
```

Expected: FAIL.

**Step 2: Add required env keys**

Append to `backend/.env.example`:

```env
# Storage mode selection: file | cosmos
STORAGE_MODE=file

# Relative to backend/ when not absolute
GRAPHRAG_SETTINGS_FILE=settings.yaml

# Cosmos emulator vars (used by settings.cosmos-emulator.yaml)
COSMOS_ENDPOINT=https://localhost:8081
COSMOS_KEY=C2y6yDjf5/R+ob0N8A7Cgv30VRDJIWEHLM+ZLw==
COSMOS_DATABASE=gtog
COSMOS_CONTAINER=graphrag
```

**Step 3: Update backend README run modes**

Add section in `backend/README.md`:
- File mode command
- Cosmos mode command (with `.env.cosmos-emulator`)
- Compose commands:
  - full stack: `docker compose -f docker-compose.dev.yaml up --build`
  - cosmos-only quick run: `docker compose -f docker-compose.dev.yaml up cosmos-emulator`

**Step 4: Re-run env key check**

Run:

```bash
python -c "from pathlib import Path; t=Path('backend/.env.example').read_text(); assert 'GRAPHRAG_SETTINGS_FILE' in t and 'STORAGE_MODE' in t"
```

Expected: PASS.

**Step 5: Commit**

```bash
git add backend/.env.example backend/README.md
git commit -m "docs: document storage mode and cosmos profile configuration"
```

---

### Task 7: Verify Phase 1 end-to-end startup behavior

**Files:**
- Test only (no code changes required unless failures found)

**Step 1: Bring up full stack**

Run:

```bash
docker compose -f docker-compose.dev.yaml up --build -d
```

Expected: cosmos-emulator, backend, frontend all healthy/running.

**Step 2: Verify backend health endpoint**

Run:

```bash
curl http://localhost:8000/health
```

Expected: HTTP 200 with `{"status":"healthy","version":"1.0.0"}`.

**Step 3: Verify frontend reachable**

Run:

```bash
curl http://localhost:3000
```

Expected: HTTP 200.

**Step 4: Verify cosmos-only quick run path**

Run:

```bash
docker compose -f docker-compose.dev.yaml down
docker compose -f docker-compose.dev.yaml up -d cosmos-emulator
```

Expected: cosmos emulator starts independently using same compose file.

**Step 5: Final check commands**

Run:

```bash
pytest backend/tests/unit/test_config.py backend/tests/unit/test_helpers_config_overrides.py -v
```

Expected: PASS.

**Step 6: Commit verification notes (if scripts/docs changed)**

```bash
git add docs/plans/2026-02-13-cosmosdb-emulator-phase-1-compose-and-profile.md
git commit -m "docs: add phase 1 implementation plan for cosmos compose and profile plumbing"
```

---

## Phase 1 Completion Checklist

- [ ] Single `docker-compose.dev.yaml` exists and validates.
- [ ] Backend and frontend can run from compose.
- [ ] `backend/settings.cosmos-emulator.yaml` exists and loads.
- [ ] Backend config supports explicit profile switching (`GRAPHRAG_SETTINGS_FILE`).
- [ ] Helpers stop forcing file overrides in cosmos mode.
- [ ] `.env` templates and README document both file and cosmos modes.
- [ ] Targeted unit tests pass.

---

## Notes for Executor

- Use @superpowers:test-driven-development before changing code.
- Use @superpowers:verification-before-completion before claiming task completion.
- Keep commits small and frequent (one task = one commit where practical).
- Do not expand scope into repository abstraction or query-path migration in this phase (those are later phases).
