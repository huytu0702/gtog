# CosmosDB Emulator Phase 5 (DX + Verification Hardening) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Provide developer-friendly workflows (bootstrap, smoke, troubleshooting, reset) and harden validation so Cosmos emulator mode is reliable and easy to operate locally.

**Architecture:** Add small deterministic scripts for bootstrap/smoke/reset around existing compose profile and backend API flow. Expand integration coverage to include end-to-end API smoke, persistence across restarts, and expected failures (emulator down / bad config). Keep scripts simple and explicit, no hidden state.

**Tech Stack:** Python scripts (or shell scripts where already standard), Docker Compose, pytest integration tests, backend FastAPI endpoints.

---

### Task 1: Add bootstrap script for Cosmos local development

**Files:**
- Create: `scripts/cosmos/bootstrap.py`
- Create: `scripts/cosmos/__init__.py` (if package needed)
- Test: `python scripts/cosmos/bootstrap.py --help`

**Step 1: Write failing test/check**

Run:

```bash
python scripts/cosmos/bootstrap.py --help
```

Expected: FAIL (file missing).

**Step 2: Implement minimal bootstrap script**

Script responsibilities:
1. Validate required tools (`docker`, `docker compose`).
2. Validate required env file presence (`.env`, `.env.cosmos-emulator` optionally from example copy).
3. Run compose up command:

```bash
docker compose -f docker-compose.dev.yaml up -d cosmos-emulator backend frontend
```

4. Poll backend `/health` endpoint until healthy or timeout.
5. Print next-step commands.

**Step 3: Run help check**

Run:

```bash
python scripts/cosmos/bootstrap.py --help
```

Expected: PASS.

**Step 4: Commit**

```bash
git add scripts/cosmos/bootstrap.py scripts/cosmos/__init__.py
git commit -m "feat: add cosmos bootstrap script for local development"
```

---

### Task 2: Add deterministic smoke script for API happy path

**Files:**
- Create: `scripts/cosmos/smoke_api.py`
- Test: `python scripts/cosmos/smoke_api.py --help`

**Step 1: Write failing smoke script check**

Run:

```bash
python scripts/cosmos/smoke_api.py --help
```

Expected: FAIL.

**Step 2: Implement smoke flow script**

Flow (single command):
1. Create temporary collection (`POST /api/collections`).
2. Upload sample `.txt` document.
3. Start indexing and poll status until completed/failed.
4. Execute one query endpoint (local or global).
5. Print structured pass/fail summary.
6. Optional `--cleanup` to delete collection.

Use idempotent collection naming prefix like `smoke-<timestamp>`.

**Step 3: Run help command**

Run:

```bash
python scripts/cosmos/smoke_api.py --help
```

Expected: PASS.

**Step 4: Commit**

```bash
git add scripts/cosmos/smoke_api.py
git commit -m "feat: add end-to-end cosmos api smoke validation script"
```

---

### Task 3: Add explicit reset workflow for emulator data cleanup

**Files:**
- Create: `scripts/cosmos/reset.py`
- Modify: `backend/README.md`
- Test: `python scripts/cosmos/reset.py --help`

**Step 1: Write failing check**

Run:

```bash
python scripts/cosmos/reset.py --help
```

Expected: FAIL.

**Step 2: Implement reset script with safety prompt**

Behavior:
- Show warning that data will be removed.
- Require explicit `--yes` to execute destructive action.
- Execute:

```bash
docker compose -f docker-compose.dev.yaml down -v
```

- Optionally remove local backend storage mirror path if `--clean-local-storage` is passed.

**Step 3: Document reset workflow in README**

Add section:
- when to use reset,
- exact command,
- expected effect.

**Step 4: Re-run help check**

Run:

```bash
python scripts/cosmos/reset.py --help
```

Expected: PASS.

**Step 5: Commit**

```bash
git add scripts/cosmos/reset.py backend/README.md
git commit -m "feat: add explicit cosmos reset workflow with safety confirmation"
```

---

### Task 4: Add persistence verification integration test (restart scenario)

**Files:**
- Create: `backend/tests/integration/test_cosmos_persistence_restart.py`

**Step 1: Write failing integration test**

Scenario:
1. Start stack.
2. Create collection + upload doc.
3. Restart backend container only.
4. Verify collection/doc still listed.
5. Restart cosmos + backend (without volume delete).
6. Verify data still exists.

**Step 2: Run integration test to confirm fail**

Run:

```bash
pytest backend/tests/integration/test_cosmos_persistence_restart.py -v
```

Expected: FAIL initially.

**Step 3: Implement minimal fixes**

Potential fixes:
- ensure cosmos emulator uses persistent volume in compose.
- ensure backend cosmos repository uses deterministic database/container names from env.

**Step 4: Re-run integration test**

Run:

```bash
pytest backend/tests/integration/test_cosmos_persistence_restart.py -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add backend/tests/integration/test_cosmos_persistence_restart.py docker-compose.dev.yaml

git commit -m "test: verify cosmos-backed data persists across service restarts"
```

---

### Task 5: Add negative-path tests for emulator unavailable and bad config

**Files:**
- Create: `backend/tests/integration/test_cosmos_failure_modes.py`
- Modify (if needed): `backend/app/main.py`, `backend/app/services/storage_service.py`

**Step 1: Write failing tests**

Test cases:
1. Cosmos mode + missing endpoint/key => startup/config validation failure.
2. Cosmos mode + emulator down => API call returns clear 5xx with actionable message.
3. Cosmos mode + invalid key => initialization/query failure is explicit and not silently falling back.

**Step 2: Run tests and confirm fail**

Run:

```bash
pytest backend/tests/integration/test_cosmos_failure_modes.py -v
```

Expected: FAIL.

**Step 3: Implement minimal hardening**

- Ensure clear exception mapping in routers/services (no silent fallback to file mode).
- Add concise user-facing messages while logging full exception details.

**Step 4: Re-run tests**

Run:

```bash
pytest backend/tests/integration/test_cosmos_failure_modes.py -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add backend/tests/integration/test_cosmos_failure_modes.py backend/app/main.py backend/app/services/storage_service.py

git commit -m "test: cover cosmos emulator failure modes and explicit error behavior"
```

---

### Task 6: Add frontend API base URL env support for compose friendliness

**Files:**
- Modify: `frontend/lib/api.ts`
- Modify: `frontend/.env.example` (create if missing)
- Test: `npm --prefix frontend run lint`

**Step 1: Write failing test/check**

Add a small unit/static check (or use direct assertion command):

```bash
python -c "from pathlib import Path; t=Path('frontend/lib/api.ts').read_text(); assert 'NEXT_PUBLIC_API_BASE_URL' in t"
```

Expected: FAIL (currently hardcoded URL).

**Step 2: Implement minimal env-based base URL**

Change:

```ts
const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? 'http://127.0.0.1:8000/api';
```

Add/Update `frontend/.env.example`:

```env
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000/api
```

**Step 3: Run checks**

Run:

```bash
python -c "from pathlib import Path; t=Path('frontend/lib/api.ts').read_text(); assert 'NEXT_PUBLIC_API_BASE_URL' in t"
npm --prefix frontend run lint
```

Expected: PASS.

**Step 4: Commit**

```bash
git add frontend/lib/api.ts frontend/.env.example
git commit -m "chore: make frontend api base url configurable via env"
```

---

### Task 7: Add consolidated verification target/documentation

**Files:**
- Modify: `backend/README.md`
- Modify: project root `README.md` (quick-start snippet)

**Step 1: Write failing doc check**

Run:

```bash
python -c "from pathlib import Path; t=Path('backend/README.md').read_text(encoding='utf-8'); assert 'Cosmos local validation suite' in t"
```

Expected: FAIL.

**Step 2: Add verification section**

Document command bundle:

```bash
python scripts/cosmos/bootstrap.py
python scripts/cosmos/smoke_api.py --cleanup
pytest backend/tests/integration/test_collections_documents_cosmos.py -v
pytest backend/tests/integration/test_indexing_cosmos_profile.py -v
pytest backend/tests/integration/test_query_cosmos_storage_abstraction.py -v
pytest backend/tests/integration/test_cosmos_persistence_restart.py -v
pytest backend/tests/integration/test_cosmos_failure_modes.py -v
```

Add troubleshooting bullets:
- cert/connectivity errors,
- emulator warmup delay,
- reset command.

**Step 3: Re-run doc check**

Run:

```bash
python -c "from pathlib import Path; t=Path('backend/README.md').read_text(encoding='utf-8'); assert 'Cosmos local validation suite' in t"
```

Expected: PASS.

**Step 4: Commit**

```bash
git add backend/README.md README.md
git commit -m "docs: add cosmos local validation suite and troubleshooting guide"
```

---

### Task 8: Final phase verification run

**Files:**
- Test only (unless fixes discovered)

**Step 1: Execute full phase checks**

Run:

```bash
python scripts/cosmos/bootstrap.py
python scripts/cosmos/smoke_api.py --cleanup
pytest backend/tests/unit/test_config.py -v
pytest backend/tests/unit/test_helpers_config_overrides.py -v
pytest backend/tests/unit/test_query_service_storage_mode.py -v
pytest backend/tests/integration/test_collections_documents_cosmos.py -v
pytest backend/tests/integration/test_indexing_cosmos_profile.py -v
pytest backend/tests/integration/test_query_cosmos_storage_abstraction.py -v
pytest backend/tests/integration/test_cosmos_persistence_restart.py -v
pytest backend/tests/integration/test_cosmos_failure_modes.py -v
```

Expected: all PASS.

**Step 2: If any fail, fix minimally and rerun only affected tests**

**Step 3: Commit final hardening adjustments (if changed)**

```bash
git add <changed-files>
git commit -m "chore: finalize cosmos emulator dx and verification hardening"
```

---

## Phase 5 Completion Checklist

- [ ] Bootstrap, smoke, and reset scripts exist and are documented.
- [ ] End-to-end smoke flow is automatable.
- [ ] Persistence across restart is verified.
- [ ] Negative scenarios produce explicit, actionable failures.
- [ ] Frontend API base URL is env-configurable for compose.
- [ ] Docs provide clear local run/validate/troubleshoot/reset workflows.

---

## Notes for Executor

- Keep scripts readable and deterministic; avoid hidden side effects.
- Use explicit timeouts/retries in smoke script (no unbounded waits).
- Do not add cloud deployment or CI pipeline changes in this phase.
