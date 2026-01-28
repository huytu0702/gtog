# Running GraphRAG Backend and Frontend

This guide covers how to run the GraphRAG application using:
- **uv** for backend development
- **npm** for frontend development
- **Docker Compose** for production-style deployment

---

## Prerequisites

- **Python 3.11+** (for backend)
- **Node.js 20+** (for frontend)
- **uv** - Python package manager: `pip install uv`
- **Docker & Docker Compose** (for containerized deployment)

---

## Directory Structure

```
gtog/
├── backend/           # FastAPI backend
│   ├── app/          # Application code
│   ├── alembic/      # Database migrations
│   ├── Dockerfile
│   └── docker-compose.yml
└── frontend/         # Next.js frontend
    ├── app/
    └── package.json
```

---

## Quick Start with Docker (Recommended)

The fastest way to run the full stack is using Docker Compose.

### 1. Configure Environment Variables

Create a `.env` file in the project root (or `backend/`):

```bash
# Required for GraphRAG indexing
GRAPHRAG_API_KEY=your-openai-api-key
OPENAI_API_KEY=your-openai-api-key

# PostgreSQL (defaults used by docker-compose)
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_USER=graphrag
POSTGRES_PASSWORD=graphrag
POSTGRES_DB=graphrag

# Redis
REDIS_URL=redis://redis:6379/0
```

### 2. Start All Services

```bash
# From project root
cd backend
docker-compose up -d
```

This starts:
- `postgres` - PostgreSQL with pgvector on port 5432
- `redis` - Redis on port 6379
- `backend` - FastAPI API on port 8000
- `worker` - Celery/RQ worker for indexing jobs

### 3. Initialize Database

```bash
# Run Alembic migrations to create tables
docker-compose exec backend alembic upgrade head
```

### 4. Access Services

- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **PostgreSQL**: localhost:5432
- **Redis**: localhost:6379

### 5. Stop Services

```bash
cd backend
docker-compose down

# To remove volumes (including all data)
docker-compose down -v
```

---

## Development with uv and npm

### Backend (Python/FastAPI)

#### 1. Install Dependencies with uv

```bash
cd backend

# Create and sync virtual environment
uv sync

# Or install from requirements.txt
uv pip install -r requirements.txt
```

#### 2. Configure Environment

Create `.env` in `backend/`:

```bash
# Local development settings
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=graphrag
POSTGRES_PASSWORD=graphrag
POSTGRES_DB=graphrag

REDIS_URL=redis://localhost:6379/0

GRAPHRAG_API_KEY=your-openai-api-key
OPENAI_API_KEY=your-openai-api-key
```

#### 3. Start PostgreSQL and Redis (Local or Docker)

**Option A - Using Docker for dependencies only:**

```bash
# Start just postgres and redis
cd backend
docker-compose up -d postgres redis

# Wait for services to be healthy
docker-compose ps
```

**Option B - Local installations:**

Install PostgreSQL 16 with pgvector, and Redis locally.

#### 4. Run Database Migrations

```bash
cd backend

# Run migrations
uv run alembic upgrade head

# Verify tables created
uv run python -c "from app.db.session import engine; print('DB connected')"
```

#### 5. Start Backend Server

```bash
cd backend

# Development server with auto-reload
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Or use the alias
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Backend will be available at http://localhost:8000

#### 6. Run Background Worker (for indexing)

```bash
# In a separate terminal
cd backend

# Start RQ worker
uv run rq worker --url redis://localhost:6379/0 graphrag-indexing
```

#### 7. Common Backend Commands

```bash
cd backend

# Run tests
uv run pytest

# Format code
ruff format .

# Lint code
ruff check . --fix

# Type checking
pyright
```

---

### Frontend (Next.js)

#### 1. Install Dependencies

```bash
cd frontend

# Install npm packages
npm install
```

#### 2. Configure Environment (if needed)

Create `.env.local` in `frontend/`:

```bash
# Backend API URL
NEXT_PUBLIC_API_URL=http://localhost:8000
```

#### 3. Start Development Server

```bash
cd frontend

# Development server with hot reload
npm run dev
```

Frontend will be available at http://localhost:3000

#### 4. Common Frontend Commands

```bash
cd frontend

# Production build
npm run build

# Run production server
npm run start

# Lint code
npm run lint
```

---

## Database Operations

### Alembic Migrations

```bash
cd backend

# Create a new migration
uv run alembic revision --autogenerate -m "description"

# Apply migrations
uv run alembic upgrade head

# Rollback one migration
uv run alembic downgrade -1

# View migration history
uv run alembic history
```

### Direct Database Access

```bash
# Connect using psql
psql -h localhost -U graphrag -d graphrag

# From within Docker
docker-compose exec postgres psql -U graphrag -d graphrag
```

---

## Architecture Overview

### Services

| Service | Technology | Port | Purpose |
|---------|------------|------|---------|
| Backend API | FastAPI | 8000 | REST API endpoints |
| Worker | RQ/Celery | - | Background indexing jobs |
| PostgreSQL | pgvector/pg16 | 5432 | Graph storage + embeddings |
| Redis | Redis 7 | 6379 | Job queue |

### Data Flow

**Indexing Workflow:**
1. Upload document → stored in `documents` table (bytea + text)
2. Start indexing → creates `index_runs` entry (status=queued)
3. Worker dequeues job → runs GraphRAG pipeline
4. Worker writes graph outputs → entities, relationships, communities, embeddings
5. Worker updates `index_runs` → status=completed/failed

**Query Workflow:**
1. Search request → API reads from latest completed `index_run_id`
2. Data fetched from DB tables → no parquet files
3. Results returned → formatted JSON response

---

## Troubleshooting

### Backend Issues

**Port 8000 already in use:**
```bash
# Windows (PowerShell)
netstat -ano | findstr :8000
# Kill the process using PID
taskkill /PID <PID> /F

# Mac/Linux
lsof -ti:8000 | xargs kill -9
```

**Database connection refused:**
```bash
# Check if postgres is running
docker-compose ps postgres

# Check postgres logs
docker-compose logs postgres

# Restart postgres
docker-compose restart postgres
```

**Migration errors:**
```bash
# Reset to initial state (WARNING: destroys data)
uv run alembic downgrade base
uv run alembic upgrade head
```

### Frontend Issues

**Port 3000 already in use:**
```bash
# Kill the process on port 3000
# Mac/Linux
lsof -ti:3000 | xargs kill -9

# Windows (PowerShell)
netstat -ano | findstr :3000
taskkill /PID <PID> /F
```

**Build errors:**
```bash
# Clear Next.js cache
rm -rf .next

# Reinstall dependencies
rm -rf node_modules package-lock.json
npm install
```

### Docker Issues

**Container won't start:**
```bash
# View logs
docker-compose logs <service-name>

# Rebuild without cache
docker-compose build --no-cache <service-name>

# Remove and recreate containers
docker-compose down
docker-compose up -d
```

**Volume issues:**
```bash
# Remove all volumes (WARNING: destroys data)
docker-compose down -v

# Remove specific volume
docker volume rm <volume-name>
```

---

## Production Considerations

### Security
- Change default passwords in production
- Use environment-specific `.env` files
- Enable HTTPS
- Implement rate limiting

### Scaling
- Scale backend: `docker-compose up -d --scale backend=3`
- Scale workers: `docker-compose up -d --scale worker=3`
- Use external managed PostgreSQL and Redis for production

### Monitoring
- Add health checks to docker-compose
- Configure logging aggregation
- Set up metrics collection (Prometheus/Grafana)

---

## Useful Links

- **Backend API Docs**: http://localhost:8000/docs
- **FastAPI Documentation**: https://fastapi.tiangolo.com/
- **Next.js Documentation**: https://nextjs.org/docs
- **GraphRAG Docs**: See `CLAUDE.md` in project root
