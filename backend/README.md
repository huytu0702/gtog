# GraphRAG FastAPI Backend

FastAPI backend for GraphRAG that enables document indexing and querying with multiple search methods.

## Features

- **Collection Management**: Create, list, and delete document collections
- **Document Upload**: Upload documents to collections for indexing
- **Indexing**: Background indexing using GraphRAG's indexing pipeline
- **Search Methods**: Support for 4 search methods:
  - **Global Search**: High-level overview across entire dataset
  - **Local Search**: Entity-specific detailed information
  - **ToG Search**: Think-on-Graph reasoning-based exploration
  - **DRIFT Search**: Multi-hop reasoning and relationships

## Setup

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Install GraphRAG Package

From the project root:

```bash
cd ..
pip install -e .
```

### 3. Configure Environment

Create a `.env` file in the `backend/` directory (copy from `.env.example`):

```bash
cp .env.example .env
```

Edit `.env` and set your API keys:

```env
GRAPHRAG_API_KEY=your_openai_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

### 4. Verify Settings

The backend uses a shared `settings.yaml` file. Make sure `/backend/settings.yaml` exists (copied from root during setup).

## Running the Server

From the `backend/` directory:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Running Modes

The backend supports two storage modes:

#### File Mode (Default)
Uses local file storage for GraphRAG data:

```bash
# Set in .env
STORAGE_MODE=file
GRAPHRAG_SETTINGS_FILE=settings.yaml

# Run
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

#### Cosmos DB Emulator Mode
Uses Azure Cosmos DB Emulator for storage:

```bash
# Copy cosmos env template
cp .env.cosmos-emulator.example .env.cosmos-emulator

# Set in .env.cosmos-emulator
STORAGE_MODE=cosmos
GRAPHRAG_SETTINGS_FILE=settings.cosmos-emulator.yaml
COSMOS_ENDPOINT=https://localhost:8081
COSMOS_KEY=C2y6yDjf5/R+ob0N8A7Cgv30VRDJIWEHLM+ZLw==
COSMOS_DATABASE=gtog
COSMOS_CONTAINER=graphrag

# Run with cosmos env file
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 --env-file .env.cosmos-emulator
```

### Running with Docker Compose

Use the root `docker-compose.dev.yaml` for full-stack development:

```bash
# Full stack (Cosmos emulator + backend + frontend)
docker compose -f docker-compose.dev.yaml up --build

# Cosmos emulator only (for external backend)
docker compose -f docker-compose.dev.yaml up -d cosmos-emulator
```

The API will be available at:
- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

## API Usage

### 1. Create a Collection

```bash
curl -X POST http://localhost:8000/api/collections \
  -H "Content-Type: application/json" \
  -d '{"name": "my_docs", "description": "My document collection"}'
```

### 2. Upload Documents

```bash
curl -X POST http://localhost:8000/api/collections/my_docs/documents \
  -F "file=@sample.txt"
```

### 3. Start Indexing

```bash
curl -X POST http://localhost:8000/api/collections/my_docs/index
```

### 4. Check Indexing Status

```bash
curl http://localhost:8000/api/collections/my_docs/index
```

### 5. Search (after indexing completes)

#### Global Search
```bash
curl -X POST http://localhost:8000/api/collections/my_docs/search/global \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the main topics?"}'
```

#### Local Search
```bash
curl -X POST http://localhost:8000/api/collections/my_docs/search/local \
  -H "Content-Type: application/json" \
  -d '{"query": "Tell me about specific entities", "community_level": 2}'
```

#### ToG Search
```bash
curl -X POST http://localhost:8000/api/collections/my_docs/search/tog \
  -H "Content-Type: application/json" \
  -d '{"query": "How are concepts related?"}'
```

#### DRIFT Search
```bash
curl -X POST http://localhost:8000/api/collections/my_docs/search/drift \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the connections between entities?"}'
```

## API Endpoints

### Collections
- `POST /api/collections` - Create a new collection
- `GET /api/collections` - List all collections
- `GET /api/collections/{id}` - Get collection details
- `DELETE /api/collections/{id}` - Delete a collection

### Documents
- `POST /api/collections/{id}/documents` - Upload a document
- `GET /api/collections/{id}/documents` - List documents
- `DELETE /api/collections/{id}/documents/{name}` - Delete a document

### Indexing
- `POST /api/collections/{id}/index` - Start indexing
- `GET /api/collections/{id}/index` - Get indexing status

### Search
- `POST /api/collections/{id}/search/global` - Global search
- `POST /api/collections/{id}/search/local` - Local search
- `POST /api/collections/{id}/search/tog` - ToG search
- `POST /api/collections/{id}/search/drift` - DRIFT search

## Project Structure

```
backend/
├── app/
│   ├── main.py                 # FastAPI application
│   ├── config.py               # Configuration
│   ├── models/
│   │   ├── schemas.py          # Pydantic models
│   │   └── enums.py            # Enumerations
│   ├── services/
│   │   ├── storage_service.py  # File storage
│   │   ├── indexing_service.py # Indexing operations
│   │   └── query_service.py    # Search operations
│   ├── routers/
│   │   ├── collections.py      # Collection endpoints
│   │   ├── documents.py        # Document endpoints
│   │   ├── indexing.py         # Indexing endpoints
│   │   └── search.py           # Search endpoints
│   └── utils/
│       └── helpers.py          # Utility functions
├── storage/                     # Collections storage
├── settings.yaml                # Shared GraphRAG config
├── requirements.txt
├── .env                         # Environment variables
└── README.md
```

## Development

### Code Style

The project follows standard Python conventions. Key packages:
- **FastAPI**: Web framework
- **Pydantic**: Data validation
- **GraphRAG**: Document indexing and search

### Logging

Logs are written to stdout with the format:
```
%(asctime)s - %(name)s - %(levelname)s - %(message)s
```

## Troubleshooting

### Indexing Issues

If indexing fails:
1. Check the logs for errors
2. Verify API keys in `.env`
3. Ensure documents are uploaded to the collection
4. Check `settings.yaml` configuration

### Search Errors

If search returns errors:
1. Verify the collection has been successfully indexed
2. Check that all required parquet files exist in `storage/collections/{id}/output/`
3. Ensure the query is not empty

### Performance

- Indexing time depends on document size and API rate limits
- For large collections, consider adjusting chunk size in `settings.yaml`
- Search performance varies by method (ToG is slower but more thorough)

## License

MIT License - see the main project LICENSE file.
