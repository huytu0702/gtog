# GToG Frontend (Next.js)

## Local development

```bash
cd frontend
npm install
npm run dev
```

Default app URL: `http://localhost:3000`

## Required environment variable

`NEXT_PUBLIC_API_BASE_URL` must point to the public backend origin:

```bash
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

The frontend calls `${NEXT_PUBLIC_API_BASE_URL}/api/*` for backend APIs.

In the Cloudflare deployment model:

- frontend UI is served from `https://app.<domain>`
- backend API is served from `https://api.<domain>`

## Health endpoint

Frontend health route:
- `GET /api/health` -> `{ "status": "ok" }`

## Docker

Build args:
- `NEXT_PUBLIC_API_BASE_URL` (build-time)

Example:

```bash
docker build -f frontend/Dockerfile --build-arg NEXT_PUBLIC_API_BASE_URL=http://localhost:8000 frontend
```
