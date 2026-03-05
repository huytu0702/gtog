# GToG Frontend (Next.js)

## Local development

```bash
cd frontend
npm install
npm run dev
```

Default app URL: `http://localhost:3000`

## Required environment variable

`NEXT_PUBLIC_API_BASE_URL` must point to the public backend origin (or Front Door domain in cloud):

```bash
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

The frontend calls:
- `${NEXT_PUBLIC_API_BASE_URL}/api/*` for backend APIs
- `${NEXT_PUBLIC_API_BASE_URL}/.auth/me` for Easy Auth token retrieval
- `${NEXT_PUBLIC_API_BASE_URL}/.auth/login/aad` and `/.auth/logout` for auth UI actions

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
