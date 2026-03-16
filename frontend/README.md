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

The frontend calls:
- `${NEXT_PUBLIC_API_BASE_URL}/api/*` for backend APIs
- `${NEXT_PUBLIC_API_BASE_URL}/.auth/me` for Easy Auth session inspection
- `${NEXT_PUBLIC_API_BASE_URL}/.auth/login/aad` for Microsoft Entra sign-in
- `${NEXT_PUBLIC_API_BASE_URL}/.auth/login/google` for Google sign-in
- `${NEXT_PUBLIC_API_BASE_URL}/.auth/logout` for sign-out

In the Cloudflare deployment model:

- frontend UI is served from `https://app.<domain>`
- backend auth and API are served from `https://api.<domain>`
- login/logout URLs should include redirect parameters so users return to `app.<domain>` after auth actions

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
