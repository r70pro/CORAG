# KIRAG frontend

The frontend uses a same-origin Backend-for-Frontend route at `/api/*`. Browser code never receives the FastAPI origin or an API credential. The Next.js server streams requests to FastAPI, injects `KIRAG_API_KEY` as `X-API-Key`, optionally injects `KIRAG_ADMIN_API_KEY` for administrative controls, and streams the response back unchanged.

Do not create `NEXT_PUBLIC_API_URL`, `NEXT_PUBLIC_KIRAG_API_KEY`, or any similar public variable. Next.js embeds `NEXT_PUBLIC_*` values in browser JavaScript. Do not put either key in `localStorage`, `sessionStorage`, IndexedDB, cookies, or application state.

## Configuration

These variables belong to the Next.js server process:

| Variable | Required | Purpose |
|---|---:|---|
| `KIRAG_API_URL` | No | Server-to-server FastAPI base URL; defaults to `http://127.0.0.1:8001` |
| `KIRAG_API_KEY` | Yes | General API credential injected by the Next.js proxy |
| `KIRAG_ADMIN_API_KEY` | For the complete UI | Credential for cleanup, deletion, indexing, settings, Docker, and other administrative actions |

The same API credentials must be configured on the FastAPI process. Keep real values in process environment variables or an untracked `frontend/.env.local`; all `.env.*` files are ignored except examples.

## Profile 1: local-only workstation

Keep both services on loopback and enable API authentication. Generate two independent random values, then make them available to both processes:

```bash
export KIRAG_API_KEY="replace-with-a-long-random-api-key"
export KIRAG_ADMIN_API_KEY="replace-with-a-different-long-random-admin-key"
uvicorn api.main:app --host 127.0.0.1 --port 8001
```

In a second terminal:

```bash
cd frontend
export KIRAG_API_URL="http://127.0.0.1:8001"
export KIRAG_API_KEY="replace-with-the-same-api-key"
export KIRAG_ADMIN_API_KEY="replace-with-the-same-admin-key"
npm ci
npm run dev -- --hostname 127.0.0.1
```

Open `http://127.0.0.1:3000`. Browser requests go only to `http://127.0.0.1:3000/api/*`; FastAPI remains inaccessible to other hosts.

For repeatable local startup, copy `.env.example` to `.env.local` inside this directory and set permissions so only the service account can read it. The backend still needs the same values in its own process environment.

## Profile 2: authenticated remote reverse proxy

Run FastAPI and Next.js on loopback or a private application network. Put a TLS reverse proxy with user authentication in front of the entire Next.js application, including `/api/*`. Authenticating only page routes is unsafe because the Next.js API proxy holds server credentials.

Example Next.js service environment:

```text
KIRAG_API_URL=http://127.0.0.1:8001
KIRAG_API_KEY=<shared-api-secret>
KIRAG_ADMIN_API_KEY=<shared-admin-secret>
```

Build once, then start the runtime with those server-side values:

```bash
cd frontend
npm ci
npm run build
npm run start -- --hostname 127.0.0.1 --port 3000
```

An nginx profile that preserves uploads, byte-range PDF viewing, and SSE streaming:

```nginx
server {
    listen 443 ssl http2;
    server_name kirag.example.com;

    ssl_certificate     /etc/letsencrypt/live/kirag.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/kirag.example.com/privkey.pem;

    auth_basic "KIRAG";
    auth_basic_user_file /etc/nginx/kirag.htpasswd;

    client_max_body_size 500m;

    location / {
        proxy_pass http://127.0.0.1:3000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-Host $host;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;

        proxy_request_buffering off;
        proxy_buffering off;
        proxy_read_timeout 2h;
        proxy_send_timeout 2h;
    }
}
```

Use OAuth/OIDC/SSO instead of basic authentication where available. The authentication policy must cover every path. Do not publish port `8001`, do not route public traffic directly to FastAPI, and do not log request headers containing API credentials.

If FastAPI is on another host, use a private authenticated network or HTTPS for `KIRAG_API_URL`. Ensure intermediate load balancers also disable response buffering and allow request/response durations long enough for ingestion.

## Verification

With both services running and API authentication enabled:

```bash
# Same-origin proxy succeeds.
curl -u '<user>:<password>' https://kirag.example.com/api/health

# Direct API access is unavailable externally. From the API host it rejects no-key requests.
curl -i http://127.0.0.1:8001/api/health
```

The first request should return `200`; the second should return `401`. In browser developer tools, uploads, JSON requests, downloads, PDF range requests, and event streams should all target the frontend origin under `/api/*`, with no `X-API-Key` or backend origin visible in browser request headers.

## Development commands

```bash
npm run typecheck
npm run lint
npm test
npm run build
```
