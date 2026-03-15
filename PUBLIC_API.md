# Public API

This project now exposes a public API endpoint for search so you can use app functionality without the web UI.

## Base URL

Use your deployment URL, for example:

- `http://localhost:25565`

## Authentication

If `PUBLIC_API_KEY` is configured in `.env`, you must pass it in one of these headers:

- `X-API-Key: <your-key>`
- `Authorization: Bearer <your-key>`

If `PUBLIC_API_KEY` is not set, the endpoint is open (development mode).

## CORS

CORS headers are applied on all `/api/public/*` endpoints.

- Configure allowed origins with `PUBLIC_API_ALLOWED_ORIGINS` (comma-separated list)
- Default is `*`

Example:

- `PUBLIC_API_ALLOWED_ORIGINS=https://example.com,https://app.example.com`

## Endpoints

### `GET /api/public/health`

Health check endpoint.

Response:

```json
{
  "status": "ok",
  "service": "veridion-public-api",
  "api_key_required": true
}
```

### `POST /api/public/search`

Runs the same search pipeline as the UI.

**Synchronous mode** (default) — waits for results and returns them directly:

Request body:

```json
{
  "prompt": "German packaging suppliers with over 200 employees",
  "top_k": 20
}
```

**Asynchronous mode** — pass a `callback_url` to return immediately and receive results via webhook:

```json
{
  "prompt": "German packaging suppliers with over 200 employees",
  "top_k": 20,
  "callback_url": "https://yourapp.com/webhook/results"
}
```

Returns `202 Accepted`:

```json
{
  "job_id": "a1b2c3d4...",
  "status": "pending",
  "poll_url": "/api/public/jobs/a1b2c3d4..."
}
```

When the pipeline completes, a `POST` is sent to `callback_url` with the full result payload plus a `job_id` field and an `X-Job-Id` header. On failure the payload contains `"status": "failed"` and an `"error"` field.

Notes:

- `prompt` is required
- `top_k` is optional (defaults to `20`, clamped to `1..100`)
- `callback_url` must be an `http://` or `https://` URL; webhook delivery is best-effort (one attempt, 10 s timeout)
- Job results are retained for 1 hour

### `GET /api/public/jobs/<job_id>`

Poll the status of an async search job.

Response while running:

```json
{
  "job_id": "a1b2c3d4...",
  "status": "pending",
  "prompt": "...",
  "created_at": 1234567890.0,
  "completed_at": null,
  "callback_delivered": false
}
```

Response when complete — same shape as the synchronous search response, plus the job metadata fields above (`status`, `created_at`, etc.).

Possible `status` values: `pending`, `running`, `completed`, `failed`.

Response shape:

```json
{
  "prompt": "...",
  "detected_language": "en",
  "total": 10,
  "results": [
    {
      "rank": 1,
      "name": "Example Company",
      "company": {}
    }
  ],
  "pipeline": "stage1_stage2_stage3",
  "prefilter_applied": false,
  "prefilter_filters": {},
  "prefilter_candidate_count": 123
}
```

## cURL examples

Health:

```bash
curl -s http://localhost:25565/api/public/health
```

Search (with API key):

```bash
curl -s -X POST http://localhost:25565/api/public/search \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_PUBLIC_API_KEY" \
  -d '{"prompt":"German companies","top_k":20}'
```

Search (Bearer auth):

```bash
curl -s -X POST http://localhost:25565/api/public/search \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_PUBLIC_API_KEY" \
  -d '{"prompt":"Automotive electronics manufacturers in Romania"}'
```

Async search with webhook:

```bash
curl -s -X POST http://localhost:25565/api/public/search \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_PUBLIC_API_KEY" \
  -d '{"prompt":"German companies","top_k":20,"callback_url":"https://yourapp.com/webhook"}'
```

Poll job status:

```bash
curl -s http://localhost:25565/api/public/jobs/a1b2c3d4...
```

## Environment variables

Add these to `.env` as needed:

- `PUBLIC_API_KEY=change-me` (recommended for production)
- `PUBLIC_API_ALLOWED_ORIGINS=https://yourdomain.com`

Existing required variables for the search pipeline still apply:

- `FEATHERLESS_API_KEY`
- `GOOGLE_TRANSLATE_API_KEY` (optional but recommended)
