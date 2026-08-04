# Reliable single-machine deployment

This profile makes Docker/systemd—not a browser process—the owner of KIRAG's
persistent services. It is intended for one Linux host with one NVIDIA GPU.
It improves restart recovery and repeatability, but it is not high availability:
a host, GPU, or disk failure still interrupts service.

## 1. Prepare a dedicated configuration

Install the locked Python environment and Node dependencies, then copy
`.env.example` to a host-only environment file such as `/etc/kirag/kirag.env`.
Set its owner to the service account and mode to `0600`.

Required production values include distinct, randomly generated API/admin keys,
strong PostgreSQL and MinIO credentials, an immutable vLLM image digest, an
immutable 40-character Hugging Face commits for both models, and absolute `KIRAG_HF_HOME`
and `KIRAG_LOG_DIR` paths. Keep `KIRAG_ENABLE_REMOTE_LIFECYCLE=false`; operators
should use systemd for lifecycle changes.

To expose the guarded KIRAG shutdown button in the UI, set
`KIRAG_ENABLE_APP_SHUTDOWN=true`. The endpoint requires the dedicated admin
credential and an exact typed confirmation. The installer below adds a narrow
root-owned systemd path trigger that stops KIRAG services and containers while
leaving the host powered on; the API itself receives no sudo privileges. The
application units are disabled at installation and by the shutdown action, so
they stay off across host restarts. Opening the desktop launcher explicitly
starts the complete dependency chain again.

## 2. Stage and verify the model while online

Runtime vLLM is deliberately offline. Downloading is a separate deployment
operation so a Hugging Face outage or expired signed URL cannot break a restart:

```bash
HF_TOKEN='<scoped-download-token>' .venv/bin/python scripts/prepare-production-model.py \
  "$OCR_MODEL" --revision "$OCR_MODEL_COMMIT_SHA" \
  --cache-dir /absolute/path/to/workspace/huggingface

HF_TOKEN='<scoped-download-token>' .venv/bin/python scripts/prepare-production-model.py \
  "$ANALYSIS_MODEL" --revision "$ANALYSIS_MODEL_COMMIT_SHA" \
  --cache-dir /absolute/path/to/workspace/huggingface

.venv/bin/python scripts/prepare-production-model.py \
  "$OCR_MODEL" --revision "$OCR_MODEL_COMMIT_SHA" \
  --cache-dir /absolute/path/to/workspace/huggingface --offline-check

.venv/bin/python scripts/prepare-production-model.py \
  "$ANALYSIS_MODEL" --revision "$ANALYSIS_MODEL_COMMIT_SHA" \
  --cache-dir /absolute/path/to/workspace/huggingface --offline-check
```

The command rejects moving branch/tag names, incomplete downloads, snapshots
without `config.json`, and snapshots without a recognized weight file.

## 3. Build the frontend artifact

The production service never runs `next dev`:

```bash
cd frontend
npm ci
npm run lint
npm run typecheck
npm test -- --runInBand
npm run build
cd ..
```

KIRAG pins the stable Next.js release in both `package.json` and the lockfile.

## 4. Validate and install supervision

```bash
.venv/bin/python scripts/production-preflight.py \
  --root "$PWD" --env-file /etc/kirag/kirag.env

sudo scripts/install-systemd-services.sh kirag "$PWD" /etc/kirag/kirag.env
sudo systemctl start kirag-frontend
```

The resulting dependency chain is:

```text
kirag-infrastructure -> kirag-api -> kirag-frontend
```

Infrastructure uses `docker-compose.rag.yml` plus
`docker-compose.production.yml`. Compose waits for PostgreSQL, Redis, MinIO,
Qdrant and one exclusive `kirag_vllm` inference slot. OCR publishes port 8000;
analysis publishes port 8002; only the active role's port is open. Guarded
switching drains active work and verifies the target model before activation.
Systemd then initializes schemas/buckets/vector
collections idempotently. API and frontend failures restart with a bounded
systemd start limit. An explicit service stop does not become a restart loop.

## 5. Operate and verify

```bash
systemctl status kirag-infrastructure kirag-api kirag-frontend
curl --fail http://127.0.0.1:8001/livez
curl --fail http://127.0.0.1:8001/readyz
curl --fail http://127.0.0.1:8001/inference/ready
journalctl -u kirag-api -u kirag-frontend --since today
docker compose -f docker-compose.rag.yml -f docker-compose.production.yml ps
```

`/livez` checks the API process. `/readyz` returns HTTP 503 until every backing
service and vLLM is usable. Application logs rotate as JSON under
`KIRAG_LOG_DIR`; Docker JSON logs are capped at five 20 MiB files per container.

Use `systemctl restart kirag-api` or `kirag-frontend` for application updates.
Those operations do not stop vLLM or databases. To deliberately recycle the
full stack, use `systemctl restart kirag-infrastructure`, then restart the API
and frontend. Grace periods allow active work to terminate before forced kill.

## Updates and rollback

Build and test a release before changing the active checkout. Record the source
commit, image digest, model commit, Python lockfiles, npm lockfile, and a backup
identifier together. After deployment, require `/readyz` and a representative
OCR/RAG smoke test to pass. If they fail, restore the prior source revision,
image/model revisions and frontend build, run the preflight again, then restart
the dependency chain. Do not change model artifacts during runtime startup.

Persistent data remains under `workspace/`. Back up PostgreSQL, MinIO objects,
Qdrant storage/snapshots, `settings.json`, the environment file, and audit logs
before upgrades. A backup is not accepted until a periodic restore rehearsal on
a disposable host has successfully passed schema initialization, `/readyz`, and
a representative retrieval query.
