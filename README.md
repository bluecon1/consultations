# NESO Consultation Summariser (Local-first, Azure-portable)

Summarises confidential consultation responses from `data/data.csv` with:

- Approach 1: organisation-level hybrid summary (section summaries -> roll-up)
- Approach 2: question-level summary (distribution + mainstream/minority capture)
- Evidence linking (`record_id` + excerpt)
- KPIs (coverage, evidence coverage, compression, uncertainty, latency, cost estimate)
- SQLite caching to avoid repeated LLM calls

## Portability Abstractions

The code is structured so local OpenAI and Azure stack can be swapped without changing summarisation logic:

- `neso_consultations/llm/base.py`: provider interface (`LLMProvider`)
- `neso_consultations/llm/factory.py`: provider selection by config
- `neso_consultations/config.py`: all env-driven runtime settings
- `neso_consultations/service.py`: orchestration layer used by UI/CLI
- `neso_consultations/summarisation/`: approach logic independent of provider/runtime

## Prerequisites

- Python `3.11+` (recommended; Python 3.8 will fail on this codebase)
- pip `23+`

## Setup

1. Enter project folder:

```bash
cd neso-consultations
```

2. Create env and install dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

3. Configure `.env`:

```bash
cp .env.example .env
```

Important: `.env` must be at `neso-consultations/.env` (project root), not under `src/`.

## Provider Configuration

Set `LLM_PROVIDER` in `.env`:

- OpenAI public API
  - `LLM_PROVIDER=openai`
  - `OPENAI_API_KEY=...`
  - optional `OPENAI_BASE_URL=...`

- Azure OpenAI
  - `LLM_PROVIDER=azure`
  - `AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com`
  - `AZURE_OPENAI_DEPLOYMENT=<deployment_name>`
  - `AZURE_OPENAI_API_VERSION=2024-06-01`
  - auth option 1: `AZURE_OPENAI_API_KEY=...`
  - auth option 2: `AZURE_OPENAI_USE_AAD=true` (uses `DefaultAzureCredential`)

## Run

UI:

```bash
python -m main ui --host 127.0.0.1 --port 8501
```

Open `http://127.0.0.1:8501`.

CLI:

```bash
python -m main list-orgs
python -m main summary-org --response-id <RESPONSE_ID>
python -m main list-questions
python -m main summary-question --question-id <QUESTION_ID>
```

## Docker

From `neso-consultations/`:

```bash
docker compose up --build
```

UI is reachable at `http://localhost:8501`.

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
```

## Azure ML Studio Uplift Checklist

1. Copy this folder to your client workspace with same structure (`data/`, `.env`, `main.py`, `neso_consultations/`).
2. Build a Python 3.11 environment in Azure ML (or container image) and install `requirements.txt`.
3. Set runtime environment variables from `.env` in AML job/endpoint settings (do not hardcode secrets).
4. For Azure OpenAI auth:
   - API key mode: set `AZURE_OPENAI_API_KEY`.
   - Managed identity mode: set `AZURE_OPENAI_USE_AAD=true` and grant the identity access to Azure OpenAI.
5. Keep data local to mounted storage and set `DATA_CSV_PATH`/`SECTION_MAPPING_PATH` accordingly.
6. For cached outputs, point `CACHE_PATH` at writable mounted storage.
7. Smoke test in AML terminal/job:
   - `python -m main list-orgs`
   - `python -m main summary-org --response-id <id>`
   - `python -m main summary-question --question-id <qid>`
8. If hosting UI in Azure, containerize via the provided `Dockerfile` and expose port `8501`.

## Troubleshooting

- `ModuleNotFoundError: No module named 'dotenv'`
  - Install dependencies in the active venv: `pip install -r requirements.txt`.
- `pip ... BackendUnavailable`
  - Usually old pip/build tooling. Run:
    - `python -m pip install --upgrade pip setuptools wheel`
    - then reinstall requirements.
- `Unsupported LLM_PROVIDER`
  - Set `LLM_PROVIDER` to `openai` or `azure`.
- Azure AAD auth errors
  - Confirm managed identity permissions and `AZURE_OPENAI_ENDPOINT`/deployment/api-version.
- `OPENAI_API_KEY is not set`
  - Ensure key is in `neso-consultations/.env` and restart process.

## Confidentiality

- Data is read from local files only.
- Application avoids logging raw consultation text.
- UI only displays excerpts needed for evidence traceability.
