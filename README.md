# Legal RAG — GDPR Research Assistant

An AI-powered legal research tool that answers natural-language queries about GDPR and EDPB guidelines by retrieving exact source passages and generating structured legal briefs with verbatim citations.

---

## Architecture

```
User query (plain language)
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│                    QUERY PIPELINE                        │
│                                                          │
│  Query → Term Expansion                                  │
│     │                                                    │
│     ├──► BM25 retrieval (top-50) ──┐                    │
│     └──► Dense retrieval (top-50) ─┴─► RRF Fusion       │
│                                            │             │
│                                    Cross-encoder rerank  │
│                                            │             │
│                                      Top-8 chunks        │
│                                            │             │
│                                    LLM (Mistral / Claude)│
│                                            │             │
│                                    Structured Brief      │
└─────────────────────────────────────────────────────────┘
        │
        ▼
   Markdown brief with verbatim citations + source table
```

### Components

| Component | Technology | Purpose |
|---|---|---|
| PDF Parser | `pdfplumber` + `ftfy` | Extract and clean text from EUR-Lex and EDPB PDFs |
| Chunker | Custom Python | Structure-aware splitting at article/section boundaries |
| Embeddings | `BAAI/bge-large-en-v1.5` | Dense semantic vectors for retrieval |
| BM25 | `rank-bm25` | Keyword retrieval for exact legal terminology |
| Vector Store | `ChromaDB` | Persistent storage for ~4,600 embedded chunks |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Re-scores top candidates for precision |
| LLM (local) | Mistral 7B via Ollama | Brief generation (development) |
| LLM (cloud) | Claude Haiku via Anthropic API | Brief generation (production) |
| API | FastAPI | REST endpoint + frontend serving |
| Frontend | Vanilla HTML/CSS/JS | Query UI with Markdown rendering |

### Cloud Infrastructure (AWS)

```
Internet → ALB (port 80/443)
               │
           ECS Fargate
           (backend container)
               │
           EFS volume
           (ChromaDB index + PDFs)

ECR → stores Docker images
Secrets Manager → ANTHROPIC_API_KEY
CloudWatch → container logs
```

---

## Quick Start (Local)

### Prerequisites
- Python 3.12+
- [Ollama](https://ollama.com) with `mistral` model pulled

```bash
# 1. Clone and install
git clone https://github.com/octapricot/legal_rag
cd legal_rag
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env — set LLM_BACKEND=ollama

# 3. Add PDFs
# Place GDPR PDF in data/raw/gdpr/
# Place EDPB guideline PDFs in data/raw/edpb/
# File paths must match data/raw/manifest.json

# 4. Build the index (one-time, ~20 min)
python3 main.py ingest --reset

# 5. Query via CLI
python3 main.py query "What are the conditions for valid consent under GDPR?"

# 6. Start the web UI
uvicorn src.api:app --host 0.0.0.0 --port 8000
# Open http://localhost:8000
```

---

## Docker (Local)

```bash
# Development (backend + Ollama)
docker-compose up --build

# Production mode (uses Claude API instead of Ollama)
ANTHROPIC_API_KEY=sk-ant-... docker-compose -f docker-compose.prod.yml up --build
```

---

## Cloud Deployment (AWS)

### 1. Provision infrastructure with Terraform

```bash
cd infra
terraform init
terraform plan -var="aws_region=eu-west-1"
terraform apply
```

Terraform creates: VPC, ECS Fargate cluster, ECR repository, EFS volume, ALB, CloudWatch log group, IAM roles, Secrets Manager entry.

### 2. Store the API key

```bash
aws secretsmanager put-secret-value \
  --secret-id legal-rag/anthropic-api-key \
  --secret-string '{"ANTHROPIC_API_KEY":"sk-ant-..."}'
```

### 3. Push image and deploy

The GitHub Actions pipeline (`.github/workflows/deploy.yml`) handles this automatically on every push to `main`. To trigger manually:

```bash
# Build and push to ECR
docker build -t <ECR_URL>:latest .
docker push <ECR_URL>:latest

# Force ECS redeployment
aws ecs update-service --cluster legal-rag --service legal-rag --force-new-deployment
```

### Required GitHub Secrets

| Secret | Description |
|---|---|
| `AWS_ACCESS_KEY_ID` | IAM user key |
| `AWS_SECRET_ACCESS_KEY` | IAM user secret |
| `AWS_REGION` | e.g. `eu-west-1` |
| `ECR_REPOSITORY` | Full ECR image URL |
| `ECS_CLUSTER` | ECS cluster name |
| `ECS_SERVICE` | ECS service name |

---

## API Reference

### `POST /query`

Generate a legal brief.

```json
{
  "query": "When is a DPIA required?",
  "top_k": 8,
  "doc_type_filter": null
}
```

Response: `{ "query", "brief_markdown", "sources_used", "warnings" }`

### `GET /health`

Returns index status: `{ "status": "ok", "chunks_indexed": 4633 }`

---

## Knowledge Base

- **GDPR** — Regulation (EU) 2016/679 (173 recitals + 99 articles → 663 chunks)
- **EDPB Guidelines** — 43 guideline documents → ~3,970 chunks
- **Total indexed:** 4,633 chunks

See [`data/raw/manifest.json`](data/raw/manifest.json) for the full document list.

---

## Environment Variables

See [`.env.example`](.env.example) for all variables with descriptions.

---

## Example Briefs

- [`examples/consent.md`](examples/consent.md) — Requirements for valid consent
- [`examples/breach_notification.md`](examples/breach_notification.md) — Data breach obligations
- [`examples/dpia.md`](examples/dpia.md) — When a DPIA is required
