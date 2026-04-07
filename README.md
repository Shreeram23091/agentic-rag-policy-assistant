# ACME RAG Agent

An AI-powered question-answering agent that answers employee and customer queries about internal company documents using **Retrieval-Augmented Generation (RAG)**. Built with FastAPI, OpenAI / Azure OpenAI, and FAISS.

Interactive API Docs: http://localhost:8000/docs
(Available once the Uvicorn server is running locally)
---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Client / API Consumer                        │
│                    POST /ask  { query, session_id }                  │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      FastAPI Application                             │
│                                                                      │
│  ┌──────────────┐    ┌─────────────────┐    ┌────────────────────┐ │
│  │   /ask route │───▶│  Session Memory │    │   /health route    │ │
│  │              │    │  (in-process,   │    │                    │ │
│  │              │    │   TTL-based)    │    │                    │ │
│  └──────┬───────┘    └─────────────────┘    └────────────────────┘ │
│         │                                                            │
│         ▼                                                            │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                         AI Agent                              │   │
│  │                                                               │   │
│  │  1. Build messages (system prompt + history + user query)     │   │
│  │  2. Call LLM with tool definitions                            │   │
│  │  3. If LLM calls `search_documents` tool:                     │   │
│  │       a. Embed query  ──▶  FAISS vector search               │   │
│  │       b. Retrieve top-k document chunks                       │   │
│  │       c. Re-call LLM with retrieved context                   │   │
│  │  4. If LLM answers directly → return immediately              │   │
│  │  5. Return { answer, sources, used_rag }                      │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
                │ embed query / chat completion
                ▼
┌───────────────────────────────┐
│   OpenAI / Azure OpenAI API   │
│  - gpt-4o  (chat)             │
│  - text-embedding-3-small     │
└───────────────────────────────┘

┌───────────────────────────────────────────────────────┐
│              RAG Pipeline (build-time)                 │
│                                                        │
│  docs/*.txt / *.pdf                                    │
│        │                                               │
│        ▼                                               │
│  Document Loader → Chunker (sliding window, overlap)  │
│        │                                               │
│        ▼                                               │
│  Embedding Service (batched, with retry)               │
│        │                                               │
│        ▼                                               │
│  FAISS IndexFlatIP (cosine similarity, L2-normalised) │
│        │                                               │
│        └── persisted to disk → loaded at startup       │
└───────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **API Framework** | [FastAPI](https://fastapi.tiangolo.com/) 0.111 + Uvicorn |
| **LLM** | OpenAI `gpt-4o` / Azure OpenAI |
| **Embeddings** | OpenAI `text-embedding-3-small` / Azure OpenAI |
| **Vector Store** | [FAISS](https://github.com/facebookresearch/faiss) (`faiss-cpu`) |
| **Validation** | Pydantic v2 |
| **Logging** | [structlog](https://www.structlog.org/) |
| **Monitoring** | Azure Monitor / Application Insights (optional) |
| **Containerisation** | Docker + Docker Compose |
| **IaC** | Azure Bicep |
| **CI/CD** | Azure Pipelines |
| **Language** | Python 3.11 |

---

## Project Structure

```
rag-agent/
├── app/
│   ├── main.py                  # FastAPI app factory + lifespan
│   ├── config.py                # Pydantic settings (env vars)
│   ├── llm_client.py            # OpenAI / Azure OpenAI client factory
│   ├── agent/
│   │   └── agent.py             # Core AI agent (tool-calling, RAG orchestration)
│   ├── rag/
│   │   ├── document_loader.py   # Load + chunk documents
│   │   ├── embeddings.py        # Batched embedding generation
│   │   ├── vector_store.py      # FAISS index build/save/load/search
│   │   └── retriever.py         # High-level RAG retriever
│   ├── memory/
│   │   └── session_memory.py    # TTL-based in-memory session store
│   └── api/
│       ├── routes.py            # FastAPI route handlers
│       ├── schemas.py           # Pydantic request/response models
│       └── dependencies.py      # FastAPI dependency injection
├── docs/
│   └── sample_documents/        # 5 sample company documents
│       ├── leave_policy.txt
│       ├── it_security_policy.txt
│       ├── product_faq.txt
│       ├── remote_work_policy.txt
│       └── technical_onboarding.txt
├── scripts/
│   └── build_index.py           # CLI: pre-build FAISS index
├── tests/
│   ├── test_document_loader.py
│   ├── test_session_memory.py
│   └── test_api.py              # Integration tests (mocked LLM)
├── deployment/
│   ├── main.bicep               # Azure IaC (App Service + Azure OpenAI + ACR)
│   ├── azure-pipelines.yml      # CI/CD pipeline
│   └── deploy_azure.sh          # Manual deployment script
├── .env.example                 # Environment variable template
├── Dockerfile                   # Multi-stage Docker build
├── docker-compose.yml           # Local Docker Compose
├── requirements.txt
└── pytest.ini
```

---

## API Reference

### `POST /ask`

Submit a query to the AI agent.

**Request:**
```json
{
  "query": "How many days of annual leave do I get?",
  "session_id": "user-abc-123"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `query` | string | ✅ | The user's question (1–2000 chars) |
| `session_id` | string | ❌ | Session ID for multi-turn memory |

**Response:**
```json
{
  "answer": "According to the Leave Policy, full-time employees are entitled to **18 days** of paid annual leave per calendar year...",
  "sources": ["leave_policy.txt"],
  "session_id": "user-abc-123",
  "used_rag": true
}
```

### `GET /health`

Returns service readiness status.

```json
{
  "status": "ok",
  "rag_ready": true,
  "active_sessions": 3,
  "version": "1.0.0"
}
```

### `DELETE /session/{session_id}`

Clears conversation history for a session. Returns `204 No Content`.

**Interactive docs:** `http://localhost:8000/docs` (Swagger UI)

---

## Setup Instructions

### Prerequisites

- Python 3.11+
- An OpenAI API key **or** an Azure OpenAI resource
- Docker (optional, for containerised run)

---

### Local Setup (without Docker)

**1. Clone and install dependencies:**
```bash
git clone https://github.com/your-org/rag-agent.git
cd rag-agent
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**2. Configure environment:**
```bash
cp .env.example .env
# Edit .env — set OPENAI_API_KEY (or Azure credentials)
```

Minimum required variables for standard OpenAI:
```
OPENAI_API_TYPE=openai
OPENAI_API_KEY=sk-...
```

For Azure OpenAI:
```
OPENAI_API_TYPE=azure
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com/
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4o
AZURE_OPENAI_EMBED_DEPLOYMENT=text-embedding-3-small
```

**3. Build the FAISS index (one-time):**
```bash
python scripts/build_index.py
# Use --force to rebuild an existing index
```

**4. Start the API server:**
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**5. Test it:**
```bash
curl -X POST http://localhost:8000/ask \
     -H "Content-Type: application/json" \
     -d '{"query": "How many days of annual leave do employees get?"}'
```

---

### Local Setup (with Docker)

```bash
# Copy and configure environment
cp .env.example .env
# Edit .env with your API keys

# Build and start (index is built automatically on first run)
docker-compose up --build

# The API is available at http://localhost:8000
```

---

### Running Tests

```bash
# Unit tests only (no API key required)
pytest tests/test_document_loader.py tests/test_session_memory.py -v

# All tests including API integration tests (uses mocks, no API key required)
pytest -v
```

---

### Azure Deployment

#### Option A — Automated script (recommended for first deploy)

```bash
# Set your Azure OpenAI key
export AZURE_OPENAI_API_KEY="your-key-here"

# Log in to Azure
az login

# Run the deployment script (provisions everything + deploys)
chmod +x deployment/deploy_azure.sh
./deployment/deploy_azure.sh
```

The script will:
1. Create a Resource Group
2. Deploy Bicep infrastructure (Azure OpenAI, Container Registry, App Service)
3. Build and push the Docker image to ACR
4. Configure App Service environment variables
5. Perform a health check and print the public URL

#### Option B — Azure DevOps CI/CD

1. Import `deployment/azure-pipelines.yml` into your Azure DevOps project
2. Create service connections for ACR and Azure Subscription
3. Push to `main` — the pipeline tests, builds, and deploys automatically

#### Manual environment variable configuration (Azure Portal)

Navigate to **App Service → Configuration → Application settings** and add:

| Variable | Value |
|---|---|
| `OPENAI_API_TYPE` | `azure` |
| `AZURE_OPENAI_API_KEY` | `<your-key>` |
| `AZURE_OPENAI_ENDPOINT` | `https://<resource>.openai.azure.com/` |
| `AZURE_OPENAI_API_VERSION` | `2024-02-01` |
| `AZURE_OPENAI_CHAT_DEPLOYMENT` | `gpt-4o` |
| `AZURE_OPENAI_EMBED_DEPLOYMENT` | `text-embedding-3-small` |
| `ENVIRONMENT` | `production` |
| `FAISS_INDEX_PATH` | `/home/data/faiss_index` |

> **Security note:** For production, store `AZURE_OPENAI_API_KEY` in **Azure Key Vault** and reference it as a Key Vault secret reference in App Service settings.

---

## Design Decisions

### 1. Tool-calling for RAG routing (vs. keyword matching)
Rather than using keyword heuristics or a separate classifier to decide when to retrieve documents, the LLM itself decides via OpenAI's tool-calling API. The `search_documents` tool description is carefully engineered so the model reliably calls it for internal knowledge questions and answers general questions directly. This avoids unnecessary RAG lookups and keeps latency low for simple queries.

### 2. FAISS over managed vector DBs
FAISS (`faiss-cpu`) was chosen for simplicity, zero infrastructure cost, and zero latency overhead. It runs in-process, persists to disk, and loads in under a second. For a production system with millions of documents or multi-instance deployments, this would be replaced with **Azure AI Search** or **Pinecone**.

### 3. In-process session memory
Session history is stored in a Python dictionary with TTL-based expiration. This is intentionally simple for single-instance deployments. For horizontal scaling, replace with **Azure Cache for Redis**.

### 4. Sliding-window chunker with overlap
Documents are chunked by sentence boundaries with configurable overlap. Overlap ensures that relevant sentences at chunk boundaries aren't missed during retrieval. Chunk size (500 chars) and overlap (50 chars) were tuned for the sample documents — larger technical documents may benefit from larger chunks.

### 5. Dual OpenAI/Azure OpenAI support
A single config flag (`OPENAI_API_TYPE`) switches between standard OpenAI and Azure OpenAI with no code changes. This makes local development easy (standard OpenAI key) while production uses Azure OpenAI for compliance and data residency.

### 6. Multi-stage Docker build
The Dockerfile uses a two-stage build to separate the build environment from the runtime image, reducing the final image size significantly by excluding build tools.

---

## Limitations & Future Improvements

### Current Limitations

| Limitation | Impact |
|---|---|
| In-process FAISS index | Not horizontally scalable; index must be rebuilt per instance |
| In-process session memory | Lost on restart; not shared across multiple app instances |
| No authentication | API is open; suitable only for internal/trusted environments |
| FAISS index is static | Adding new documents requires a full index rebuild |
| Single-tenant | No per-user document access control |

### Future Improvements

1. **Replace FAISS with Azure AI Search** — Native Azure integration, incremental indexing, hybrid search (keyword + semantic), and no need to manage index files.

2. **Redis for session memory** — Enables horizontal scaling and session persistence across restarts. Azure Cache for Redis integrates natively with App Service.

3. **Authentication** — Add Azure AD / OAuth 2.0 authentication via FastAPI middleware or Azure API Management in front of the app.

4. **Streaming responses** — Use `StreamingResponse` with OpenAI's streaming API to reduce perceived latency for long answers.

5. **Incremental document ingestion** — Expose a `POST /documents` endpoint to add new documents and update the index without a full rebuild.

6. **Re-ranking** — Add a cross-encoder re-ranking step after FAISS retrieval to improve answer quality, especially for ambiguous queries.

7. **Evaluation harness** — Build a ground-truth Q&A dataset from the sample documents and track retrieval precision/recall and answer quality metrics automatically in CI.

8. **Multi-turn context in retrieval** — Currently, only the current query is embedded for retrieval. Incorporating condensed conversation history into the retrieval query would improve follow-up question handling.

---

## Sample Queries to Try

```bash
# Leave policy
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "How many days of paternity leave do I get?"}'

# IT security
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the password requirements?"}'

# Product FAQ
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "Does the platform support SSO?"}'

# Multi-turn with session
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the remote work policy?", "session_id": "demo-session"}'

curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What about the home office stipend?", "session_id": "demo-session"}'

# General knowledge (no RAG needed)
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the difference between REST and GraphQL?"}'
```

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes with a descriptive message
4. Ensure tests pass (`pytest -v`)
5. Open a Pull Request

---

## License

MIT License — see `LICENSE` for details.
