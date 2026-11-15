# Phase 4 — Production RAG (Weeks 13–16)

## Goal
Deploy, monitor, and scale RAG systems. Production-grade API with caching, observability, cost optimization, and CI/CD.

---

## Week 13 — API Design + Caching

### Day 1: FastAPI Endpoints
`/ingest` (upload documents), `/query` (ask questions), `/feedback` (thumbs up/down), `/health`

```python
# production_rag/api.py
from fastapi import FastAPI, UploadFile, HTTPException
from pydantic import BaseModel

app = FastAPI()

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5

class QueryResponse(BaseModel):
    answer: str
    sources: list
    latency_ms: float

@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    import time
    start = time.time()
    result = rag_system.run(req.question)
    latency = (time.time() - start) * 1000
    return QueryResponse(answer=result.answer, sources=result.sources, latency_ms=latency)

@app.post("/ingest")
async def ingest(file: UploadFile):
    # Save file, load, split, embed, add to vector store
    return {"status": "ingested", "chunks": 0}
```

### Day 2: Rate Limiting + Auth
API keys, rate limiting per user, request validation.

### Day 3: Semantic Caching
Cache similar queries to avoid redundant LLM calls. Use embedding similarity for cache lookup.

### Day 4: Exact Match + Cache Invalidation
Exact match cache, TTL, invalidation when documents change.

**Deliverable:** `production_rag/api.py` with caching layer

---

## Week 14 — Observability + Cost

### Day 1: Tracing
LangSmith, Arize Phoenix, or Weights & Biases. Trace every retrieval and generation step.

### Day 2: Latency Tracking
p50, p95, p99 latency per endpoint. Identify bottlenecks (embedding vs retrieval vs LLM).

### Day 3: Cost Tracking
Cost per query, cost per token, model routing (cheap model for simple queries).

### Day 4: Dashboard
Visualize traces, latency, cost, error rates.

**Deliverable:** `production_rag/observability.py` + dashboard screenshots

---

## Week 15 — Scaling + Security

### Day 1: Async Ingestion
Background tasks for document ingestion, queue systems (Celery, RQ, or simple asyncio).

### Day 2: Vector DB Scaling
Sharding, batch embedding, index optimization.

### Day 3: PII Redaction
Detect and mask PII (emails, phones, SSNs) before sending to LLM.

### Day 4: Prompt Injection Guardrails
Detect and block injection attempts in user queries.

**Deliverable:** `production_rag/security.py` with PII masking + guardrails

---

## Week 16 — CI/CD + Final Gate

### Day 1: GitHub Actions
Test retrieval quality on every push, deploy artifacts.

### Day 2: A/B Testing Framework
Compare two RAG configurations on live traffic.

### Day 3: Human-in-the-Loop
Feedback collection, review queue, continuous improvement.

### Day 4: Production Deploy + Report
Deploy to cloud, final cost/latency report, runbook.

**Deliverable:** Deployed API URL + `EVAL_REPORTS/production_rag_final.json`

---

## Phase 4 Hard Gate
- [ ] Deployed FastAPI API with `/query` and `/ingest`
- [ ] Semantic caching reducing LLM calls 30%+
- [ ] Observability dashboard with traces
- [ ] Cost per query logged and optimized
- [ ] PII redaction + prompt injection guardrails
- [ ] CI/CD pipeline running on every push
- [ ] p95 latency < 2 seconds
