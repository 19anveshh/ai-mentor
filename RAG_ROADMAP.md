# RAG Learning Path: Beginner to Advanced

Structured roadmap for mastering Retrieval-Augmented Generation (RAG) from fundamentals to production systems.

---

## Phase 0 — Prerequisites (Weeks 1–2)

**Goal:** Solid foundation before touching RAG.

| Topic | What to learn | Deliverable |
|-------|--------------|-------------|
| Python fluency | Type hints, OOP, async/await, decorators, context managers, virtual envs | 3 Python scripts using classes + async |
| ML fundamentals | Embeddings, cosine similarity, nearest neighbors, train/test split, metrics | Notebook comparing embedding models |
| LLMs basics | Tokenization, context window, temperature, top-p, system prompts, few-shot | Prompt engineering cheat sheet |
| Vector databases | What is a vector index, approximate nearest neighbor (ANN), metadata filtering | Chroma/FAISS quickstart with 100 docs |
| LangChain / LlamaIndex | Document loaders, text splitters, basic chains | Simple "chat with a PDF" script |

**Exit criteria:** You can load a PDF, split it, embed it, store in a vector DB, and retrieve relevant chunks with an LLM answer.

---

## Phase 1 — Beginner RAG: Naive RAG (Weeks 3–4)

**Goal:** Build a working naive RAG pipeline and understand its failure modes.

| Topic | What to learn | Deliverable |
|-------|--------------|-------------|
| RAG architecture | Index → Retrieve → Generate; the RAG triad | Diagram + explanation |
| Document loading | PDF, HTML, Markdown, text, Docx loaders; metadata extraction | Loader for 3+ file types |
| Text splitting | Character vs recursive vs token-based splitting; chunk size/overlap tradeoffs | Experiment comparing 3 splitting strategies |
| Embedding models | sentence-transformers, OpenAI embeddings, BGE, E5; dimensionality, cost | Embedding comparison table |
| Vector stores | Chroma, FAISS, Pinecone, Qdrant, Weaviate; add/query/delete | Vector DB with 1k docs |
| Basic retrieval | Similarity search, max marginal relevance (MMR), top-k tuning | Retrieval quality test set |
| Generation | Prompt templates for RAG, citation formatting, handling "no answer" | QA bot with source citations |
| RAGAS evaluation | Faithfulness, answer relevancy, context precision, context recall | `EVAL_REPORTS/rag_baseline.json` |

**Phase 1 project:** "Chat with your documents" — upload PDFs, ask questions, get cited answers.

**Hard gate:** Working naive RAG + eval report with RAGAS metrics.

---

## Phase 2 — Intermediate RAG: Advanced Retrieval (Weeks 5–8)

**Goal:** Fix naive RAG's retrieval problems (missing context, wrong chunks, irrelevant results).

| Topic | What to learn | Deliverable |
|-------|--------------|-------------|
| Query transformation | Query expansion, multi-query, hypothetical document embeddings (HyDE), query decomposition | Multi-query retriever |
| Re-ranking | Cross-encoders (bge-reranker, Cohere rerank), sentence-transformers cross-encoder | Re-ranking module cutting top-20 → top-5 |
| Hybrid search | BM25 + vector search, reciprocal rank fusion (RRF), weighted scoring | Hybrid retriever (BM25 + dense) |
| Advanced splitting | Semantic chunking, document-aware splitting, parent-child retrieval | Parent-child document retriever |
| Metadata filtering | Pre-filtering, post-filtering, hybrid filtering strategies | Filtered retrieval by date/category |
| Context compression | LongLLMLingua, contextual compression, lost-in-the-middle awareness | Compressed context pipeline |
| Evaluation framework | Custom eval dataset (50+ QA pairs), retrieval metrics (MRR, NDCG, hit@k) | `EVAL_REPORTS/intermediate_rag.json` |

**Phase 2 project:** Domain-specific RAG (e.g., legal contracts, medical docs, codebase) with hybrid search + re-ranking.

**Hard gate:** Retrieval hit@k > 0.8 on custom eval set + re-ranking + hybrid search.

---

## Phase 3 — Advanced RAG: Modular & Agentic (Weeks 9–12)

**Goal:** RAG that reasons, routes, and self-corrects.

| Topic | What to learn | Deliverable |
|-------|--------------|-------------|
| Modular RAG | Retrieval → rerank → compress → generate as swappable modules | Pluggable RAG framework |
| Agentic RAG | LangGraph / LlamaIndex agents; retrieve → critique → rewrite → re-retrieve | Agent that decides when to retrieve |
| Router patterns | Route to vector DB vs keyword search vs LLM-only vs SQL database | Intelligent query router |
| Multi-hop reasoning | Decompose complex questions, retrieve for each sub-question, synthesize | Multi-hop QA system |
| Self-RAG / CRITIC | Self-reflection, adaptive retrieval, iterative refinement | Self-correcting RAG loop |
| Multi-modal RAG | Images + text, document layout parsing (Unstructured, Marker), table extraction | RAG over PDFs with tables + images |
| Graph RAG | Knowledge graphs, entity extraction, graph traversal + vector search | GraphRAG over document corpus |
| Long-context RAG | Large context windows, needle-in-haystack testing, lost-in-the-middle | Context window stress test |

**Phase 3 project:** Agentic research assistant — browses multiple sources, critiques its own answers, retrieves more if needed.

**Hard gate:** Agentic RAG with self-correction loop + multi-modal support + graph RAG option.

---

## Phase 4 — Production RAG (Weeks 13–16)

**Goal:** Deploy, monitor, and scale RAG systems.

| Topic | What to learn | Deliverable |
|-------|--------------|-------------|
| API design | FastAPI endpoints for ingest, query, feedback; rate limiting, auth | Production RAG API |
| Caching | Semantic caching, exact match caching, cache invalidation | Cache layer reducing LLM calls 30%+ |
| Observability | LangSmith / Arize Phoenix / Weights & Biases tracing; latency, cost, token usage | Dashboard with traces |
| Evaluation in production | Online evaluation, A/B testing, human-in-the-loop feedback | Continuous eval pipeline |
| Scaling | Vector DB sharding, batch embedding, async ingestion, queue systems | Ingest pipeline for 10k+ docs |
| Cost optimization | Model routing (cheap model for simple queries), prompt compression, caching | Cost log showing 40% reduction |
| Security | PII redaction, prompt injection guardrails, access control per document | Secure RAG with PII masking |
| CI/CD for RAG | Test retrieval quality on every change, deploy artifacts | GitHub Actions pipeline |

**Phase 4 project:** Production RAG API with auth, caching, observability, and CI/CD.

**Hard gate:** Deployed API with <2s p95 latency, cost/logged, 90%+ retrieval hit rate.

---

## Phase 5 — Expert / Specialized RAG (Weeks 17–20)

**Goal:** Cutting-edge and domain-specific RAG.

| Topic | What to learn | Deliverable |
|-------|--------------|-------------|
| Fine-tuning for RAG | Fine-tune embedding models (triplet loss), fine-tune rerankers, adapter-based LLMs | Domain-specific embedding model |
| RAG with structured data | Text-to-SQL + RAG, graph + vector hybrid, document + table reasoning | Text-to-SQL RAG system |
| Real-time RAG | Streaming ingestion, webhooks, change data capture (CDC), live document updates | Live news/feed RAG |
| Multimodal agentic RAG | Image + text retrieval, video frame extraction, OCR + RAG | Multimodal research assistant |
| RAG evaluation research | LLM-as-judge, pairwise comparison, RAG triad improvements | Custom eval framework published |
| Open-source contribution | Contribute to LangChain, LlamaIndex, RAGAS, or vector DB repos | 2+ merged PRs |

**Phase 5 project:** Domain-expert RAG (e.g., medical, legal, engineering) with fine-tuned embeddings + graph RAG.

**Hard gate:** Fine-tuned embedding model outperforms base by 10%+ on domain eval set.

---

## Evaluation Metrics (Used Throughout)

| Category | Metrics | When to use |
|----------|---------|-------------|
| Retrieval | Hit@k, MRR, NDCG, precision@k, recall@k | Every phase |
| Generation | Faithfulness, answer relevancy, semantic similarity | Every phase |
| End-to-end | RAGAS score, correctness, citation accuracy | Phase 1+ |
| Production | Latency (p50/p95), cost/query, token usage, cache hit rate | Phase 4+ |
| User | Thumbs up/down, task completion, repeat queries | Phase 4+ |

---

## Tools & Stack Recommendations

| Layer | Beginner | Intermediate | Advanced | Production |
|-------|----------|--------------|----------|------------|
| Framework | LangChain | LangChain + LlamaIndex | LangGraph | LangGraph + FastAPI |
| Vector DB | Chroma / FAISS | Qdrant / Weaviate | Pinecone / Milvus | Pinecone / Weaviate (managed) |
| Embeddings | sentence-transformers | BGE / E5 | Fine-tuned BGE | Fine-tuned + cached |
| Re-ranking | None | Cross-encoder | Cohere / bge-reranker | Cohere Rerank + caching |
| Eval | RAGAS | RAGAS + custom dataset | LLM-as-judge | Arize Phoenix + online eval |
| Deployment | Local | Docker | K8s | K8s + autoscaling |

---

## Project Portfolio (Build These)

1. **Beginner:** Chat with PDFs (naive RAG)
2. **Intermediate:** Legal document QA (hybrid search + re-ranking)
3. **Advanced:** Research assistant agent (self-correction + multi-hop)
4. **Production:** Enterprise RAG API (caching, observability, auth)
5. **Expert:** Domain RAG with fine-tuned embeddings + graph RAG

---

## Escalation Rules (from mentorship)

- Stuck 2 weeks → scope change (simplify the project)
- Burnout → mandatory reset (1 week off, then resume)
- Scope creep → backlog freeze (finish current phase before adding features)

---

## Next Immediate Action

Pick your starting point:
- **Week 1–2:** Finish Python/ML prerequisites (you're already doing this in Phase 1 Foundation)
- **Week 3:** Build naive RAG with LangChain + Chroma + PDF loader

Want me to create Week 3 starter code (naive RAG pipeline) and push it to your `ai-mentor` GitHub repo?
