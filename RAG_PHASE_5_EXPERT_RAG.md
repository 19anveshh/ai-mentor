# Phase 5 — Expert / Specialized RAG (Weeks 17–20)

## Goal
Cutting-edge and domain-specific RAG. Fine-tuned embeddings, graph + vector hybrid, real-time updates, and open-source contributions.

---

## Week 17 — Fine-Tuning for RAG

### Day 1: Embedding Fine-Tuning
Fine-tune sentence-transformers on your domain data using triplet loss or contrastive loss.

```python
# expert_rag/fine_tune_embedding.py
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

model = SentenceTransformer("all-MiniLM-L6-v2")
train_examples = [
    InputExample(texts=["query", "positive_passage"], label=1.0),
    InputExample(texts=["query", "negative_passage"], label=0.0),
]
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(model=model)
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1)
```

### Day 2: Reranker Fine-Tuning
Fine-tune cross-encoder rerankers on domain-specific query-document pairs.

### Day 3: Domain Eval Set
Create 100+ domain-specific QA pairs with ground truth.

### Day 4: Compare Base vs Fine-Tuned
Measure improvement in hit@k and RAGAS scores.

**Deliverable:** Fine-tuned embedding model + eval comparison report

---

## Week 18 — Structured Data + Text-to-SQL RAG

### Day 1: Text-to-SQL Basics
Convert natural language to SQL queries, execute against database.

### Day 2: SQL + Vector Hybrid
Retrieve relevant table schemas via vector search, generate SQL, execute, return results.

### Day 3: Document + Table Reasoning
Combine unstructured document RAG with structured database queries.

### Day 4: Multi-Source Synthesis
Answer questions requiring both document text and database data.

**Deliverable:** `expert_rag/text_to_sql_rag.py`

---

## Week 19 — Real-Time RAG

### Day 1: Streaming Ingestion
Watch folders, webhooks, or CDC (change data capture) for new documents.

### Day 2: Live Document Updates
Add new documents to vector store without re-indexing everything.

### Day 3: Incremental Embedding
Embed only new/changed chunks, update vector index incrementally.

### Day 4: Live Feed RAG
Build RAG over live news, RSS feeds, or internal updates.

**Deliverable:** `expert_rag/realtime_rag.py` with file watcher

---

## Week 20 — Open Source + Final Expert Project

### Day 1: Choose OSS Project
LangChain, LlamaIndex, RAGAS, Chroma, Qdrant, or similar.

### Day 2: Find Good-First-Issue
Label: "good first issue", "help wanted", documentation, bug fix.

### Day 3: Submit PR
Fork, create branch, implement, test, open PR.

### Day 4: Expert Project
Combine everything: fine-tuned embeddings + graph RAG + multi-modal + agentic.

**Deliverable:** 2+ merged OSS PRs + final expert RAG project

---

## Phase 5 Hard Gate
- [ ] Fine-tuned embedding model outperforms base by 10%+ on domain eval
- [ ] Text-to-SQL + vector hybrid RAG working
- [ ] Real-time ingestion pipeline working
- [ ] 2+ OSS PRs merged
- [ ] Final expert project with eval report
