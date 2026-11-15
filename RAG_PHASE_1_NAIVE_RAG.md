# Phase 1 — Beginner RAG: Naive RAG (Weeks 3–4)

## Goal
Build a working naive RAG pipeline and understand its failure modes. By the end, you can load documents, split them, embed them, store in a vector DB, retrieve relevant chunks, and generate answers with citations.

---

## Architecture: Index → Retrieve → Generate

```
Documents → Loader → Text Splitter → Embedding Model → Vector Store
                                                          ↓
User Query → Embedding Model → Similarity Search → Top-k Chunks
                                                          ↓
                                      Prompt (context + question) → LLM → Answer + Citations
```

---

## Week 3 — Build Naive RAG

### Day 1: Document Loading

**What to learn:** PDF, HTML, Markdown, text, Docx loaders; metadata extraction (source, page, date)

**Task:**
```python
# naive_rag/loaders.py
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader

def load_documents(folder: str):
    docs = []
    for path in Path(folder).rglob("*"):
        if path.suffix == ".pdf":
            loader = PyPDFLoader(str(path))
        elif path.suffix == ".md":
            loader = UnstructuredMarkdownLoader(str(path))
        elif path.suffix == ".txt":
            loader = TextLoader(str(path))
        else:
            continue
        docs.extend(loader.load())
    return docs
```

**Deliverable:** Loader that reads PDFs, Markdown, and text from a folder with source metadata.

### Day 2: Text Splitting

**What to learn:** Character splitting, recursive character splitting, token-based splitting; chunk size vs overlap tradeoffs

**Task:** Experiment with 3 strategies on the same document:
- Chunk size 500, overlap 50
- Chunk size 1000, overlap 100
- Chunk size 2000, overlap 200

Measure: number of chunks, chunk coherence, retrieval quality

**Deliverable:** `naive_rag/splitting_experiment.py` with comparison results

### Day 3: Embeddings + Vector Store

**What to learn:** sentence-transformers, OpenAI embeddings, BGE, E5; dimensionality, cost per 1k tokens; Chroma, FAISS, Pinecone

**Task:**
```python
# naive_rag/vector_store.py
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(documents, embedding, persist_directory="./chroma_db")
```

**Deliverable:** Vector store with 100+ document chunks, queryable via similarity search

### Day 4: Retrieval + Generation

**What to learn:** Similarity search, top-k tuning, prompt templates for RAG, citation formatting, "no answer" handling

**Task:** Build a QA chain:
1. Embed user query
2. Retrieve top-k chunks
3. Format context with citations
4. Pass to LLM with system prompt
5. Return answer + source list

**Deliverable:** `naive_rag/qa_chain.py` — working "chat with documents" script

### Day 5: Evaluation with RAGAS

**What to learn:** Faithfulness, answer relevancy, context precision, context recall; building an eval dataset

**Task:** Create 20 QA pairs from your documents, run RAGAS, record scores

**Deliverable:** `EVAL_REPORTS/rag_naive_baseline.json` with RAGAS metrics

### Days 6–7: Sunday ritual + polish + commit

---

## Phase 1 Hard Gate
- [ ] Working naive RAG pipeline (load → split → embed → retrieve → generate)
- [ ] Eval report with RAGAS metrics (faithfulness, answer relevancy, context precision, recall)
- [ ] Cost/latency numbers logged in `COST_LOG.md`
- [ ] Runbook: how to ingest new documents and query

---

## Starter Code: Naive RAG

See `naive_rag_starter.py` in this repo for a complete working example.
