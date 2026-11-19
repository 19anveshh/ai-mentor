# Phase 2 — Intermediate RAG Projects

## multi_query_rag.py

**Technique:** Multi-Query Retrieval (Query Rewriting)

**What it does:**
- Loads a PDF and splits it into chunks
- Embeds chunks and stores in FAISS vector database
- Takes a user question and rewrites it into 3 alternative queries using an LLM
- Searches FAISS for each query and merges unique results
- Generates a final answer using all retrieved context

**Why query rewriting matters:**
Users often ask vague or poorly worded questions. Multi-query retrieval generates multiple variations of the same question, increasing the chance of finding relevant chunks that a single query might miss.

**Run:**
```bash
python projects/phase_2_intermediate/multi_query_rag.py
```

**Requirements:**
- `.env` file with `GOOGLE_API_KEY`
- `uv sync` to install dependencies
- PDF file at the path specified in the script

**Related roadmap:** RAG_PHASE_2_INTERMEDIATE_RAG.md, Week 5 — Query Transformation
