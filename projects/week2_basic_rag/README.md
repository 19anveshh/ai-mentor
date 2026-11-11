# Week 2 Progress

## Completed
- Basic RAG retrieval pipeline built and understood
- Code saved at: projects/week2_basic_rag/basic_rag.py

## What was learned
- Document loading with PyPDFLoader
- Text splitting with RecursiveCharacterTextSplitter (chunk_size=500, overlap=50)
- Embedding generation with HuggingFaceEmbeddings (all-MiniLM-L6-v2)
- Vector storage and similarity search with FAISS
- Retrieval returns top-k most relevant chunks

## Next steps
- Add Step 7: LLM generation using retrieved chunks as context
- Build a simple Q&A interface (CLI or Streamlit)
- Add evaluation: check if retrieved chunks actually answer the question
