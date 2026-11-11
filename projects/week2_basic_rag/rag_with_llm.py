# Week 2 - RAG with LLM Generation (Full Pipeline)
# Steps: Load PDF -> Split -> Embed -> FAISS -> Retrieve -> Context -> LLM -> Answer

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage

# 1. Load PDF
loader = PyPDFLoader("sample.pdf")
documents = loader.load()

# 2. Split documents into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)

# 3. Create embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# 4. Store chunks and embeddings in FAISS
vector_store = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)

# 5. Retrieve relevant chunks
query = "What are the easy questions in module 1?"
results = vector_store.similarity_search(query, k=3)

# 6. Convert retrieved documents into context
context = "\n\n".join(
    doc.page_content for doc in results
)

# 7. Initialize LLM
model = init_chat_model(
    model="YOUR_GROQ_MODEL_NAME",  # Replace with actual model name, e.g. "llama-3.1-8b-instant"
    model_provider="groq",
    temperature=0.3
)

# 8. Send context + question to LLM
response = model.invoke([
    SystemMessage(content="""
    You are a senior mathematics teacher.

    Answer using the provided PDF context.
    Explain clearly in simple English.
    If the answer is not available in the context,
    say that the PDF does not contain the information.
    """),

    HumanMessage(content=f"""
    PDF CONTEXT:
    {context}

    STUDENT QUESTION:
    {query}
    """)
])

print(response.content)