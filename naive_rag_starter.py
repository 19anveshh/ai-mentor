"""Complete naive RAG starter: load PDFs, split, embed, store, retrieve, generate."""
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama  # or use OpenAI, etc.

# 1. Load documents
def load_documents(folder: str):
    docs = []
    for path in Path(folder).rglob("*"):
        if path.suffix == ".pdf":
            docs.extend(PyPDFLoader(str(path)).load())
        elif path.suffix in (".txt", ".md"):
            docs.extend(TextLoader(str(path)).load())
    return docs

# 2. Split
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# 3. Embed + store
embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# 4. RAG chain
rag_prompt = ChatPromptTemplate.from_template(
    "Answer the question using only the following context. "
    "If the answer is not in the context, say 'I don't know'.\n"
    "Context: {context}\nQuestion: {question}"
)

def run_rag(question: str, vectorstore, llm, k: int = 4):
    docs = vectorstore.similarity_search(question, k=k)
    context = "\n".join(d.page_content for d in docs)
    chain = rag_prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    return answer, docs

if __name__ == "__main__":
    print("Naive RAG starter ready. Import run_rag() in your notebook.")