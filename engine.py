from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline

DATA_PATH = r"C:\Users\Radha\Desktop\GenAI\OPSAgent\knowledge_base"
INDEX_PATH = r"C:\Users\Radha\Desktop\GenAI\OPSAgent\faiss_index"


# =========================
# VECTOR STORE
# =========================
def get_vectorstore():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    if Path(INDEX_PATH).exists():

        vectorstore = FAISS.load_local(
            INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

        return vectorstore

    docs = []
    folder = Path(DATA_PATH)

    for file in folder.rglob("*"):

        try:
            if file.suffix.lower() == ".pdf":
                loader = PyPDFLoader(str(file))

            elif file.suffix.lower() == ".docx":
                loader = Docx2txtLoader(str(file))

            elif file.suffix.lower() == ".txt":
                loader = TextLoader(str(file), encoding="utf-8")

            else:
                continue

            loaded_docs = loader.load()

            for doc in loaded_docs:
                doc.metadata["source"] = str(file)

            docs.extend(loaded_docs)

        except Exception as e:
            print(f"Error loading {file}: {e}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(INDEX_PATH)

    return vectorstore


# =========================
# LLM
# =========================
def get_llm():

    return pipeline(
        task="text2text-generation",
        model="google/flan-t5-base",
        max_new_tokens=200,
        do_sample=False
    )


# =========================
# RAG PIPELINE
# =========================
vectorstore = get_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
generator = get_llm()


def ask_question(question: str):

    docs = retriever.invoke(question)

    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""
You are a document assistant.

Answer only using the context.

If answer not found, say:
"Information not found in documents."

Context:
{context}

Question:
{question}

Answer:
"""

    result = generator(prompt)
    answer = result[0]["generated_text"]

    sources = list(set(
        doc.metadata.get("source", "Unknown")
        for doc in docs
    ))

    return {
        "answer": answer,
        "sources": sources,
        "context_docs": [
            {
                "source": doc.metadata.get("source"),
                "content": doc.page_content[:500]
            }
            for doc in docs
        ]
    }