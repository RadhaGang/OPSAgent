import streamlit as st
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


# ==================================================
# CONFIG
# ==================================================

DATA_PATH = r"C:\Users\Radha\Desktop\GenAI\OPSAgent\knowledge_base"
INDEX_PATH = r"C:\Users\Radha\Desktop\GenAI\OPSAgent\faiss_index"

st.set_page_config(page_title="Local RAG Assistant")
st.title("📚 Local RAG Assistant")


# ==================================================
# VECTOR STORE
# ==================================================

@st.cache_resource
def create_vectorstore():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Load existing FAISS index
    if Path(INDEX_PATH).exists():

        st.info("Loading existing FAISS index...")

        return FAISS.load_local(
            INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

    st.info("Creating FAISS index for first time...")

    docs = []

    folder = Path(DATA_PATH)

    if not folder.exists():
        st.error(f"Folder not found: {DATA_PATH}")
        st.stop()

    for file in folder.rglob("*"):

        try:

            if file.suffix.lower() == ".pdf":

                loader = PyPDFLoader(str(file))
                loaded_docs = loader.load()

            elif file.suffix.lower() == ".docx":

                loader = Docx2txtLoader(str(file))
                loaded_docs = loader.load()

            elif file.suffix.lower() == ".txt":

                loader = TextLoader(
                    str(file),
                    encoding="utf-8"
                )
                loaded_docs = loader.load()

            else:
                continue

            for doc in loaded_docs:
                doc.metadata["source"] = str(file)

            docs.extend(loaded_docs)

        except Exception as e:
            st.warning(f"Could not load {file}: {e}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    vectorstore.save_local(INDEX_PATH)

    st.success("FAISS index created and saved")

    return vectorstore


# ==================================================
# LLM
# ==================================================

@st.cache_resource
def load_model():

    return pipeline(
        task="text2text-generation",
        model="google/flan-t5-base",
        max_new_tokens=200,
        do_sample=False
    )


# ==================================================
# LOAD COMPONENTS
# ==================================================

vectorstore = create_vectorstore()

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5}
)

generator = load_model()


# ==================================================
# UI
# ==================================================

question = st.text_input(
    "Ask a question about your documents"
)

if question:

    with st.spinner("Searching documents..."):

        docs = retriever.invoke(question)

        context = "\n\n".join(
            doc.page_content
            for doc in docs
        )

        prompt = f"""
You are a document assistant.

Answer only using the supplied context.

If the answer is not available in the context,
say:
'Information not found in documents.'

Context:
{context}

Question:
{question}

Answer:
"""

        result = generator(prompt)

        answer = result[0]["generated_text"]

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Source Documents")

    displayed = set()

    for doc in docs:

        source = doc.metadata.get(
            "source",
            "Unknown"
        )

        if source not in displayed:

            st.write(source)

            displayed.add(source)

    with st.expander("Retrieved Context"):

        for doc in docs:

            st.markdown(
                f"**Source:** {doc.metadata.get('source')}"
            )

            st.write(doc.page_content[:1000])

            st.divider()