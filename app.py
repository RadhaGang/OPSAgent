import os
import streamlit as st
import requests

st.set_page_config(page_title="Local RAG Assistant")
st.title("📚 Local RAG Assistant")

# Reads the FastAPI backend URL from environment variable
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

question = st.text_input("Ask a question about your documents")

if question:

    with st.spinner("Thinking..."):

        try:
            response = requests.post(
                f"{BACKEND_URL}/ask",
                json={"question": question},
                timeout=60
            )
            response.raise_for_status()
            data = response.json()

            st.subheader("Answer")
            st.write(data["answer"])

            st.subheader("Sources")
            for src in data["sources"]:
                st.write(src)

            with st.expander("Retrieved Context"):
                for doc in data["context_docs"]:
                    st.markdown(f"**Source:** {doc['source']}")
                    st.write(doc["content"])
                    st.divider()

        except requests.exceptions.ConnectionError:
            st.error(f"❌ Could not c