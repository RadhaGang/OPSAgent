import streamlit as st
import requests

st.set_page_config(page_title="Local RAG Assistant")
st.title("📚 Local RAG Assistant")

question = st.text_input("Ask a question about your documents")

if question:

    with st.spinner("Thinking..."):

        response = requests.post(
            "http://localhost:8000/ask",
            json={"question": question}
        )

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