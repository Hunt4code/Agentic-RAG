import streamlit as st
import tempfile
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.ingest import ingest
from app.agent import ask



st.title("RAG Agent — Ask Your Documents")
st.subheader("Step 1 — Load a document")

uploaded_file = st.file_uploader("Upload PDF or DocX",type=["pdf","docx"])
url_input = st.text_input("Or paste a URL")

if uploaded_file:
    save_path = f"data/uploads/{uploaded_file.name}"
    with open(save_path,"wb") as f:
        f.write(uploaded_file.read())
    with st.spinner("Ingesting document.."):
        ingest(save_path)
    st.success(f"Uploaded file : {uploaded_file.name}")

if url_input:
    with st.spinner("Ingesting URL.."):
        ingest(url_input)
    st.success(f"Ingested :{url_input}")


st.subheader("Step 2 — Ask a question")
question = st.text_input("Ask a question")


if st.button("Ask"):
    if not question:
        st.warning("Please enter a question")
    else:
        with st.spinner("Thinking"):
            answer = ask(question)
        st.subheader("Answer")
        st.write(answer)

