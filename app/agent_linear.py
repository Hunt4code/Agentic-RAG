from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from app.retriever import load_retriever
import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

PROMPT = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the question using only the context provided below.
If the answer is not in the context, say "I don't know based on the provided document."

Context:
{context}

Question:
{question}

Answer:
""")

def ask(question:str):

    retriever = load_retriever()
    chunks = retriever.invoke(question)

    # context = ""
    # for chunk in chunks:
    #     context = context + "\n\n" + chunk.page_content

    context = "\n\n".join([chunk.page_content for chunk in chunks])
    
    llm = ChatOllama(model=OLLAMA_MODEL)

    chain = PROMPT | llm
    response = chain.invoke({"context": context, "question": question})

    return response.content
