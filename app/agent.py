from langgraph.graph import StateGraph,END
from langchain_ollama import ChatOllama
from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from typing import TypedDict, List
from app.retriever import load_retriever
import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

class AgentState(TypedDict):
    question : str
    chunks : List[Document]
    relevant : bool
    answer: str
    retries: int

PROMPT = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the question using only the context provided.
If the answer is not in the context, say "I don't know based on the provided document."

Context:
{context}

Question:
{question}

Answer:
""")

def retrieve_node(state:AgentState)-> AgentState:

    question = state["question"]
    retriever = load_retriever()

    chunks = retriever.invoke(question)


    return {"chunks": chunks,"retries": state.get("retries", 0) + 1}

def grade_node(state : AgentState)-> AgentState:

    question = state["question"]
    chunks = state["chunks"]

    context = "\n\n".join([chunk.page_content for chunk in chunks])

    #llm = ChatOllama(model=OLLAMA_MODEL)
    llm = ChatAnthropic(model=ANTHROPIC_MODEL)
    prompt = f"Are these chunks relevant to answer this question?\nQuestion: {question}\nChunks: {context}\nReply with only yes or no."
    
    response = llm.invoke(prompt)
    relevant = "yes" in response.content.lower()

    return {"relevant": relevant}

def answer_node(state: AgentState)->AgentState:

    question = state["question"]
    chunks = state["chunks"]

    context = "\n\n".join([chunk.page_content for chunk in chunks])
    #llm = ChatOllama(model=OLLAMA_MODEL)
    llm = ChatAnthropic(model=ANTHROPIC_MODEL)
    chain = PROMPT | llm
    response = chain.invoke({"context":context,"question":question})

    return {"answer":response.content}

def should_continue(state:AgentState)->str:
    if state["relevant"]:
        return "answer"
    elif state.get("retries",0)>=2:
        return "answer"
    else:
        return "retrieve"
    

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("retrieve",retrieve_node)
    graph.add_node("grade",grade_node)
    graph.add_node("answer",answer_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve","grade")

    graph.add_conditional_edges("grade",
                                should_continue,
                                {
                                    "answer":"answer",
                                    "retrieve": "retrieve"
                                }

                                )
    graph.add_edge("answer",END)

    return graph.compile()

def ask(question:str)->str:
    app = build_graph()
    result = app.invoke({"question":question})

    return result["answer"]


    