from app.retriever import load_retriever
from app.ingest import ingest
from app.agent import ask
from langchain_community.document_loaders import PyMuPDFLoader
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper



#Test ingest
#vectorstore = ingest("data/uploads/Hrishikesh_Balakrishnan_Resume.pdf")

#Test retriever
# retriever = load_retriever()
# results = retriever.invoke("work experience TCS KGS")

# for i, doc in enumerate(results):
#     print(f"\n--- Chunk {i+1} ---")
#     print(doc.page_content)

#Test agent
questions = [
    "What is Hrishikesh's work experience?",
    "What programming languages does he know?",
    "What is his education background?",
    "What is the capital of France?"  
]

for q in questions:
    print(f"\nQ: {q}")
    print(f"A: {ask(q)}")

#Test PDFLoader
# loader = PyMuPDFLoader("data/uploads/Hrishikesh_Balakrishnan_Resume.pdf")
# docs = loader.load()
# print(docs[0].page_content)


