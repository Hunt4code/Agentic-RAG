from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from app.agent import ask
from app.retriever import load_retriever
import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings


llm = LangchainLLMWrapper(ChatAnthropic(
    model="claude-sonnet-4-20250514",
    api_key=os.getenv("ANTHROPIC_API_KEY")
))

# configure RAGAS to use local models
#llm = LangchainLLMWrapper(ChatOllama(model="llama3.2"))
embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
))

testdata = [
    {
        "question": "How many years of experience does Hrishikesh have?",
        "ground_truth": "4+ years"
    },
    {
        "question": "What university did Hrishikesh attend for his Masters?",
        "ground_truth": "University of Massachusetts Boston"
    },
    {
        "question": "What company did he work at as Senior Software Engineer?",
        "ground_truth": "Tata Consultancy Services"
    },
    {
        "question": "What is Hrishikesh's current location?",
        "ground_truth": "Boston, Massachusetts"
    },
    {
        "question": "What is Hrishikesh's current role?",
        "ground_truth": "Data Scientist at CrowdDoing since August 2024"
    }
]

def run_eval():

    retriever = load_retriever()

    all_questions = []
    all_contexts = []
    all_answers = []
    all_ground_truths = []

    for item in testdata:

        q = item["question"]
        gt = item["ground_truth"]

        answer = ask(q)
        chunks = retriever.invoke(q)

        context = [chunk.page_content for chunk in chunks]

        all_questions.append(q)
        all_contexts.append(context)
        all_answers.append(answer)
        all_ground_truths.append(gt)
        print(f"Done : {q}")

        dataset = Dataset.from_dict({
            "question": all_questions,
            "answer": all_answers,
            "contexts": all_contexts,
            "ground_truth": all_ground_truths
        })

        results = evaluate(dataset = dataset,metrics=[
             faithfulness,
             answer_relevancy,
             context_precision,
             context_recall
        ],
        llm = llm,
        embeddings = embeddings)

        df = results.to_pandas()

        print("\n=== RAGAS Evaluation Results ===")
        print(df[["user_input", "faithfulness", "answer_relevancy", 
          "context_precision", "context_recall"]])
        print("\n=== Average Scores ===")
        print(df[["faithfulness", "answer_relevancy", 
          "context_precision", "context_recall"]].mean())
        
    return df
    
if __name__ == "__main__":
    run_eval()