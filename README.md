# Agentic-RAG

Tired of reading through lengthy documents or web pages to find a specific answer? 
Agentic-RAG lets you query any PDF, Word document, or URL and get accurate answers 
in seconds. Built with LangGraph's conditional edge routing — the agent retrieves, 
grades relevance, and retries automatically until it finds the optimal answer.

## Architecture

[ReAct-style agentic loop]
User Question → Retrieve Chunks → Grade Relevance → Answer (or Retry)

- If retrieved chunks are relevant → generate answer
- If not relevant → rephrase and retry (up to 3 times)
- If retries exhausted → answer with best available context

## Features

- Multi-source ingestion — PDF, DOCX, and URLs
- LangGraph stateful agent with retrieve-grade-answer loop
- Automatic retry when retrieved chunks are not relevant
- Anthropic Claude as the LLM backbone
- Local HuggingFace embeddings with ChromaDB vector store
- Streamlit chat interface
- RAGAS evaluation suite with faithfulness, relevancy, precision and recall metrics

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Agent orchestration | LangGraph |
| LLM backbone | Anthropic Claude (claude-sonnet-4-20250514) |
| Embeddings | HuggingFace sentence-transformers (all-MiniLM-L6-v2) |
| Vector store | ChromaDB |
| LLM abstractions | LangChain |
| Frontend | Streamlit |
| Evaluation | RAGAS |

## RAGAS Evaluation Results

Evaluated on 5 questions against a ground truth dataset using Anthropic Claude as the judge LLM.

| Question | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|----------|-------------|-----------------|-------------------|----------------|
| Years of experience | 1.00 | 0.84 | 1.00 | 1.00 |
| University attended | 0.75 | 0.81 | 1.00 | 1.00 |
| Senior SWE company | 1.00 | 0.57 | 0.50 | 1.00 |
| Current location | 1.00 | 1.00 | 1.00 | 1.00 |
| Current role | 0.50 | 0.00 | 0.00 | 0.00 |
| **Average** | **0.85** | **0.64** | **0.70** | **0.80** |

**Key observations:**

- Faithfulness of 0.85 means answers are well grounded in retrieved context
- Context recall of 0.80 means the retriever finds most of the needed information
- Q5 (current role) scored low due to chunk boundary issues — identified as an area for improvement

## Project Structure

Agentic-RAG/
├── app/
│   ├── ingest.py        # Multi-source loader — PDF, DOCX, URL chunking and embedding
│   ├── retriever.py     # ChromaDB similarity search
│   ├── agent.py         # LangGraph agentic RAG — retrieve, grade, answer loop
│   ├── agent_linear.py  # Linear RAG chain — baseline version
│   └── ui.py            # Streamlit chat interface
├── evaluate.py          # RAGAS evaluation suite
├── test.py              # Development testing scripts
├── requirements.txt     # Python dependencies
└── Dockerfile           # Container configuration

## Setup

### Prerequisites
- Python 3.11+
- Anthropic API key — get one at console.anthropic.com

### Installation
```bash
git clone https://github.com/Hunt4code/Agentic-RAG.git
cd Agentic-RAG
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file:

ANTHROPIC_API_KEY=your-key-here
ANTHROPIC_MODEL=claude-sonnet-4-20250514
EMBED_MODEL=all-MiniLM-L6-v2
CHROMA_PATH=./data/chroma
USER_AGENT=agentic-rag/1.0

### Run
```bash
streamlit run app/ui.py
```

Open `http://localhost:8501`

### Evaluate
```bash
python evaluate.py
```

## How It Works

1. **Ingest** — document is loaded, split into overlapping chunks, embedded and stored in ChromaDB
2. **Retrieve** — question is embedded, ChromaDB returns top 3 semantically similar chunks
3. **Grade** — Claude evaluates whether chunks are relevant to the question
4. **Answer or Retry** — relevant chunks go to answer node, irrelevant trigger retry up to 3 times

## Why LangGraph over a Simple Chain

A standard chain runs fixed steps with no decisions. LangGraph enables a stateful graph where the agent loops, retries, and routes based on intermediate results — better answer quality on complex queries.

## Future Improvements

- Multi-document support with per-document ChromaDB collections
- LangSmith tracing for production observability
- Upgraded UI with Next.js
- Support for scanned PDFs via OCR
- Swap to text-embedding-3-small for higher quality retrieval