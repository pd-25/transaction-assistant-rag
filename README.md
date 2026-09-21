# Transaction Assistant RAG

A basic Retrieval-Augmented Generation (RAG) project for working with transaction data using LangChain and LLM-based question answering.

## Overview

This project is designed to help users ask natural-language questions about transactions, such as:

- Which expenses were highest last month?
- What were my largest recurring transactions?
- Show transactions related to food or travel.
- Summarize spending patterns from the dataset.

Instead of only relying on a model's general knowledge, the project retrieves relevant transaction records first and then uses an LLM to generate a grounded answer based on the actual data.

## What this project does

The basic RAG flow is:

1. Load transaction records from a JSON dataset.
2. Convert transaction data into chunks or documents.
3. Create embeddings for the data.
4. Store embeddings in a vector database or in-memory retriever.
5. Retrieve the most relevant records for a user query.
6. Send the retrieved context to an LLM for final response generation.

## Tech stack

- Python
- LangChain
- LangChain Community
- Hugging Face embeddings
- Groq LLM integration
- Sentence Transformers
- dotenv for environment management

## Project structure

```text
transaction-assistant-rag/
├── main.py                  # Application entry point
├── transactions.json        # Transaction dataset
├── experiments.ipynb        # Notebook for experimentation
├── pyproject.toml           # Python project config and dependencies
├── README.md                # Project documentation
└── .env                     # Local environment variables (not committed)
```

## Prerequisites

- Python 3.13+
- pip or uv installed
- A valid LLM provider API key (for example Groq)

## Setup

1. Clone the repository.
2. Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

If you are using the project configuration in `pyproject.toml`, you can also install with:

```bash
pip install -e .
```

4. Create a `.env` file with your environment variables:

```env
GROQ_API_KEY=your_api_key_here
```

## Run the project

```bash
python main.py
```

You can extend `main.py` to:

- load the transaction dataset
- build embeddings
- create a retriever
- run a query against the RAG pipeline
- display the final answer

## Example use cases

- Ask about recent spending patterns
- Find high-value transactions
- Understand category-wise spending
- Answer finance questions using transaction evidence

## Notes

This repository is a starter template for a transaction-focused RAG pipeline. You can expand it by adding:

- better document chunking for transactions
- a persistent vector database
- a FastAPI backend
- a frontend UI
- more advanced agent workflows

## License

This project is currently provided for educational and experimental use.
