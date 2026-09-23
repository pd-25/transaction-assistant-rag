from fastapi import FastAPI
from pydantic import BaseModel
import os, json
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

load_dotenv(override=True)

os.environ["HF"] = os.getenv("HF")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found")

app = FastAPI(title="Transaction RAG Application", version="0.1.0")

# model
LLM = ChatGroq(
    model=os.getenv("GROQ_MODEL"), api_key=os.getenv("GROQ_API_KEY"), temperature=0
)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def format_transaction(record: dict, metadata: dict):
    metadata["id"] = record.get("id")
    metadata["status"] = record.get("status")
    metadata["type"] = record.get("type")
    metadata["amount"] = record.get("amount")
    metadata["name"] = record.get("name")
    return metadata


def load_docs():
    loader = JSONLoader(
        file_path="transactions.json",
        jq_schema=".[]",
        text_content=False,
        metadata_func=format_transaction,
    )
    docs = loader.load()

    for doc in docs:
        data = json.loads(doc.page_content)
        direction = "Sent to" if data.get("type") == "debit" else "Recived from"
        doc.page_content = (
            f"Transaction ID: {data.get("id")} | "
            f"{direction} {data.get("name")}    ({data.get("upiId")}) | "
            f"Amount: {data.get("amount")} | "
            f"Status: {data.get("status")} | "
            f"Method: {data.get("paymentMethod")} | "
            f"Date: {data.get("date")} | "
            f"Note: {data.get("note")} | "
            f"Bank Ref: {data.get("bankRef")}"
        )


# build chain
docs = load_docs()
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(
    search_type="mmr", search_kwargs={"k": 20, "fetch_k": 30}
)

SYSTEM_PROMT = """You are a transaction assistant for phonepe style payment app.
        Make sure the answer is short and clean.
        For long answer in point style.    
        Answer only from the context provided. Do not guess or hallucinate.
        Use ruppe symbol from amount. Keep answer short and simple
        <context>
        {context}
        </context>
        """

# 47.50
