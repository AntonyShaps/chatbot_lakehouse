from fastapi import FastAPI, Query
from pydantic import BaseModel
from openai import OpenAI
import chromadb
import os
from dotenv import load_dotenv
from typing import List

# Load environment variables
load_dotenv()

# Initialize clients
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
chroma_client = chromadb.PersistentClient(path="../chroma_store")
collection = chroma_client.get_collection("bitmovin_docs")

app = FastAPI()

class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_question(request: QueryRequest):
    question = request.question

    # Step 1: Embed query
    try:
        embedding_response = client.embeddings.create(
            input=question,
            model="text-embedding-3-small",
            dimensions=128
        )
        query_embedding = embedding_response.data[0].embedding
    except Exception as e:
        return {"error": f"Embedding failed: {str(e)}"}

    # Step 2: Vector search
    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )
    except Exception as e:
        return {"error": f"ChromaDB query failed: {str(e)}"}

    # Step 3: Build GPT-4 context
    contexts = []
    sources = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        context_text = f"Title: {meta['title']}\nURL: {meta['url']}\nContent:\n{doc}"
        contexts.append(context_text)
        sources.append(meta["url"])

    context_for_gpt = "\n\n---\n\n".join(contexts)

    # Step 4: Ask GPT-4
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant answering questions based on internal Bitmovin documentation provided below."},
                {"role": "user", "content": f"""Use the following documentation to answer the question.

DOCUMENTATION:
{context_for_gpt}

QUESTION:
{question}

Add links to the documentation at the end of your answer.
"""}
            ],
            temperature=0.3
        )
        answer = response.choices[0].message.content
    except Exception as e:
        return {"error": f"GPT-4 completion failed: {str(e)}"}

    return {
        "answer": answer,
        "sources": sources
    }
