import os
import json
import faiss
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
import google.generativeai as genai
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

#01. Setup
load_dotenv()
genai.configure(api_key=os.getenv("Enter your Gemini API key here"))
# Note: The API key should be set in your environment variables.

# Incase of an error occuring, please check the API key and ensure it is set correctly in your environment variables.
# If you are using a different API key, please update the line above with your key.

#02. Load and chunk PDF
def load_textbook(path):
    loader = PyPDFLoader(path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)

#03. Embed textbook
def embed_chunks(chunks):
    texts = [c.page_content for c in chunks]
    model = genai.get_model("models/embedding-001")
    embeddings = model.embed_content(content=texts, task_type="retrieval_document")["embedding"]
    return np.array(embeddings).astype("float32")
        
#04. Retrieve Relevant chunks
def retrieve_context(query, index, chunks, k=5):
    embed_model = genai.get_model("models/embedding-001")
    q_embed = embed_model.embed_content(content=query, task_type="retrieval_query")["embedding"]
    q_vec = np.array(q_embed).astype("float32").reshape(1, -1)
    _, I = index.search(q_vec, k)
    return [chunks[i].page_content for i in I[0]], [chunks[i].metadata.get("page", "Unknown") for i in I[0]]

#05. Generates answer
def generate_answer(query, context_chunks):
    prompt = f"""You are a helpful history tutor. Use ONLY the context below to answer the question accurately.

Context:
{'\n\n'.join(context_chunks)}

Question: {query}
"""
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text.strip()

#06. Summarizes the answer
def summarize_answer(answer):
    prompt = f"""Summarize the following answer in 1-2 sentences for a student who wants a quick overview:

Answer:
{answer}
"""
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text.strip()

# ========== 6. Run Over Queries ==========
def run_chatbot():
    print("📘 Loading and embedding textbook...")
    chunks = load_textbook("Grade_11_History_Textbook.pdf")
    embeddings = embed_chunks(chunks)

    print("🧠 Building FAISS index...")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    print("📄 Loading queries...")
    with open("queries.json") as f:
        queries = json.load(f)

    results = []

    print("🚀 Generating answers...\n")
    for q in tqdm(queries, desc="Answering queries"):
        qid = q["ID"]
        user_query = q["context"]

        context, pages = retrieve_context(user_query, index, chunks)
        answer = generate_answer(user_query, context)
        summary = summarize_answer(answer)

        results.append({
            "ID": qid,
            "Context": " | ".join(context),
            "Answer": answer,
            "Summary": summary,
            "Sections": json.dumps(q["references"]["sections"]),
            "Pages": json.dumps(q["references"]["pages"])
        })

    print("\n✅ Saving submission...")
    df = pd.DataFrame(results)
    df.to_csv("future_minds_submission.csv", index=False)

    with open("pipeline.json", "w") as f:
        json.dump({
            "textbook_loader": "PyPDFLoader",
            "chunking": "RecursiveCharacterTextSplitter",
            "embedding_model": "Gemini Embedding-001",
            "vector_database": "FAISS",
            "retrieval": "FAISS similarity search",
            "generation_model": "gemini-1.5-flash",
            "summarizer": "Gemini summarizer agent"
        }, f, indent=4)

    print("🎉 Submission complete! Files saved.")

if __name__ == "__main__":
    run_chatbot()