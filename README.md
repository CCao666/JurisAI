# ⚖️ JurisAI: Pro Intellectual Property Law Assistant

**JurisAI** is a high-precision RAG (Retrieval-Augmented Generation) assistant specialized in US Intellectual Property Law (Copyright, Patent, and Trademark). It navigates complex legal statutes to provide accurate answers with direct citations to the US Code.



## ✨ Key Features

* **Two-Stage Retrieval Pipeline**: 
    * **Phase 1 (Recall)**: Uses FAISS and HuggingFace Embeddings for fast candidate retrieval from legal PDFs.
    * **Phase 2 (Re-ranking)**: Integrates the **BGE-Reranker** (Cross-Encoder) to re-score candidates, ensuring the most contextually relevant legal sections are prioritized.
* **Context-Aware Reasoning**: Maintains conversation history to handle follow-up questions and pronoun references (e.g., "Does *this right* apply to...").
* **Statutory Citations**: Automatically identifies and cites specific sections (e.g., § 203, § 107) from the source material.
* **Domain Specificity**: Specifically tuned to distinguish between similar legal concepts across Title 15, 17, and 35 of the U.S.C.

## 📺 Project Demo

[![JurisAI Project Demo](https://img.youtube.com/vi/9rgE5FexYv0/maxresdefault.jpg)](https://www.youtube.com/watch?v=9rgE5FexYv0)

> *💡 Click the image above to watch the full JurisAI demo on YouTube.*


## 🏗️ Tech Stack

* **LLM**: Azure OpenAI (GPT-4o)
    * **Orchestration**: LangChain
    * **Vector Database**: FAISS (Facebook AI Similarity Search)
    * **Embeddings**: `all-MiniLM-L6-v2`
    * **Re-ranker**: `BAAI/bge-reranker-base`
    * **UI**: Gradio

## 📂 Project Structure

```text
├── assets/              # Demo videos and architecture diagrams
├── data/                # US Code PDFs (Title 15, 17, 35)
├── .env.example         # Template for environment variables
├── .gitignore           # Excludes local cache and credentials
├── main.py              # CLI entry point
├── web_app.py           # Gradio Web UI entry point
└── requirements.txt     # Project dependencies

```
