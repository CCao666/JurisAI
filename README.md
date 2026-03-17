# ⚖️ JurisAI: Pro Intellectual Property Law Assistant

**JurisAI** is a high-precision RAG (Retrieval-Augmented Generation) assistant specialized in US Intellectual Property Law. It navigates complex legal statutes to provide accurate answers with direct citations to the US Code.

## 🚀 Project Status: MVP
Currently, **JurisAI** serves as an **MVP (Minimum Viable Product)** focusing on three core pillars of IP Law:
* **Copyright** (Title 17 U.S.C.)
* **Patent** (Title 35 U.S.C.)
* **Trademark** (Title 15 U.S.C.)

### 🌐 Scalability & Future Scope
While the current version is optimized for Intellectual Property, the underlying **Two-Stage Retrieval Architecture** is domain-agnostic and highly scalable. By simply updating the source documents in the `/data` folder, the system can be seamlessly adapted to any domain of **US Law** (e.g., Contract Law, Tax Law, or Civil Procedure).



## ✨ Key Features

* **Two-Stage Retrieval Pipeline**: 
    * **Phase 1 (Recall)**: Uses FAISS and HuggingFace Embeddings for fast candidate retrieval from legal PDFs.
    * **Phase 2 (Re-ranking)**: Integrates the **BGE-Reranker** (Cross-Encoder) to re-score candidates, ensuring the most contextually relevant legal sections are prioritized.
* **Context-Aware Reasoning**: Maintains conversation history to handle follow-up questions and pronoun references.
* **Statutory Citations**: Automatically identifies and cites specific sections (e.g., § 203, § 107) from the source material.
* **Domain Specificity**: Specifically tuned to distinguish between similar legal concepts across the U.S.C.

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

## 🛠️ Getting Started

### 1. Installation
Clone the repository and install the required dependencies:
```bash
git clone [https://github.com/CCao666/JurisAI.git](https://github.com/CCao666/JurisAI.git)
cd JurisAI
pip install -r requirements.txt
```

### 2. Configuration
Create a `.env` file in the root directory and add your credentials:

```text
AZURE_OPENAI_API_KEY=your_api_key_here
AZURE_OPENAI_ENDPOINT=your_endpoint_here
```

### 3. Running the Application
* **Web UI (Gradio)**: `python web_app.py`
* **CLI Version**: `python main.py`

> **Note**: On the first run, the system will automatically parse the PDFs in the `/data` folder and build the local FAISS index.

---

## 🧠 System Architecture

JurisAI implements a sophisticated **Two-Stage Retrieval** mechanism to ensure legal accuracy:



* **Semantic Search (FAISS)**: The system chunks legal documents and uses `all-MiniLM-L6-v2` to retrieve the top 10 most similar fragments based on vector similarity.
* **Cross-Encoder Re-ranking (BGE)**: To filter out "hallucinated" relevance, the `bge-reranker-base` model performs a deep semantic comparison between the user's specific query and the 10 candidates, selecting only the most legally pertinent sections.
* **Prompt Engineering**: The final response is generated using a strictly constrained prompt that forces the LLM to provide citations in the `§ Section` format and admit if the information is not present in the provided context.

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## ⚖️ Disclaimer
This project is for educational and research purposes only. It is not intended to provide legal advice. Always consult with a qualified legal professional for official matters.
