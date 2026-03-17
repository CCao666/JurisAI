import os
import gradio as gr
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_openai import AzureChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_core.messages import HumanMessage, AIMessage
# New import for Re-ranking
from sentence_transformers import CrossEncoder

# --- 1. CONFIGURATION ---
os.environ["AZURE_OPENAI_API_KEY"] = "YOUR_KEY_HERE"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://azv-oai-poc-tst.openai.azure.com/"
LLM_DEPLOYMENT = "azv-oai-gpt-4o-2024-05-13"

# --- 2. ENGINE INITIALIZATION ---
llm = AzureChatOpenAI(azure_deployment=LLM_DEPLOYMENT, api_version="2024-05-01-preview", temperature=0)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize the Re-ranker (This model is specifically designed for re-ranking)
# BAAI/bge-reranker-base is powerful yet small enough to run on most CPUs
print("Loading Re-ranker model...")
reranker = CrossEncoder('BAAI/bge-reranker-base')

def build_vector_store(data_folder):
    """Load PDFs and split into chunks."""
    all_docs = []
    for filename in os.listdir(data_folder):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(data_folder, filename))
            all_docs.extend(loader.load())
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    chunks = text_splitter.split_documents(all_docs)
    return FAISS.from_documents(chunks, embeddings)

# Load or Build Vector Index
DATA_FOLDER = "data"
INDEX_PATH = "faiss_index_storage"
if os.path.exists(INDEX_PATH):
    vector_db = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
else:
    vector_db = build_vector_store(DATA_FOLDER)
    vector_db.save_local(INDEX_PATH)

# --- 3. CUSTOM RE-RANKING LOGIC ---
def re_rank_documents(query, documents, top_n=3):
    """
    Takes the initial retrieved documents and re-scores them 
    against the query using a Cross-Encoder.
    """
    if not documents:
        return []
    
    # Prepare pairs for the Cross-Encoder: [[query, doc1], [query, doc2]...]
    pairs = [[query, doc.page_content] for doc in documents]
    
    # Predict scores (higher is more relevant)
    scores = reranker.predict(pairs)
    
    # Sort documents by their new scores
    scored_docs = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
    
    # Return top N documents
    return [doc for score, doc in scored_docs[:top_n]]

# --- 4. RAG CHAIN CONSTRUCTION ---
def create_chain(vectorstore):
    # Set k=10 for initial retrieval to give Re-ranker enough candidates
    retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Analyze chat history and re-write the user question into a standalone version."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    # This part handles context-awareness
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    system_prompt = (
        "You are JurisAI, a high-precision specialist in US Intellectual Property Law. "
        "Your core expertise covers Copyright (Title 17), Patent (Title 35), and Trademark (Title 15). "
        "\n\n"
        "CRITICAL GUIDELINES:\n"
        "1. SEMANTIC CONTEXT: Many legal terms are polysemous. When you encounter terms like 'transfer', 'infringement', or 'registration', "
        "interpret them STRICTLY within the framework of Intellectual Property rights unless the context explicitly refers to other domains.\n"
        "2. NO OUTSIDE KNOWLEDGE: Use ONLY the provided context. If the context discusses financial transfers instead of copyright transfers, "
        "and you cannot find the relevant IP law in the context, state that the information is missing from the library.\n"
        "3. CITATION RIGOR: You must cite specific Section numbers (§). If multiple titles are provided, prioritize Title 17 for Copyright queries.\n"
        "4. ACCURACY OVER COMPLETION: Do not conflate banking/remittance rules with IP statutes.\n\n"
        "Context:\n{context}"
    )
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    # Note: We will manually handle the re-ranking in the predict function 
    # for better control, so we return the helper components here.
    return history_aware_retriever, qa_chain

history_aware_retriever, qa_chain = create_chain(vector_db)

def predict(message, history):
    """
    Enhanced Gradio interaction logic with Re-ranking and Clean Source Attribution.
    """
    # 1. Convert Gradio history format to LangChain message objects
    langchain_history = []
    for human, ai in history:
        langchain_history.append(HumanMessage(content=human))
        # Strip existing sources from history to prevent context pollution
        clean_ai_content = ai.split("---")[0].strip()
        langchain_history.append(AIMessage(content=clean_ai_content))

    # 2. Step 1: Retrieve standalone question and initial document candidates
    # The history_aware_retriever uses LLM to re-write query based on history
    initial_docs = history_aware_retriever.invoke({
        "input": message, 
        "chat_history": langchain_history
    })

    # 3. Step 2: Apply Re-ranking (Cross-Encoder)
    # Refines the top 10 candidates down to the most relevant top_n
    refined_docs = re_rank_documents(message, initial_docs, top_n=4)

    # 4. Step 3: Generate final answer using refined context
    # We pass the refined_docs directly into the 'context' variable
    result = qa_chain.invoke({
        "input": message, 
        "chat_history": langchain_history,
        "context": refined_docs
    })
    
    # Handle different return types from LangChain (String or Dict)
    answer = result["answer"] if isinstance(result, dict) else result

    # 5. Step 4: Format and Append Source Metadata
    sources = sorted({os.path.basename(d.metadata.get('source', 'Unknown')) for d in refined_docs})
    
    if sources:
        source_str = ", ".join(sources)
        # Use a clear separator and bold formatting
        source_info = f"\n\n---\n📚 **Verified Sources:** {source_str}"
    else:
        source_info = ""

    # 6. Final safety check: ensure the LLM didn't already hallucinate a source line
    if "Verified Sources:" in answer:
        return answer
    
    return f"{answer}{source_info}"

# --- 6. LAUNCH WEB INTERFACE ---
demo = gr.ChatInterface(
    fn=predict, 
    title="⚖️ JurisAI: Pro IP Law Assistant",
    description="A specialized AI chatbot for US Law, powered by Two-Stage Retrieval (FAISS + BGE Re-ranker).",
    examples=["Who owns copyright for contractors?", "What is Section 107?", "How to terminate a transfer?"],
    theme="soft"
)

if __name__ == "__main__":
    demo.launch()