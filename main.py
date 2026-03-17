import os
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_openai import AzureChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_core.messages import HumanMessage, AIMessage
from sentence_transformers import CrossEncoder

# --- 1. CONFIGURATION ---
os.environ["AZURE_OPENAI_API_KEY"] = "YOUR_KEY_HERE"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://azv-oai-poc-tst.openai.azure.com/"
LLM_DEPLOYMENT = "azv-oai-gpt-4o-2024-05-13"

# --- 2. INITIALIZATION ---
llm = AzureChatOpenAI(azure_deployment=LLM_DEPLOYMENT, api_version="2024-05-01-preview", temperature=0)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
reranker = CrossEncoder('BAAI/bge-reranker-base')

# Load Index
INDEX_PATH = "faiss_index_storage"
vector_db = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

# --- 3. CORE LOGIC (RE-RANKING) ---
def re_rank_documents(query, documents, top_n=4):
    if not documents: return []
    pairs = [[query, doc.page_content] for doc in documents]
    scores = reranker.predict(pairs)
    scored_docs = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
    return [doc for score, doc in scored_docs[:top_n]]

# --- 4. CHAIN SETUP ---
def get_legal_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    
    # Contextualize Question
    context_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Re-write the user question to be standalone based on chat history."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, context_q_prompt)

    # QA Prompt
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
    return history_aware_retriever, qa_chain

history_aware_retriever, qa_chain = get_legal_chain(vector_db)

# --- 5. CLI INTERACTION LOOP ---
def run_cli():
    chat_history = []
    print("\n" + "="*50)
    print("⚖️  LexiCounsel AI - Terminal Edition")
    print("Type 'exit' or 'quit' to stop.")
    print("="*50)

    

    while True:
        user_input = input("\n👤 User: ")
        if user_input.lower() in ['exit', 'quit']:
            break

        # Step 1: Retrieval with History
        initial_docs = history_aware_retriever.invoke({
            "input": user_input, 
            "chat_history": chat_history
        })

        # Step 2: Re-rank
        refined_docs = re_rank_documents(user_input, initial_docs)

        # Step 3: Generate
        result = qa_chain.invoke({
            "input": user_input, 
            "chat_history": chat_history,
            "context": refined_docs
        })
        
        answer = result["answer"] if isinstance(result, dict) else result
        
        # Format Sources for CLI
        sources = sorted({os.path.basename(d.metadata.get('source', 'Unk')) for d in refined_docs})
        
        print(f"\n🤖 AI: {answer}")
        print(f"\n📚 Sources: {', '.join(sources)}")
        print("-" * 30)

        # Update History
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=answer))

if __name__ == "__main__":
    run_cli()