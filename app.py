import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Set page config
st.set_page_config(
    page_title="Document Q&A Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "initialized" not in st.session_state:
    st.session_state.initialized = False

# Set API key - try Streamlit secrets first, then environment variable, then hardcoded (for local dev)
try:
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    elif "GOOGLE_API_KEY" not in os.environ:
        # Fallback for local development
        os.environ["GOOGLE_API_KEY"] = "AIzaSyA32QT_Nb6f2-6NxG31ZMx6AxBwbrTOOIw"
except Exception:
    # No secrets file found, use environment variable or fallback
    if "GOOGLE_API_KEY" not in os.environ:
        os.environ["GOOGLE_API_KEY"] = "AIzaSyA32QT_Nb6f2-6NxG31ZMx6AxBwbrTOOIw"

@st.cache_resource
def initialize_qa_chain():
    """Initialize the QA chain and vector store (cached to avoid re-initialization)"""
    # Initialize model and embeddings
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    # Create vector store
    vector_store = InMemoryVectorStore(embeddings)
    
    # Load PDF document
    file_path = "Inventory Management.pdf"
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    vector_store.add_documents(docs)
    
    # Create retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    # Create prompt template
    system_prompt = (
        "You are a helpful assistant that answers questions about inventory management "
        "on a software called AdvEntPOS and how it integrates with DoorDash. "
        "Use the provided context to answer questions. Be helpful, accurate, and concise. "
        "If the context doesn't contain enough information to answer the question, say so."
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt + "\n\nContext: {context}"),
        ("human", "{input}")
    ])
    
    # Create document chain
    document_chain = create_stuff_documents_chain(model, prompt)
    
    # Create retrieval chain
    qa_chain = create_retrieval_chain(retriever, document_chain)
    
    return qa_chain

# Initialize QA chain
if not st.session_state.initialized:
    with st.spinner("Loading document and initializing..."):
        st.session_state.qa_chain = initialize_qa_chain()
        st.session_state.initialized = True

# Title and description
st.title("🤖 Document Q&A Chatbot")
st.markdown("Ask questions about the **Inventory Management** document. The chatbot will retrieve relevant information to answer your queries.")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about the document..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        try:
            message_placeholder = st.empty()
            full_response = ""
            
            # Try to stream the response
            try:
                for chunk in st.session_state.qa_chain.stream({"input": prompt}):
                    if "answer" in chunk:
                        chunk_answer = chunk["answer"]
                        # If we get incremental updates, append them
                        if isinstance(chunk_answer, str):
                            if len(chunk_answer) > len(full_response):
                                full_response = chunk_answer
                                message_placeholder.markdown(full_response)
                            elif chunk_answer != full_response:
                                full_response = chunk_answer
                                message_placeholder.markdown(full_response)
            except Exception as stream_error:
                # If streaming fails, fall back to invoke
                st.warning("Streaming not available, using standard response...")
            
            # If streaming didn't work or didn't produce a response, use invoke
            if not full_response:
                result = st.session_state.qa_chain.invoke({"input": prompt})
                full_response = result.get("answer", str(result))
                message_placeholder.markdown(full_response)
            
            # Add assistant response to chat history
            if full_response:
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            else:
                error_msg = "Sorry, I couldn't generate a response. Please try again."
                message_placeholder.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                
        except Exception as e:
            error_message = f"Sorry, I encountered an error: {str(e)}"
            st.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})

# Sidebar with info
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This chatbot uses:
    - **Google Gemini 2.5 Flash Lite** for language understanding
    - **RAG (Retrieval-Augmented Generation)** to answer questions
    - **Vector search** to find relevant document sections
    
    Ask questions about:
    - Inventory management features
    - AdvEntPOS software
    - DoorDash integration
    - And more!
    """)
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

