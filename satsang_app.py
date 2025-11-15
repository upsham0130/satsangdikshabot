import os
import time
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Set page config
st.set_page_config(
    page_title="Satsang Diksha Q&A Chatbot",
    page_icon="📖",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "initialized" not in st.session_state:
    st.session_state.initialized = False
if "loading_progress" not in st.session_state:
    st.session_state.loading_progress = None

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
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    # Create vector store
    vector_store = InMemoryVectorStore(embeddings)
    
    # Load PDF document
    file_path = "Satsang Diksha.pdf"
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # chunk size (characters)
        chunk_overlap=200,  # chunk overlap (characters)
        add_start_index=True,  # track index in original document
    )
    all_splits = text_splitter.split_documents(docs)
    
    # Add documents in batches to respect rate limits
    BATCH_SIZE = 5  # Process 5 documents at a time
    DELAY_SECONDS = 2  # Wait 2 seconds between batches
    
    total_batches = (len(all_splits) + BATCH_SIZE - 1) // BATCH_SIZE
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(0, len(all_splits), BATCH_SIZE):
        batch = all_splits[i:i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        
        status_text.text(f"Processing batch {batch_num}/{total_batches} ({len(batch)} documents)...")
        progress_bar.progress(batch_num / total_batches)
        
        try:
            vector_store.add_documents(batch)
        except Exception as e:
            error_str = str(e)
            # Check if it's a quota error
            if "429" in error_str or "quota" in error_str.lower():
                status_text.text(f"Quota exceeded in batch {batch_num}. Waiting 60 seconds...")
                time.sleep(60)
            else:
                status_text.text(f"Error in batch {batch_num}. Waiting 4 seconds before retry...")
                time.sleep(4)
            
            # Retry the batch
            try:
                vector_store.add_documents(batch)
            except Exception as retry_error:
                retry_error_str = str(retry_error)
                if "429" in retry_error_str or "quota" in retry_error_str.lower():
                    status_text.text(f"Quota still exceeded for batch {batch_num}. Skipping...")
                else:
                    status_text.text(f"Failed to process batch {batch_num} after retry. Skipping...")
                continue
        
        # Wait between batches (except for the last one)
        if i + BATCH_SIZE < len(all_splits):
            time.sleep(DELAY_SECONDS)
    
    progress_bar.progress(1.0)
    status_text.text("✓ All documents loaded!")
    time.sleep(1)  # Show completion message briefly
    progress_bar.empty()
    status_text.empty()
    
    # Create retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    # Create prompt template
    system_prompt = (
        "You are a helpful assistant that answers questions about a book called Satsang Diksha. "
        "It is a compilation of religious sermons written in 3 different languages: Gujarati leepee, "
        "Sanskrit leepee, and English. Use English primarily for answering questions, but refer to "
        "the other languages when asked. There is a glossary at the end of the book to help you "
        "understand and answer questions about words not in English. "
        "Be helpful, accurate, and concise. If the context doesn't contain enough information to "
        "answer the question, say so. If asked a morally or ethically questionable question, "
        "say so and don't answer it."
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
st.title("📖 Satsang Diksha Q&A Chatbot")
st.markdown("""
Ask questions about **Satsang Diksha**, a compilation of religious sermons written in Gujarati, Sanskrit, and English. 
The chatbot will retrieve relevant information from the book to answer your queries.
""")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about Satsang Diksha..."):
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
                with st.spinner("Thinking..."):
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
    - **Google Gemini 2.5 Flash** for language understanding
    - **RAG (Retrieval-Augmented Generation)** to answer questions
    - **Vector search** to find relevant document sections
    
    **Satsang Diksha** contains:
    - Religious sermons in Gujarati, Sanskrit, and English
    - A glossary for understanding non-English terms
    - Spiritual teachings and guidance
    """)
    
    st.header("⚙️ Settings")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    if st.button("🔄 Reload Document"):
        st.cache_resource.clear()
        st.session_state.initialized = False
        st.session_state.qa_chain = None
        st.rerun()

