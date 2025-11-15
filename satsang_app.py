import os
import time
import re
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

# Set API key - try Streamlit secrets first, then environment variable
try:
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except Exception:
    pass  # No secrets file found, will use environment variable

# Load from .env file if available (for local development)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, skip

# Check if API key is set
if "GOOGLE_API_KEY" not in os.environ or not os.environ["GOOGLE_API_KEY"]:
    st.error("⚠️ GOOGLE_API_KEY not found! Please set it in:")
    st.markdown("""
    - **Local development**: Create a `.env` file with `GOOGLE_API_KEY=your-key`
    - **Streamlit Cloud**: Add it in Settings → Secrets
    """)
    st.stop()

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
    
    # Split documents with smart separators to avoid cutting words
    # This is especially important for multilingual content (Gujarati, Sanskrit, English)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # Reduced size to minimize word breaks
        chunk_overlap=250,  # Increased overlap to preserve context across boundaries
        length_function=len,
        separators=[
            "\n\n\n",  # Triple newline (paragraph breaks)
            "\n\n",    # Double newline (paragraph breaks)
            "\n",      # Single newline (line breaks)
            ". ",      # Sentence endings with space
            "! ",      # Exclamation with space
            "? ",      # Question mark with space
            "; ",      # Semicolon with space
            ", ",      # Comma with space
            " ",       # Word boundaries (space)
            "",        # Character boundaries (last resort)
        ],
        add_start_index=True,  # track index in original document
        keep_separator=True,   # Keep separators to maintain context
    )
    
    # Split documents
    all_splits = text_splitter.split_documents(docs)
    
    # Clean up broken words at chunk boundaries
    def clean_chunk_text(text):
        """Remove incomplete words at the start/end of chunks"""
        if not text:
            return text
        
        # Remove incomplete words at the start (words that don't start with space/punctuation and are followed by space)
        # This catches cases like "hānt. He" -> ". He"
        text = re.sub(r'^[^\s\.,!?;:()[\]{}"\'-]+(?=\s)', '', text)
        
        # Remove incomplete words at the end (words that don't end with space/punctuation)
        # This catches cases like "the hānt" -> "the "
        text = re.sub(r'(?<=\s)[^\s\.,!?;:()[\]{}"\'-]+$', '', text)
        
        # Remove standalone incomplete words (like "hānt" at start/end)
        # Match words that are very short and don't end with punctuation
        text = re.sub(r'^\s*[^\s\.,!?;:()[\]{}"\'-]{1,4}\s+', '', text)  # Start
        text = re.sub(r'\s+[^\s\.,!?;:()[\]{}"\'-]{1,4}\s*$', '', text)  # End
        
        # Clean up multiple spaces
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    # Apply cleaning to all chunks
    for split in all_splits:
        split.page_content = clean_chunk_text(split.page_content)
    
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
                time.sleep(10)
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
    
    # Create retriever - retrieve more chunks for better context
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    
    # Create prompt template
    system_prompt = (
        "You are a helpful assistant that answers questions about a book called Satsang Diksha. "
        "It is a compilation of religious sermons written in 3 different languages: Gujarati leepee, "
        "Sanskrit leepee, and English. Use English primarily for answering questions, but refer to "
        "the other languages when asked. There is a glossary at the end of the book to help you "
        "understand and answer questions about words not in English. "
        "Be helpful, accurate, and concise. If the context doesn't contain enough information to "
        "answer the question, say so. If asked a morally or ethically questionable question, "
        "say so and don't answer it. "
        "IMPORTANT: If you encounter incomplete words or text fragments in the context (like 'hānt' "
        "or other broken words), ignore them and focus on complete, meaningful sentences. "
        "Do not include partial words or fragments in your answer."
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
            message_placeholder.markdown("Thinking...")
            
            # Use invoke for more reliable response display
            # Streaming with retrieval chains can be inconsistent
            with st.spinner("Generating answer..."):
                result = st.session_state.qa_chain.invoke({"input": prompt})
                full_response = result.get("answer", str(result))
            
            # Display the full response
            if full_response:
                message_placeholder.markdown(full_response)
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

