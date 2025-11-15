import os
import time
from langchain_google_genai import ChatGoogleGenerativeAI

os.environ["GOOGLE_API_KEY"] = "AIzaSyA32QT_Nb6f2-6NxG31ZMx6AxBwbrTOOIw"

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

from langchain_core.vectorstores import InMemoryVectorStore

vector_store = InMemoryVectorStore(embeddings)


from langchain_community.document_loaders import PyPDFLoader

file_path = "Satsang Diksha.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # chunk size (characters)
    chunk_overlap=200,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)
all_splits = text_splitter.split_documents(docs)

print(f"Split blog post into {len(all_splits)} sub-documents.")

# Add documents in batches to respect rate limits
# Free tier has very strict limits, so we process in small batches with delays

BATCH_SIZE = 5  # Process 5 documents at a time
DELAY_SECONDS = 2  # Wait 2 seconds between batches (adjust based on your quota)

print(f"Adding documents in batches of {BATCH_SIZE} with {DELAY_SECONDS}s delays...")
for i in range(0, len(all_splits), BATCH_SIZE):
    batch = all_splits[i:i + BATCH_SIZE]
    batch_num = (i // BATCH_SIZE) + 1
    total_batches = (len(all_splits) + BATCH_SIZE - 1) // BATCH_SIZE
    
    try:
        print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} documents)...")
        vector_store.add_documents(batch)
        print(f"✓ Batch {batch_num} completed")
        
        # Wait between batches (except for the last one)
        if i + BATCH_SIZE < len(all_splits):
            time.sleep(DELAY_SECONDS)
    except Exception as e:
        error_str = str(e)
        # Check if it's a quota error
        if "429" in error_str or "quota" in error_str.lower():
            wait_time = 60  # Wait 60 seconds for quota to reset
            print(f"✗ Quota exceeded in batch {batch_num}. Waiting {wait_time} seconds...")
            time.sleep(wait_time)
        else:
            wait_time = DELAY_SECONDS * 2
            print(f"✗ Error in batch {batch_num}: {e}")
            print(f"Waiting {wait_time} seconds before retry...")
            time.sleep(wait_time)
        
        # Retry the batch
        try:
            vector_store.add_documents(batch)
            print(f"✓ Batch {batch_num} completed on retry")
        except Exception as retry_error:
            retry_error_str = str(retry_error)
            if "429" in retry_error_str or "quota" in retry_error_str.lower():
                print(f"✗ Quota still exceeded for batch {batch_num}. Skipping...")
            else:
                print(f"✗ Failed to process batch {batch_num} after retry: {retry_error}")
            print("Skipping this batch and continuing...")

print("✓ All documents added to vector store!")

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

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

print("\n" + "="*60)
print("Chatbot ready! Ask questions about Satsang Diksha.")
print("Type 'quit' or 'exit' to stop.")
print("="*60 + "\n")

while True:
    query = input("Enter a question: ").strip()
    
    if query.lower() in ['quit', 'exit', 'q']:
        print("Goodbye!")
        break
    
    if not query:
        continue
    
    try:
        print("\nThinking...\n")
        result = qa_chain.invoke({"input": query})
        answer = result.get("answer", str(result))
        print(f"Answer: {answer}\n")
        print("-" * 60 + "\n")
    except Exception as e:
        print(f"Error: {e}\n")