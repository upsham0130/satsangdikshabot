# Document Q&A Chatbot

Interactive web applications that allow you to ask questions about your PDF documents using RAG (Retrieval-Augmented Generation) with Google Gemini.

## Features

- 🤖 Interactive chatbot interface
- 📄 PDF document processing with text splitting
- 🔍 Semantic search for relevant context
- 💬 Chat history
- ⚡ Streaming responses
- 📊 Progress tracking for document loading
- 🔄 Batch processing with quota management

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Available Web Apps

### 1. Inventory Management Chatbot (`app.py`)

A simple chatbot for the Inventory Management document.

**Usage:**
```bash
streamlit run app.py
```

**Requirements:**
- PDF file: `Inventory Management.pdf` must be in the same directory

### 2. Satsang Diksha Chatbot (`satsang_app.py`)

A chatbot for the Satsang Diksha book with advanced features including:
- Text splitting for large documents
- Batch processing with rate limiting
- Progress tracking during document loading

**Usage:**
```bash
streamlit run satsang_app.py
```

**Requirements:**
- PDF file: `Satsang Diksha.pdf` must be in the same directory

## Command Line Script

You can also use the command-line version:

```bash
python3 chat_model.py
```

This will:
- Load and split the document
- Process embeddings in batches (respecting API quotas)
- Start an interactive Q&A session

## How it Works

1. **Document Loading**: The app loads your PDF document
2. **Text Splitting**: Large documents are split into smaller chunks for better processing
3. **Embedding Creation**: Text chunks are converted to embeddings using Google's Gemini embedding model
4. **Vector Store**: Embeddings are stored in a vector database for fast similarity search
5. **Question Answering**: When you ask a question:
   - Relevant document sections are retrieved using semantic search
   - The AI model generates an answer based on the retrieved context
   - Responses are streamed in real-time

## API Quota Management

The free tier of Google Gemini API has strict rate limits. The apps handle this by:
- Processing documents in small batches (5 documents at a time)
- Adding delays between batches (2 seconds)
- Automatically waiting 60 seconds when quota is exceeded
- Retrying failed batches

## Customization

You can modify:
- **PDF file path**: Change the `file_path` variable in the app
- **System prompt**: Modify the `system_prompt` to change the chatbot's behavior
- **Number of retrieved documents**: Adjust the `k` parameter in `search_kwargs`
- **Batch size**: Change `BATCH_SIZE` for embedding processing
- **Chunk size**: Modify `chunk_size` and `chunk_overlap` in `RecursiveCharacterTextSplitter`

## Troubleshooting

### Quota Exceeded Errors

If you see quota errors:
1. Wait a few minutes for the quota to reset
2. Reduce `BATCH_SIZE` to process fewer documents at once
3. Increase `DELAY_SECONDS` between batches
4. Consider upgrading to a paid API plan

### Document Not Loading

- Ensure the PDF file is in the same directory as the app
- Check that the filename matches exactly (case-sensitive)
- Verify the PDF is not corrupted

