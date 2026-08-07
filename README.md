# RAG Chatbot — LangChain, ChromaDB & Groq

An AI chatbot that answers questions about your own documents using Retrieval-Augmented Generation (RAG). Supports PDF, DOCX, TXT, and Excel files.

## How It Works

1. **Document Ingestion** — loads PDF/DOCX/TXT/Excel files using format-specific LangChain loaders
2. **Chunking** — splits documents into overlapping chunks for better semantic precision
3. **Embedding** — converts each chunk into a vector using a free, local HuggingFace embedding model
4. **Vector Storage** — stores embeddings in ChromaDB for fast similarity search
5. **Retrieval** — given a question, finds the most semantically relevant chunks
6. **Generation** — sends the question + retrieved chunks to a Groq-hosted LLM to generate an answer

## Tech Stack

- **LangChain** — orchestrates the pipeline
- **ChromaDB** — vector database for storing and searching embeddings
- **Groq** — fast, free LLM inference (Llama 3)
- **HuggingFace `sentence-transformers`** — free, local embedding model

## Project Structure
Chatbot/

├── data/documents/ # source documents (PDF, DOCX, TXT, XLSX)

├── chroma_db/ # ChromaDB's persistent vector storage

├── src/rag_chatbot/

│├── config.py # paths, API key loading, RAG settings

│├── document_loader.py # multi-format document ingestion

│├── vector_store.py # chunking, embedding, vector storage

│└── retriever.py # semantic search / retrieval pipeline

├── main.py # entry point

├── .env # GROQ_API_KEY (gitignored)

└── requirements.txt

## Setup

1. Clone the repo
2. Install dependencies: `python -m pip install -r requirements.txt`
3. Create a `.env` file with: `GROQ_API_KEY=your_key_here`
4. Add documents to `data/documents/`
5. Run the chatbot: `python main.py`

## Status

🚧 Under active development. Currently implemented: document ingestion, chunking, embedding, vector storage, and retrieval. Conversational QA with the LLM is in progress.
