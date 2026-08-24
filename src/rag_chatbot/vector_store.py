"""
VectorStoreManager module.
Splits documents into chunks, embeds them, and manages the ChromaDB
vector store -- both building it fresh and reconnecting to an existing one.
"""

from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


class VectorStoreManager:
    """Manages document chunking, embedding, and ChromaDB vector storage."""

    def __init__(self, chroma_db_dir: Path, embedding_model_name: str,
                 chunk_size: int, chunk_overlap: int):
        self.chroma_db_dir = Path(chroma_db_dir)
        self.embedding_model_name = embedding_model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._embedding_model = None  # lazily loaded, cached after first use

    def get_embedding_model(self) -> HuggingFaceEmbeddings:
        """
        Loads (and caches) the free, local HuggingFace embedding model.
        Downloads automatically the first time this runs (~80MB).
        """
        if self._embedding_model is None:
            self._embedding_model = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
        return self._embedding_model

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """
        Splits loaded documents into smaller overlapping chunks.
        RecursiveCharacterTextSplitter tries to split on paragraph breaks first,
        then sentences, then words -- keeping text as coherent as possible.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        chunks = splitter.split_documents(documents)
        print(f"[VectorStoreManager] Split into {len(chunks)} chunks.")
        return chunks

    def build_vector_store(self, chunks: list[Document]) -> Chroma:
        """
        Embeds the given chunks and stores them in ChromaDB on disk.
        Returns the Chroma vector store object.
        """
        embeddings = self.get_embedding_model()
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=str(self.chroma_db_dir),
        )
        print(f"[VectorStoreManager] Stored {len(chunks)} chunks in ChromaDB at {self.chroma_db_dir}")
        return vector_store

    def load_vector_store(self) -> Chroma:
        """
        Loads the existing ChromaDB vector store from disk (doesn't rebuild it --
        just reconnects to what's already there).
        """
        embeddings = self.get_embedding_model()
        return Chroma(
            persist_directory=str(self.chroma_db_dir),
            embedding_function=embeddings,
        )