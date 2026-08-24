"""
DocumentLoader module.
Handles ingestion of multiple document formats (PDF, DOCX, TXT, Excel)
into a unified LangChain Document representation.
"""

from pathlib import Path
import pandas as pd
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)


class DocumentLoader:
    """Loads and ingests documents of multiple formats from a directory."""

    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".xlsx", ".xls"}

    def __init__(self, documents_dir: Path):
        self.documents_dir = Path(documents_dir)

    def _load_excel_document(self, file_path: Path) -> list[Document]:
        """
        Loads an Excel file using pandas and converts it into a LangChain Document.
        Avoids the heavy 'unstructured' dependency required by LangChain's built-in
        Excel loader -- pandas is simpler, faster, and already a project dependency.
        """
        df = pd.read_excel(file_path)
        text_content = df.to_string(index=False)
        return [Document(page_content=text_content, metadata={"source": str(file_path)})]

    def load_document(self, file_path: Path) -> list[Document]:
        """
        Loads a single document, choosing the correct loader based on file extension.
        Returns a list of LangChain Document objects.
        """
        extension = file_path.suffix.lower()

        if extension == ".pdf":
            loader = PyPDFLoader(str(file_path))
        elif extension == ".docx":
            loader = Docx2txtLoader(str(file_path))
        elif extension == ".txt":
            loader = TextLoader(str(file_path), encoding="utf-8")
        elif extension in (".xlsx", ".xls"):
            return self._load_excel_document(file_path)
        else:
            raise ValueError(f"Unsupported file type: {extension}")

        return loader.load()

    def load_all_documents(self) -> list[Document]:
        """
        Loads every supported file in self.documents_dir.
        Returns a combined list of Document objects from all files.
        """
        all_documents = []

        for file_path in self.documents_dir.iterdir():
            if file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                print(f"[DocumentLoader] Loading {file_path.name}...")
                docs = self.load_document(file_path)
                all_documents.extend(docs)
            else:
                print(f"[DocumentLoader] Skipping unsupported file: {file_path.name}")

        print(f"[DocumentLoader] Loaded {len(all_documents)} document sections total.")
        return all_documents