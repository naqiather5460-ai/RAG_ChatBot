from pathlib import Path
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredExcelLoader,
)

def load_document(file_path: Path):
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
        loader = UnstructuredExcelLoader(str(file_path))
    else:
        raise ValueError(f"Unsupported file type: {extension}")

    return loader.load()

def load_all_documents(documents_dir: Path):
    """
    Loads every supported file in the given directory.
    Returns a combined list of Document objects from all files.
    """
    all_documents = []
    supported_extensions = {".pdf", ".docx", ".txt", ".xlsx", ".xls"}

    for file_path in documents_dir.iterdir():
        if file_path.suffix.lower() in supported_extensions:
            print(f"[DocumentLoader] Loading {file_path.name}...")
            docs = load_document(file_path)
            all_documents.extend(docs)
        else:
            print(f"[DocumentLoader] Skipping unsupported file: {file_path.name}")

    print(f"[DocumentLoader] Loaded {len(all_documents)} document sections total.")
    return all_documents

