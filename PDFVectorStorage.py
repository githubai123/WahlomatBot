import os
import hashlib
import chromadb
from datetime import datetime, timezone
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings.ollama import OllamaEmbeddings
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings

class OllamaEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.embedding_func = OllamaEmbeddings(model=model_name)

    def __call__(self, input: Documents) -> Embeddings:
        return self.embedding_func.embed_documents(input)

class PDFVectorStorage:
    def __init__(self, storage_path, model_name="nomic-embed-text"):
        self.client = chromadb.PersistentClient(path=storage_path)
        self.embedding_func = OllamaEmbeddingFunction(model_name=model_name)
        self.collection = self.client.get_or_create_collection(
            name="pdf_documents",
            embedding_function=self.embedding_func,
            metadata={"hnsw:space": "cosine"}
        )

    def _compute_file_hash(self, file_path):
        """Compute the SHA256 hash of the file."""
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()

    def add_pdf(self, pdf_path):
        file_hash = self._compute_file_hash(pdf_path)
        existing_docs = self.collection.get(where={"file_hash": file_hash})
        
        if existing_docs['ids']:
            print(f"Document {pdf_path} already exists in the database.")
            return

        pdf_loader = PyPDFLoader(pdf_path)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        documents = pdf_loader.load_and_split(text_splitter=text_splitter)
        
        for i, doc in enumerate(tqdm(documents, desc="Ingesting PDF")):
            doc_id = f"{file_hash}_chunk_{i}"
            self.collection.add(
                documents=[doc.page_content],
                ids=[doc_id],
                metadatas=[{
                    "file_name": os.path.basename(pdf_path),
                    "file_hash": file_hash,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }]
            )

    def retrieve_similar(self, query, top_k=4):
        results = self.collection.query(query_texts=[query], n_results=top_k)
        return results

# Usage example
if __name__ == "__main__":
    storage = PDFVectorStorage(storage_path="./chroma_data", model_name="nomic-embed-text")
    storage.add_pdf("./data/CDU.pdf")
    results = storage.retrieve_similar("Sinne")
    for result in results['documents'][0]:
        print(result)
