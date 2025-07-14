import faiss
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.faiss import FaissVectorStore

from arm_ex.documents import create_llama_index_documents


def build_index(database_path: str, dimension: int = 768) -> VectorStoreIndex:
    """Build a FAISS-backed index from the given database file."""
    faiss_index = faiss.IndexFlatIP(dimension)
    documents = create_llama_index_documents(database_path)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    index.storage_context.persist()
    return index


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build FAISS index from JSON DB")
    parser.add_argument("database", help="Path to database JSON file")
    parser.add_argument("--dim", type=int, default=768, help="Embedding dimension")
    args = parser.parse_args()

    build_index(args.database, args.dim)
    print("Index created and persisted")
