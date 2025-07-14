import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))    

import numpy
import faiss


d = 768 # for sentence transformers, jina embeddings, bge 
# d = 1024 # for Qwen3-Embedding-0.6B
# d = 1024
# d = 1536 ## for openai text-embedding-small
# d = 3072 ## for openai text-embedding-large
# d = 1024 ## voyage embeddings
faiss_index = faiss.IndexFlatIP(d)





from llama_index.core import (
    SimpleDirectoryReader,
    load_index_from_storage,
    VectorStoreIndex,
    StorageContext,
)
from llama_index.vector_stores.faiss import FaissVectorStore
from IPython.display import Markdown, display
from typing import List, Dict, Set
from llama_index.core import Document
from langchain.embeddings import HuggingFaceEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding
from IPython.display import Markdown, display
from llama_index.core import Settings
import google.generativeai as genai
import re
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from llama_index.embeddings.gemini import GeminiEmbedding

from IPython.display import display
import json
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict, Any
import numpy as np

from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from google.genai.types import EmbedContentConfig
from llama_index.postprocessor.cohere_rerank import CohereRerank

from llama_index.llms.ollama import Ollama
from llama_index.core.schema import QueryBundle, NodeWithScore

from arm_ex.documents import create_llama_index_documents, deepjoin_column_transform, create_documents_from_table_repo, transform_retrieved_nodes


# lc_embed_model = HuggingFaceEmbeddings(
#     model_name="BAAI/bge-base-en-v1.5"
# )

lc_embed_model = HuggingFaceEmbeddings(
    model_name="Qwen/Qwen3-Embedding-0.6B"
)


# lc_embed_model = HuggingFaceEmbeddings(
#     model_name="Lajavaness/bilingual-embedding-large",
# )
# lc_embed_model = HuggingFaceEmbeddings(
#     model_name="jinaai/jina-embeddings-v2-base-en",
# )
embed_model = LangchainEmbedding(lc_embed_model)

GOOGLE_API_KEY="AIzaSyASJQghVuVjgnLkRduL0YB6eh6hb2ZUhuA"

genai.configure(api_key=GOOGLE_API_KEY)

Settings.embed_model = embed_model
Settings.chunk_size = 5500



vector_store = FaissVectorStore(faiss_index=faiss_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

# save index to disk
index.storage_context.persist()

# vector_store = FaissVectorStore.from_persist_dir("./main_stuff/storage_desc")
# storage_context = StorageContext.from_defaults(
#     vector_store=vector_store, persist_dir="./main_stuff/storage_desc"
# )
vector_store = FaissVectorStore.from_persist_dir("./MultiTableQA/himanshu/storage_desc")
storage_context = StorageContext.from_defaults(
    vector_store=vector_store, persist_dir="./MultiTableQA/himanshu/storage_desc"
)
index = load_index_from_storage(storage_context=storage_context)

from llama_index.llms.gemini import Gemini

model = llm = Gemini(
    model="models/gemini-2.0-flash",
    api_key=GOOGLE_API_KEY
)
Settings.llm = model


def embed_column_names(column_names: List[str]) -> np.ndarray:
    return np.array(lc_embed_model.embed_documents(column_names))

def get_column_samples(table, column_name, max_rows=5):
    values = []
    for row in table.get("rows", []):
        val = row.get(column_name)
        if val is not None and val != "":
            values.append(str(val))
        if len(values) >= max_rows:
            break
    return values


import json

# Specify the path to your JSON file
file_path = "./MultiTableQA/himanshu/combined_database_with_desc.json"

# Open and load the JSON file
with open(file_path, "r") as file:
    table_repository = json.load(file)

# Print the loaded data
table_repository


import re

def find_joinable_tables_with_faiss(retrieved_nodes_tr, faiss_store, embed_model, threshold=0.7):
    """
    Find joinable tables using a FAISS vector store with precomputed column embeddings.
    Uses pre-transformed node data in the retrieved_nodes_tr format.

    Args:
        retrieved_nodes_tr: Dictionary mapping table IDs to transformed node data
        faiss_store: Dictionary containing the FAISS index and metadata
        embed_model: Embedding model to compute embeddings for columns
        threshold: Distance threshold for considering columns joinable (default 0.9)
               Note: For L2 distance, SMALLER means more similar!

    Returns:
        Dictionary mapping retrieved table IDs to their joinable tables
    """
    faiss_index = faiss_store["faiss_index"]
    doc_metadata = faiss_store["doc_metadata"]

    # Extract retrieved table IDs for quick lookups
    retrieved_table_ids = set(retrieved_nodes_tr.keys())

    joinable_results = {}
    match_count = 0  # For debugging
    
    # Process each table in the transformed data
    for table_id, table_data in retrieved_nodes_tr.items():
        table_name = table_data["table_name"]
        column_descriptions = table_data["column_descriptions"]
        
        joinable_results[table_id] = []
        
        # Process each column description
        for column_description in column_descriptions:
            # Parse column name and other info from description
            # Format: "table_name (table_id). column_name contains X values (max, min, avg): val1, val2, val3"
            parts = column_description.split("contains")
            if len(parts) < 2:
                continue
                
            column_name_part = parts[0].strip()
            column_name = column_name_part.split(".")[-1].strip()
            
            # Skip empty columns
            if re.search(r"\b0\s+values\b", parts[1]):
                continue
            # Extract the column description text
            # The column description already has a good text representation
            column_text = column_description
            
            # Compute embedding for the column
            # query_embedding = embed_model.get_query_embedding(column_text)

            query_embedding = embed_model.embed_query(column_text)
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            distances, indices = faiss_index.search(np.array([query_embedding], dtype=np.float32), k=10)
            # Process matches - IMPORTANT: Lower L2 distance = more similar!
            for dist, idx in zip(distances[0], indices[0]):
                # # Debug distance values
                # print(f"Column {column_name}: Found match with distance {dist} (threshold: {threshold})")
                
                # For L2 distance, smaller means more similar
                if dist < threshold:
                    continue
                    
                if dist >=threshold:
                    candidate_meta = doc_metadata[idx]
                    candidate_table_id = candidate_meta["table_id"]

                    # Skip if the candidate is the same table or already retrieved
                    if candidate_table_id == table_id or candidate_table_id in retrieved_table_ids:
                        continue

                    # Add to joinable results
                    similarity_score = dist #vert distance to similarity (0-1)
                    joinable_results[table_id].append({
                        "with_table": candidate_table_id,
                        "with_table_name": candidate_meta["table_name"],
                        "query_column": column_name,
                        "candidate_column": candidate_meta["column_name"],
                        "similarity": round(similarity_score, 4),   
                    })
                    match_count += 1
        
        # Sort results by similarity (highest first)
        if table_id in joinable_results:
            joinable_results[table_id] = sorted(
                joinable_results[table_id], 
                key=lambda x: x["similarity"], 
                reverse=True
            )
    
    
    # Remove empty results
    joinable_results = {k: v for k, v in joinable_results.items() if v}
    
    return joinable_results

# faiss_index_file = "vs_col_bge_.bin"
# metadata_file = "doc_metadata_bge.json"
faiss_index_file = "./MultiTableQA/himanshu/vs_col_bge_.bin"
metadata_file = "./MultiTableQA/himanshu/doc_metadata_bge.json"

# Load FAISS index
faiss_index = faiss.read_index(faiss_index_file)
print("✅ FAISS index loaded.")

# Load metadata
with open(metadata_file, "r") as f:
    doc_metadata = json.load(f)
print("✅ Document metadata loaded.")
print("Number of vectors in FAISS index:", faiss_index.ntotal)
print("Vector dimension:", faiss_index.d)

faiss_store = {
    "faiss_index": faiss_index,
    "doc_metadata": doc_metadata}



def rank_tables_by_query_similarity_with_cross_encoder(query, documents, cross_encoder_model, top_k=None):
    """
    Compute similarity between a query and each document (table) using a CrossEncoder model and rank them.

    Args:
        query: The user's query string.
        documents: List of LlamaIndex Document objects (one per table).
        cross_encoder_model: An instance of a CrossEncoder model.
        top_k: Number of top documents to return (if None, return all).

    Returns:
        A list of document table_ids ranked by relevance.
    """
    # Prepare the query bundle
    query_bundle = QueryBundle(query_str=query)
    
    # Extract text from each document
    doc_texts = [doc.text_resource.text for doc in documents]
    
    # Create sentence pairs for the cross-encoder
    sentence_pairs = [[query, doc_text] for doc_text in doc_texts]
    
    # Get scores from the cross-encoder model
    scores = cross_encoder_model.predict(sentence_pairs, convert_to_tensor=True).tolist()
    
    # Create NodeWithScore objects with the computed scores
    nodes_with_scores = [
        NodeWithScore(node=doc, score=score)
        for doc, score in zip(documents, scores)
    ]
    
    # Sort the nodes by score in descending order
    nodes_with_scores.sort(key=lambda x: x.score, reverse=True)
    
    # Limit to top_k if specified
    if top_k is not None:
        nodes_with_scores = nodes_with_scores[:top_k]
    
    # Extract the top document table_ids
    top_docs = [node_with_score.node.metadata["table_id"] for node_with_score in nodes_with_scores]
    
    return top_docs

from sentence_transformers import CrossEncoder

rerank = CrossEncoder(
        "jinaai/jina-reranker-v2-base-multilingual",
        automodel_args={"torch_dtype": "auto"},
        trust_remote_code=True,
    )

def filter_documents_by_table_ids(
    documents: List[Document], allowed_ids: List[str]
) -> List[Document]:
    return [doc for doc in documents if doc.metadata.get("table_id") in allowed_ids]

def get_full_data(table_repository, retrieved_docs):
    relevant_table_ids = set()
    for i in retrieved_docs:
        if isinstance(i, NodeWithScore):
            relevant_table_ids.add(i.node.metadata["table_id"])
        elif isinstance(i, Document):
            relevant_table_ids.add(i.metadata["table_id"])
        

    documents = []
     # Process the nested table structure
    for table_name, variants in table_repository["tables"].items():
        # Process each variant of this table
        for variant_idx, variant in enumerate(variants):
            table_id = variant.get("id")
            
            # Skip variants not in relevant_table_ids
            if table_id not in relevant_table_ids:
                continue
                
            columns = variant["columns"]
            # Handle both "data" and "content" field names
            content = variant.get("data", variant.get("content", []))
            description = variant.get("table_description")
            # Format table schema as text
            schema_text = f"Table ID: {table_id}\n\n"
            schema_text += f"Table Name: {table_name}\n\n"
            schema_text += f"Columns: {', '.join(columns)}\n\n"


            # Add data sample
            if content and len(content) > 0:
                schema_text += "First 5 rows of data:\n"
                header_row = " | ".join(columns)
                separator = "-" * len(header_row)
                schema_text += f"{header_row}\n{separator}\n"
                schema_text+= f"Table Description: {description}\n"
                
                for row in content[:5]:
                    # Handle rows that might be shorter than columns
                    row_values = []
                    for i in range(len(columns)):
                        if i < len(row):
                            row_values.append(str(row[i]) if row[i] is not None else "NULL")
                        else:
                            row_values.append("NULL")
                    
                    row_str = " | ".join(row_values)
                    schema_text += f"{row_str}\n"
            else:
                schema_text += "Data: No rows available for this table.\n"

            # Create a LlamaIndex Document
            doc = Document(
                text=schema_text,
                metadata={
                    "table_name": table_name,
                    "table_id": table_id,
                    "columns": columns,
                    "row_count": len(content),
                    "rows" : content
                }
            )

            documents.append(doc)

    return documents

import pandas as pd

def document_to_dataframe(document) -> pd.DataFrame:
    """
    Convert a document object to a pandas DataFrame.
    
    Args:
        document: Document object with metadata containing table information
    
    Returns:
        pd.DataFrame: DataFrame created from the document's table data
    """
    # Extract metadata
    metadata = document.metadata
    
    # Get column names and data rows
    columns = metadata['columns']
    rows = metadata['rows']
    
    # Create DataFrame
    df = pd.DataFrame(rows, columns=columns)
    
    # Add metadata as DataFrame attributes for reference
    df.attrs['table_name'] = metadata.get('table_name', '')
    df.attrs['table_id'] = metadata.get('table_id', '')
    df.attrs['row_count'] = metadata.get('row_count', len(rows))
    
    return df


def create_table_id_mapping(documents):
    """
    Create a mapping dictionary from table_id to Document objects.
    
    Args:
        documents: List of Document objects (as shown in your data structure)
    
    Returns:
        dict: Mapping of table_id -> Document object
    """
    table_id_mapping = {}
    
    for doc in documents:
        # Extract table_id from document metadata
        table_id = doc.metadata.get('table_id')
        if table_id:
            table_id_mapping[table_id] = doc
        else:
            print(f"Warning: Document {doc.id_} has no table_id in metadata")
    
    return table_id_mapping

def get_retrieved_tables(data, question_id):
    """
    Get retrieved table names for a specific question ID.
    
    Args:
        data (dict): The parsed JSON data
        question_id (int): The question ID to look up
        
    Returns:
        list: List of retrieved table names
        
    Raises:
        ValueError: If question ID is not found
    """
    # Find the question with the matching question_id
    question = next((q for q in data['questions'] if q['question_id'] == question_id), None)
    
    if question is None:
        raise ValueError(f"Question with ID {question_id} not found")
    
    return question['retrieved_table_ids']

def normalize_column_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_")

def extract_table_id(node_text: str) -> str:
    """Extract table ID from node text"""
    # First try to find an explicit table ID in the text
    id_match = re.search(r"Table ID: (tbl_[a-zA-Z0-9_-]+)", node_text)
    if id_match:
        return id_match.group(1)
    
    # If no explicit ID, try to extract from metadata
    metadata_match = re.search(r'"table_id":\s*"([^"]+)"', node_text)
    if metadata_match:
        return metadata_match.group(1)
    
    return ""

def calculate_metrics_by_id(expected_table_ids: List[str], retrieved_table_ids: List[str], 
                        id_to_name_map: Dict[str, str] = None):
    """
    Calculate recall, precision, F1 score, full recall, and full precision based on table IDs
    
    Args:
        expected_table_ids: List of expected table IDs
        retrieved_table_ids: List of retrieved table IDs
        id_to_name_map: Optional mapping from table ID to table name for display purposes
    """
    # Create sets of expected and retrieved IDs
    expected_set = set(expected_table_ids)
    retrieved_set = set(retrieved_table_ids)
    
    # Find common table IDs
    correct_table_ids = expected_set.intersection(retrieved_set)
    
    # Calculate metrics
    recall = len(correct_table_ids) / len(expected_set) if expected_set else 0
    precision = len(correct_table_ids) / len(retrieved_set) if retrieved_set else 0
    
    # F1 score
    f1 = 0
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    
    # Full recall (1 if all expected tables were retrieved, 0 otherwise)
    full_recall = 1 if expected_set.issubset(retrieved_set) else 0
    # Full precision (1 if all retrieved tables are expected, 0 otherwise)
    full_precision = 1 if retrieved_set.issubset(expected_set) else 0
    
    # Prepare result with IDs
    result = {
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "full_recall": full_recall,
        "full_precision": full_precision,
        "expected_table_ids": list(expected_set),
        "retrieved_table_ids": list(retrieved_set),
        "correct_table_ids": list(correct_table_ids),
        "missing_table_ids": list(expected_set - retrieved_set),
        "extra_table_ids": list(retrieved_set - expected_set)
    }
    
    # # If id_to_name_map is provided, add readable table names
    # if id_to_name_map:
    #     result["expected_tables"] = [id_to_name_map.get(id_, id_) for id_ in expected_set]
    #     result["retrieved_tables"] = [id_to_name_map.get(id_, id_) for id_ in retrieved_set]
    #     result["correct_tables"] = [id_to_name_map.get(id_, id_) for id_ in correct_table_ids]
    #     result["missing_tables"] = [id_to_name_map.get(id_, id_) for id_ in expected_set - retrieved_set]
    #     result["extra_tables"] = [id_to_name_map.get(id_, id_) for id_ in retrieved_set - expected_set]
    
    return result

import os

os.environ["OPENAI_API_KEY"] = "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"


from typing import Any, List, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

doc_tokenizer = AutoTokenizer.from_pretrained(
    "naver/efficient-splade-VI-BT-large-doc"
)
doc_model = AutoModelForMaskedLM.from_pretrained(
    "naver/efficient-splade-VI-BT-large-doc"
)

query_tokenizer = AutoTokenizer.from_pretrained(
    "naver/efficient-splade-VI-BT-large-query"
)
query_model = AutoModelForMaskedLM.from_pretrained(
    "naver/efficient-splade-VI-BT-large-query"
)
# Move models to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
doc_model = doc_model.to(device)
query_model = query_model.to(device)




def sparse_doc_vectors(
    texts: List[str],
) -> Tuple[List[List[int]], List[List[float]]]:
    """
    Computes vectors from logits and attention mask using ReLU, log, and max operations.
    """
    tokens = doc_tokenizer(
        texts, truncation=True, padding=True, return_tensors="pt"
    )
    if torch.cuda.is_available():
        tokens = tokens.to("cuda")

    with torch.no_grad():  # Add this for efficiency and memory savings
        output = doc_model(**tokens)
        logits, attention_mask = output.logits, tokens.attention_mask
        relu_log = torch.log(1 + torch.relu(logits))
        weighted_log = relu_log * attention_mask.unsqueeze(-1)
        tvecs, _ = torch.max(weighted_log, dim=1)

        # Move back to CPU for processing
        tvecs = tvecs.cpu()

    # extract the vectors that are non-zero and their indices
    indices = []
    vecs = []
    for batch in tvecs:
        indices.append(batch.nonzero(as_tuple=True)[0].tolist())
        vecs.append(batch[indices[-1]].tolist())
    

    return indices, vecs


def sparse_query_vectors(
    texts: List[str],
) -> Tuple[List[List[int]], List[List[float]]]:
    """
    Computes vectors from logits and attention mask using ReLU, log, and max operations.
    """
    # TODO: compute sparse vectors in batches if max length is exceeded
    tokens = query_tokenizer(
        texts, truncation=True, padding=True, return_tensors="pt"
    )
    if torch.cuda.is_available():
        tokens = tokens.to("cuda")

    with torch.no_grad():  # Add this for efficiency and memory savings
        output = query_model(**tokens)
        logits, attention_mask = output.logits, tokens.attention_mask
        relu_log = torch.log(1 + torch.relu(logits))
        weighted_log = relu_log * attention_mask.unsqueeze(-1)
        tvecs, _ = torch.max(weighted_log, dim=1)

        # Move back to CPU for processing
        tvecs = tvecs.cpu()
    # extract the vectors that are non-zero and their indices
    indices = []
    vecs = []
    for batch in tvecs:
        indices.append(batch.nonzero(as_tuple=True)[0].tolist())
        vecs.append(batch[indices[-1]].tolist())

    return indices, vecs

from llama_index.core.vector_stores import VectorStoreQueryResult


def relative_score_fusion(
    dense_result: VectorStoreQueryResult,
    sparse_result: VectorStoreQueryResult,
    alpha: float = 0.5,  # passed in from the query engine
    top_k: int = 2,  # passed in from the query engine i.e. similarity_top_k
) -> VectorStoreQueryResult:
    """
    Fuse dense and sparse results using relative score fusion.
    """

    # print("dense_result: ", dense_result)
    # print("sparse_result: ", sparse_result)
    # sanity check
    assert dense_result.nodes is not None
    assert dense_result.similarities is not None
    assert sparse_result.nodes is not None
    assert sparse_result.similarities is not None

    # deconstruct results
    sparse_result_tuples = list(
        zip(sparse_result.similarities, sparse_result.nodes)
    )
    sparse_result_tuples.sort(key=lambda x: x[0], reverse=True)

    dense_result_tuples = list(
        zip(dense_result.similarities, dense_result.nodes)
    )
    dense_result_tuples.sort(key=lambda x: x[0], reverse=True)

    # track nodes in both results
    all_nodes_dict = {x.node_id: x for x in dense_result.nodes}
    for node in sparse_result.nodes:
        if node.node_id not in all_nodes_dict:
            all_nodes_dict[node.node_id] = node

    # normalize sparse similarities from 0 to 1
    sparse_similarities = [x[0] for x in sparse_result_tuples]
    max_sparse_sim = max(sparse_similarities)
    min_sparse_sim = min(sparse_similarities)
    sparse_similarities = [
        (x - min_sparse_sim) / (max_sparse_sim - min_sparse_sim)
        for x in sparse_similarities
    ]
    sparse_per_node = {
        sparse_result_tuples[i][1].node_id: x
        for i, x in enumerate(sparse_similarities)
    }

    # normalize dense similarities from 0 to 1
    dense_similarities = [x[0] for x in dense_result_tuples]
    max_dense_sim = max(dense_similarities)
    min_dense_sim = min(dense_similarities)
    dense_similarities = [
        (x - min_dense_sim) / (max_dense_sim - min_dense_sim)
        for x in dense_similarities
    ]
    dense_per_node = {
        dense_result_tuples[i][1].node_id: x
        for i, x in enumerate(dense_similarities)
    }

    # fuse the scores
    fused_similarities = []
    for node_id in all_nodes_dict:
        sparse_sim = sparse_per_node.get(node_id, 0)
        dense_sim = dense_per_node.get(node_id, 0)
        fused_sim = alpha * (sparse_sim + dense_sim)
        fused_similarities.append((fused_sim, all_nodes_dict[node_id]))

    fused_similarities.sort(key=lambda x: x[0], reverse=True)
    fused_similarities = fused_similarities[:top_k]

    # create final response object
    return VectorStoreQueryResult(
        nodes=[x[1] for x in fused_similarities],
        similarities=[x[0] for x in fused_similarities],
        ids=[x[1].node_id for x in fused_similarities],
    )

GOOGLE_API_KEY="KEY"

genai.configure(api_key=GOOGLE_API_KEY)

from llama_index.llms.gemini import Gemini

model = llm = Gemini(
    model="models/gemini-2.0-flash",
    api_key=GOOGLE_API_KEY
)
Settings.llm = model

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core import Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, AsyncQdrantClient

# creates a persistant index to disk
client = QdrantClient(path="./qdrant3")

# create our vector store with hybrid indexing enabled
# batch_size controls how many nodes are encoded with sparse vectors at once
vector_store = QdrantVectorStore(
    "splade",
    client=client,
    enable_hybrid=True,
    sparse_doc_fn=sparse_doc_vectors,
    sparse_query_fn=sparse_query_vectors,
    hybrid_fusion_fn=relative_score_fusion,
    batch_size=1
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)
Settings.chunk_size = 5500

index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
)

query_engine = index.as_query_engine(
    similarity_top_k=3, sparse_top_k=3, vector_store_query_mode="sparse", response_mode="no_text",
)

collection_info = client.get_collection("splade")
print(f"Points: {collection_info.points_count}")
print(f"Vectors count: {collection_info.vectors_count}")
print(f"Indexed vectors: {collection_info.indexed_vectors_count}")


from IPython.display import display, Markdown
question = "Which department currently headed by a temporary acting manager has the largest number of employees, and how many employees does it have?"

response = query_engine.query(
    question
)
response.source_nodes

import time
from tqdm import tqdm
top_k = 5
delta = 0

retriever = index.as_retriever(similarity_top_k=top_k)

from llama_index.retrievers.bm25 import BM25Retriever
import Stemmer

bm25_retriever = BM25Retriever.from_defaults(
    docstore=index.docstore,
    similarity_top_k=top_k,
    # Optional: We can pass in the stemmer and set the language for stopwords
    # This is important for removing stopwords and stemming the query + text
    # The default is english for both
    stemmer=Stemmer.Stemmer("english"),
    language="english",
)


questions_file = "./MultiTableQA/himanshu/merged_questions.json"
with open(questions_file, 'r') as f:
    questions_data = json.load(f)


# questions_data=questions_data[:200]
results_file=None

all_metrics = []
overall_metrics = {
    "total_recall": 0,
    "total_precision": 0,
    "total_f1": 0,
    "full_recall_count": 0
}


import os
retrieved_tables_json_data = "./results_single_gpu_threaded_3+2_dense.json"
if os.path.exists(retrieved_tables_json_data):
    with open(retrieved_tables_json_data, 'r') as f:
        retrieved_tables_json = json.load(f)

t = get_retrieved_tables(retrieved_tables_json,2654)
t

d = create_llama_index_documents(table_repository=table_repository, retrieved_table_ids=t)
d

t = transform_retrieved_nodes(d)
t


import torch
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import numpy as np
import time
import json
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
import os
import gc
import logging
from datetime import datetime
import pickle
import sys
import queue
import threading
from itertools import combinations
from functools import partial
import hashlib

class TableSimilarityCache:
    """
    Persistent cache for table-to-table similarity scores to avoid recomputation.
    """
    
    def __init__(self, cache_dir="table_cache", cache_version="v1"):
        self.cache_dir = cache_dir
        self.cache_version = cache_version
        self.memory_cache = {}  # In-memory cache for fast access
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Load existing cache into memory on startup
        self._load_memory_cache()
    
    def _get_cache_key(self, table_id_1, table_id_2, columns_1, columns_2):
        """Generate a deterministic cache key for a table pair"""
        # Ensure consistent ordering (smaller table_id first)
        if table_id_1 > table_id_2:
            table_id_1, table_id_2 = table_id_2, table_id_1
            columns_1, columns_2 = columns_2, columns_1
        
        # Create hash of column descriptions for content-based caching
        cols_1_str = "|".join(sorted(columns_1))
        cols_2_str = "|".join(sorted(columns_2))
        content_hash = hashlib.md5(f"{cols_1_str}||{cols_2_str}".encode()).hexdigest()[:12]
        
        return f"{table_id_1}_{table_id_2}_{content_hash}_{self.cache_version}"
    
    def _get_cache_file_path(self, cache_key):
        """Get the file path for a cache key"""
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")
    
    def _load_memory_cache(self):
        """Load all cached similarities into memory for fast access"""
        if not os.path.exists(self.cache_dir):
            return
        
        cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.pkl')]
        
        for cache_file in cache_files:
            try:
                cache_key = cache_file.replace('.pkl', '')
                cache_path = os.path.join(self.cache_dir, cache_file)
                
                with open(cache_path, 'rb') as f:
                    similarity_score = pickle.load(f)
                    self.memory_cache[cache_key] = similarity_score
                    
            except Exception as e:
                logging.getLogger('table_discovery').warning(f"Failed to load cache file {cache_file}: {e}")
        
        logging.getLogger('table_discovery').info(f"Loaded {len(self.memory_cache)} cached similarities into memory")
    
    def get_similarity(self, table_id_1, table_id_2, columns_1, columns_2):
        """Get cached similarity score if available"""
        cache_key = self._get_cache_key(table_id_1, table_id_2, columns_1, columns_2)
        
        if cache_key in self.memory_cache:
            self.cache_hits += 1
            return self.memory_cache[cache_key]
        
        self.cache_misses += 1
        return None
    
    def set_similarity(self, table_id_1, table_id_2, columns_1, columns_2, similarity_score):
        """Cache a similarity score both in memory and on disk"""
        cache_key = self._get_cache_key(table_id_1, table_id_2, columns_1, columns_2)
        
        # Store in memory cache
        self.memory_cache[cache_key] = similarity_score
        
        # Store on disk (async to avoid blocking)
        try:
            cache_path = self._get_cache_file_path(cache_key)
            with open(cache_path, 'wb') as f:
                pickle.dump(similarity_score, f)
        except Exception as e:
            logging.getLogger('table_discovery').warning(f"Failed to save cache for {cache_key}: {e}")
    
    def get_cache_stats(self):
        """Get cache performance statistics"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
            "cached_entries": len(self.memory_cache)
        }
    
    def clear_cache(self):
        """Clear both memory and disk cache"""
        self.memory_cache.clear()
        
        if os.path.exists(self.cache_dir):
            for cache_file in os.listdir(self.cache_dir):
                if cache_file.endswith('.pkl'):
                    os.remove(os.path.join(self.cache_dir, cache_file))

def process_table_pair_batch_cached(table_pair_batch, table_columns, cross_encoder_model, gpu_lock, similarity_cache):
    """
    Process a batch of table pairs with caching support.
    """
    results = {}
    uncached_pairs = []
    
    try:
        # First pass: check cache for all pairs
        for (i, j, table_i, table_j) in table_pair_batch:
            columns_i = table_columns[table_i]
            columns_j = table_columns[table_j]
            
            # Try to get from cache first
            cached_score = similarity_cache.get_similarity(table_i, table_j, columns_i, columns_j)
            
            if cached_score is not None:
                results[(i, j)] = cached_score
            else:
                uncached_pairs.append((i, j, table_i, table_j, columns_i, columns_j))
        
        # Second pass: compute similarities for uncached pairs
        if uncached_pairs:
            for (i, j, table_i, table_j, columns_i, columns_j) in uncached_pairs:
                # Create all column pairs for this table pair
                pairs = [[col_i, col_j] for col_i in columns_i for col_j in columns_j]
                
                if pairs:
                    # Use GPU lock to ensure thread-safe access to GPU model
                    with gpu_lock:
                        scores = cross_encoder_model.predict(pairs, convert_to_tensor=True)
                        if hasattr(scores, 'cpu'):
                            scores = scores.cpu().float().numpy()
                        elif hasattr(scores, 'item'):
                            scores = np.array([score.item() for score in scores])
                    
                    max_score = float(np.max(scores))
                    results[(i, j)] = max_score
                    
                    # Cache the computed score
                    similarity_cache.set_similarity(table_i, table_j, columns_i, columns_j, max_score)
    
    except Exception as e:
        logging.getLogger('table_discovery').error(f"Error in cached table pair batch processing: {e}")
        
    return results

def get_table_to_table_scores_threaded_cached(tables_dict, cross_encoder_model, query_table_scores, 
                                             max_workers=4, similarity_cache=None):
    """
    Multithreaded version of table-to-table scoring with caching support.
    
    Args:
        tables_dict: Dictionary containing table information with column descriptions
        cross_encoder_model: An instance of a CrossEncoder model
        query_table_scores: Dictionary with table_ids as keys and query similarity scores as values
        max_workers: Number of threads to use for parallel processing
        similarity_cache: TableSimilarityCache instance for caching similarities

    Returns:
        A dictionary with table_ids as keys and their total weighted similarity scores as values
    """
    
    # Use default cache if none provided
    if similarity_cache is None:
        similarity_cache = TableSimilarityCache()
    
    table_ids = list(tables_dict.keys())
    n_tables = len(table_ids)
    
    # Pre-extract all column descriptions
    table_columns = {table_id: tables_dict[table_id]['column_descriptions'] 
                    for table_id in table_ids}
    
    # Create similarity matrix for table-table similarities
    similarity_matrix = np.zeros((n_tables, n_tables))
    
    # Generate all unique table pairs with their indices
    table_pairs = [(i, j, table_ids[i], table_ids[j]) for i, j in combinations(range(n_tables), 2)]
    
    # Log cache stats before processing
    cache_stats = similarity_cache.get_cache_stats()
    logging.getLogger('table_discovery').info(
        f"Cache stats before processing: {cache_stats['cached_entries']} entries, "
        f"{cache_stats['hit_rate']:.3f} hit rate"
    )
    
    # Create a GPU lock for thread-safe access
    gpu_lock = threading.Lock()
    
    # Split pairs into batches for threading
    batch_size = max(1, len(table_pairs) // (max_workers * 2))  # Dynamic batch sizing
    pair_batches = [table_pairs[i:i + batch_size] for i in range(0, len(table_pairs), batch_size)]
    
    # Process batches in parallel using threads with caching
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all batch processing tasks with cache
        future_to_batch = {
            executor.submit(process_table_pair_batch_cached, batch, table_columns, 
                          cross_encoder_model, gpu_lock, similarity_cache): batch 
            for batch in pair_batches
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_batch):
            try:
                batch_results = future.result()
                
                # Update similarity matrix with batch results
                for (i, j), max_score in batch_results.items():
                    similarity_matrix[i, j] = max_score
                    similarity_matrix[j, i] = max_score  # Symmetric
                    
            except Exception as e:
                logging.getLogger('table_discovery').error(f"Cached thread execution failed: {e}")
    
    # Log cache stats after processing
    final_cache_stats = similarity_cache.get_cache_stats()
    logging.getLogger('table_discovery').info(
        f"Cache stats after processing: {final_cache_stats['cached_entries']} entries, "
        f"{final_cache_stats['hit_rate']:.3f} hit rate, "
        f"Hits: {final_cache_stats['cache_hits']}, Misses: {final_cache_stats['cache_misses']}"
    )
    
    # Calculate weighted scores for each table (this part is already fast)
    table_scores = {}
    for i, table_i in enumerate(table_ids):
        weighted_scores = []
        for j, table_j in enumerate(table_ids):
            if i != j:
                # Get table-table similarity
                table_similarity = similarity_matrix[i, j]
                # Get query-table similarity for table_j
                query_score_j = query_table_scores.get(table_j, 0.0)
                # Multiply and collect
                weighted_scores.append(table_similarity * query_score_j)
        # Take the mean (avoid division by zero)
        table_scores[table_i] = np.mean(weighted_scores) if weighted_scores else 0.0

    return table_scores


import torch
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import numpy as np
import time
import json
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
import os
import gc
import logging
from datetime import datetime
import pickle
import sys
import queue
import threading
from itertools import combinations
from functools import partial

# Set up comprehensive logging for multiprocessing
def setup_logging():
    """Setup logging with multiple handlers for different log levels"""
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(processName)s - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Main logger
    logger = logging.getLogger('table_discovery')
    logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # File handler for all logs (detailed)
    file_handler = logging.FileHandler(f'logs/processing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    
    # File handler for metrics only
    metrics_handler = logging.FileHandler(f'logs/metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    metrics_handler.setLevel(logging.INFO)
    metrics_handler.setFormatter(simple_formatter)
    
    # Add filter to only log metrics
    class MetricsFilter(logging.Filter):
        def filter(self, record):
            return 'METRICS' in record.getMessage()
    
    metrics_handler.addFilter(MetricsFilter())
    logger.addHandler(metrics_handler)
    
    # Console handler for important messages only
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)  # Only warnings and errors to console
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    # Progress handler for notebook display
    progress_handler = logging.FileHandler(f'logs/progress_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    progress_handler.setLevel(logging.INFO)
    progress_handler.setFormatter(simple_formatter)
    
    class ProgressFilter(logging.Filter):
        def filter(self, record):
            return 'PROGRESS' in record.getMessage()
    
    progress_handler.addFilter(ProgressFilter())
    logger.addHandler(progress_handler)
    
    return logger

def calculate_metrics_by_id(expected_table_ids: List[str], retrieved_table_ids: List[str], 
                        id_to_name_map: Dict[str, str] = None):
    """
    Calculate recall, precision, F1 score, full recall, and full precision based on table IDs
    
    Args:
        expected_table_ids: List of expected table IDs
        retrieved_table_ids: List of retrieved table IDs
        id_to_name_map: Optional mapping from table ID to table name for display purposes
    """
    # Create sets of expected and retrieved IDs
    expected_set = set(expected_table_ids)
    retrieved_set = set(retrieved_table_ids)
    
    # Find common table IDs
    correct_table_ids = expected_set.intersection(retrieved_set)
    
    # Calculate metrics
    recall = len(correct_table_ids) / len(expected_set) if expected_set else 0
    precision = len(correct_table_ids) / len(retrieved_set) if retrieved_set else 0
    
    # F1 score
    f1 = 0
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    
    # Full recall (1 if all expected tables were retrieved, 0 otherwise)
    full_recall = 1 if expected_set.issubset(retrieved_set) else 0
    # Full precision (1 if all retrieved tables are expected, 0 otherwise)
    full_precision = 1 if retrieved_set.issubset(expected_set) else 0
    
    # Prepare result with IDs
    result = {
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "full_recall": full_recall,
        "full_precision": full_precision,
        "expected_table_ids": list(expected_set),
        "retrieved_table_ids": list(retrieved_set),
        "correct_table_ids": list(correct_table_ids),
        "missing_table_ids": list(expected_set - retrieved_set),
        "extra_table_ids": list(retrieved_set - expected_set)
    }
    
    return result

def process_table_pair_batch(table_pair_batch, table_columns, cross_encoder_model, gpu_lock):
    """
    Process a batch of table pairs for similarity computation.
    This function will run in a separate thread.
    """
    results = {}
    
    try:
        for (i, j, table_i, table_j) in table_pair_batch:
            columns_i = table_columns[table_i]
            columns_j = table_columns[table_j]
            
            # Create all column pairs for this table pair
            pairs = [[col_i, col_j] for col_i in columns_i for col_j in columns_j]
            
            if pairs:
                # Use GPU lock to ensure thread-safe access to GPU model
                with gpu_lock:
                    scores = cross_encoder_model.predict(pairs, convert_to_tensor=True)
                    if hasattr(scores, 'cpu'):
                        scores = scores.cpu().float().numpy()
                    elif hasattr(scores, 'item'):
                        scores = np.array([score.item() for score in scores])
                
                max_score = np.max(scores)
                results[(i, j)] = max_score
    
    except Exception as e:
        logging.getLogger('table_discovery').error(f"Error in table pair batch processing: {e}")
        
    return results

def get_table_to_table_scores_threaded(tables_dict, cross_encoder_model, query_table_scores, max_workers=4):
    """
    Multithreaded version of table-to-table scoring with optimized batching.
    
    Args:
        tables_dict: Dictionary containing table information with column descriptions
        cross_encoder_model: An instance of a CrossEncoder model
        query_table_scores: Dictionary with table_ids as keys and query similarity scores as values
        max_workers: Number of threads to use for parallel processing

    Returns:
        A dictionary with table_ids as keys and their total weighted similarity scores as values
    """
    
    table_ids = list(tables_dict.keys())
    n_tables = len(table_ids)
    
    # Pre-extract all column descriptions
    table_columns = {table_id: tables_dict[table_id]['column_descriptions'] 
                    for table_id in table_ids}
    
    # Create similarity matrix for table-table similarities
    similarity_matrix = np.zeros((n_tables, n_tables))
    
    # Generate all unique table pairs with their indices
    table_pairs = [(i, j, table_ids[i], table_ids[j]) for i, j in combinations(range(n_tables), 2)]
    
    # Create a GPU lock for thread-safe access
    gpu_lock = threading.Lock()
    
    # Split pairs into batches for threading
    batch_size = max(1, len(table_pairs) // (max_workers * 2))  # Dynamic batch sizing
    pair_batches = [table_pairs[i:i + batch_size] for i in range(0, len(table_pairs), batch_size)]
    
    # Process batches in parallel using threads
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all batch processing tasks
        future_to_batch = {
            executor.submit(process_table_pair_batch, batch, table_columns, cross_encoder_model, gpu_lock): batch 
            for batch in pair_batches
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_batch):
            try:
                batch_results = future.result()
                
                # Update similarity matrix with batch results
                for (i, j), max_score in batch_results.items():
                    similarity_matrix[i, j] = max_score
                    similarity_matrix[j, i] = max_score  # Symmetric
                    
            except Exception as e:
                logging.getLogger('table_discovery').error(f"Thread execution failed: {e}")
    
    # Calculate weighted scores for each table (this part is already fast)
    table_scores = {}
    for i, table_i in enumerate(table_ids):
        weighted_scores = []
        for j, table_j in enumerate(table_ids):
            if i != j:
                # Get table-table similarity
                table_similarity = similarity_matrix[i, j]
                # Get query-table similarity for table_j
                query_score_j = query_table_scores.get(table_j, 0.0)
                # Multiply and collect
                weighted_scores.append(table_similarity * query_score_j)
        # Take the mean (avoid division by zero)
        table_scores[table_i] = np.mean(weighted_scores) if weighted_scores else 0.0

    return table_scores

def process_query_table_batch(batch_data, cross_encoder_model, gpu_lock):
    """
    Process a batch of query-table comparisons.
    This function will run in a separate thread.
    """
    query, node_batch = batch_data
    scores = {}
    
    try:
        # Prepare all sentence pairs for this batch
        sentence_pairs = []
        table_ids = []
        
        for node in node_batch:
            if hasattr(node, 'node'):
                # If the node is a NodeWithScore, extract the underlying Document
                doc = node.node
                table_id = doc.metadata.get("table_id", "unknown_table")
                doc_texts = doc.text
            else:
                # If the node is a Document, handle it directly
                table_id = node.metadata.get("table_id", "unknown_table")
                doc_texts = node.text_resource.text
            
            sentence_pairs.append([query, doc_texts])
            table_ids.append(table_id)
        
        # Batch process all pairs at once with GPU lock
        with gpu_lock:
            batch_scores = cross_encoder_model.predict(sentence_pairs, convert_to_tensor=True)
            if hasattr(batch_scores, 'cpu'):
                batch_scores = batch_scores.cpu().float().numpy()
            elif hasattr(batch_scores, 'item'):
                batch_scores = np.array([score.item() for score in batch_scores])
        
        # Map scores back to table IDs
        for table_id, score in zip(table_ids, batch_scores):
            scores[table_id] = float(score)
    
    except Exception as e:
        logging.getLogger('table_discovery').error(f"Error in query-table batch processing: {e}")
        
    return scores

def get_table_query_scores_threaded(query, retrieved_nodes, cross_encoder_model, max_workers=4):
    """
    Multithreaded version of query-table scoring with batch processing.

    Args:
        query: The user's query string.
        retrieved_nodes: List of LlamaIndex Document objects (one per table).
        cross_encoder_model: An instance of a CrossEncoder model.
        max_workers: Number of threads to use for parallel processing.

    Returns:
        A dictionary with table_ids as keys and query similarity scores as values.
    """
    
    # Split nodes into batches for threading
    batch_size = max(1, len(retrieved_nodes) // (max_workers * 2))  # Dynamic batch sizing
    node_batches = [retrieved_nodes[i:i + batch_size] for i in range(0, len(retrieved_nodes), batch_size)]
    
    # Create a GPU lock for thread-safe access
    gpu_lock = threading.Lock()
    
    # Prepare batch data (query is same for all batches)
    batch_data_list = [(query, batch) for batch in node_batches]
    
    scores = {}
    
    # Process batches in parallel using threads
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all batch processing tasks
        future_to_batch = {
            executor.submit(process_query_table_batch, batch_data, cross_encoder_model, gpu_lock): batch_data 
            for batch_data in batch_data_list
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_batch):
            try:
                batch_scores = future.result()
                scores.update(batch_scores)
                
            except Exception as e:
                logging.getLogger('table_discovery').error(f"Query-table thread execution failed: {e}")
    
    return scores

def rerank_by_qt_ttqt(query_scores, table_table_scores, alpha=0.7, beta=0.3, threshold=0.5):
    """
    Rerank tables based on query-table and table-table scores.
    
    Args:
        query_scores: Dictionary with table_ids as keys and query similarity scores as values.
        table_table_scores: Dictionary with table_ids as keys and their total weighted similarity scores as values.
        alpha: Weight for query-table scores
        beta: Weight for table-table scores
        threshold: Threshold for filtering results
    
    Returns:
        List of tuples (table_id, final_score, probability) sorted by final_score in descending order.
    """
    
    # Combine scores
    combined_scores = {}
    for table_id in query_scores.keys():
        qt_score = query_scores.get(table_id, 0.0)
        ttqt_score = table_table_scores.get(table_id, 0.0)
        
        # Final score is a weighted sum
        final_score = alpha * qt_score + beta * ttqt_score
        combined_scores[table_id] = final_score
    
    # Sort by final score
    sorted_tables = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

    # Compute softmax over all final scores for probability distribution
    scores = np.array([score for _, score in sorted_tables])
    exp_scores = np.exp(scores - np.max(scores))
    softmax_probs = exp_scores / exp_scores.sum()
    
    # Attach probability to each table
    sorted_tables = [(table_id, score, prob) for (table_id, score), prob in zip(sorted_tables, softmax_probs)]
    # mean_prob = np.mean(softmax_probs)
    median_prob = np.median(softmax_probs)

    # Limit to tables with probability >= median probability
    # Take top 3 tables by final score (with probability >= median)
    filtered_tables = [(table_id, score, prob) for (table_id, score, prob) in sorted_tables[:3]]
    # filtered_tables = [(table_id, score, prob) for (table_id, score, prob) in sorted_tables if prob >= 0.1]
    
    return filtered_tables

class OptimizedHybridTableDiscovery:
    """Enhanced hybrid approach with multithreading for GPU operations"""
    
    def __init__(self, retriever, faiss_store, lc_embed_model, rerank, 
                 table_repository, top_k, max_cpu_workers=4, max_gpu_threads=2, cache_dir="table_cache", results_file="results.json"):
        self.retriever = retriever
        self.faiss_store = faiss_store
        self.lc_embed_model = lc_embed_model
        self.rerank = rerank
        self.table_repository = table_repository
        self.top_k = top_k
        self.max_cpu_workers = max_cpu_workers
        self.max_gpu_threads = max_gpu_threads
        self.results_file = results_file
        
        # Initialize similarity cache
        self.similarity_cache = TableSimilarityCache(cache_dir=cache_dir)
        
        
        # GPU operations lock (for threading within main process)
        self.gpu_lock = threading.Lock()
        
        # Results tracking
        self.all_metrics = []
        self.processed_count = 0
        
        # Embedding cache for GPU operations (main process only)
        self.embedding_cache = {}
        self.embedding_lock = threading.Lock()

        # Setup logging
        self.logger = logging.getLogger('table_discovery.main')
        
        # Setup CPU worker pool (if needed for other tasks)
        self.setup_cpu_workers()
        
    def setup_cpu_workers(self):
        """Setup CPU worker pool for non-GPU tasks"""
        try:
            # For now, we'll focus on GPU threading. CPU workers can be added later if needed.
            self.cpu_executor = None
            self.logger.info("GPU threading optimization enabled")
            
        except Exception as e:
            self.logger.error(f"Failed to setup CPU workers: {e}")
            self.cpu_executor = None
    
    def process_single_question(self, question_data, question_index, delta):
        """Process a single question using optimized hybrid approach with threading"""
        
        try:
            question_id = question_data.get("question_id", question_index + 1)
            question = question_data.get("question", "")
            
            self.logger.debug(f"Processing question {question_id}: {question[:100]}...")
            
            # Extract expected table IDs
            expected_table_ids = []
            id_to_name_map = {}
            
            if "tables" in question_data:
                for table in question_data["tables"]:
                    table_id = table.get("id", "")
                    table_name = table.get("name", "")
                    if table_id:
                        expected_table_ids.append(table_id)
                        id_to_name_map[table_id] = table_name
            elif "table_names" in question_data:
                expected_table_ids = question_data.get("table_names", [])
                self.logger.warning(f"Question {question_id} uses old format without table IDs")
            
            self.logger.debug(f"Expected tables for Q{question_id}: {expected_table_ids}")


            ### -----------BASE RETRIEVAL STEP ----------------- ###
            
            # retrieved_nodes = retriever.retrieve(question)
            
            # Get retrieved table IDs (assuming this function exists in your codebase)
            # retrieved_table_ids = get_retrieved_tables(retrieved_tables_json, question_id)

            # retrieved_nodes = create_llama_index_documents(table_repository=self.table_repository, retrieved_table_ids=retrieved_table_ids)

            # response = query_engine.query(
            #     question
            # )
            # retrieved_nodes = response.source_nodes

            # retrieved_nodes = bm25_retriever.retrieve(
            #         question
            #     )

            retrieved_nodes = self.retriever.retrieve(question)

            ### -----------END BASE RETRIEVAL STEP ----------------- ###



            ### ----------- ADDITION STEPS ----------------- ###

            # retrieved_nodes_tr = transform_retrieved_nodes(retrieved_nodes)
            # start_time = time.time()
            # self.logger.debug(f"Retrieved {len(retrieved_nodes_tr)} nodes for question {question_id} in {time.time() - start_time:.2f}s")
            # self.logger.debug(f"Retrieved nodes: {retrieved_nodes_tr}")
            # with self.gpu_lock:
            #     joinable_results = find_joinable_tables_with_faiss(
            #         retrieved_nodes_tr, self.faiss_store, 
            #         self.lc_embed_model, threshold=0.7
            #     )
            # new_docs = create_llama_index_documents_joinable(self.table_repository, joinable_results)
            # # self.logger.debug(f"Joinable tables found for question {question_id}: {joinable_results}")
            # self.logger.debug(f"Joinable tables found for question {question_id}: {len(new_docs)}")
            # # Step 3: Ranking with retry logic (GPU operation in main process)
            # attempts = 0
            # top_tables = []
            # while attempts < 3:
            #     try:
            #         start_time = time.time()
            #         with self.gpu_lock:
            #             top_tables = rank_tables_by_query_similarity_with_cross_encoder(
            #                 query=question, 
            #                 documents=new_docs,
            #                 cross_encoder_model=self.rerank,
            #                 top_k=delta
            #             )
            #         self.logger.debug(f"Table ranking took {time.time() - start_time:.2f}s, selected {len(top_tables)} tables")
            #         break
                    
            #     except Exception as e:
            #         attempts += 1
            #         self.logger.warning(f"Ranking attempt {attempts} failed for question {question_id}: {e}")
            #         if attempts < 3:
            #             time.sleep(1)
            #             # Clear GPU memory and try again
            #             torch.cuda.empty_cache()
            #             gc.collect()
            #         else:
            #             self.logger.error(f"All ranking attempts failed for question {question_id}")
            #             return self.create_empty_metrics(question_id, question, expected_table_ids, id_to_name_map)
                    
            
            # delta_docs = filter_documents_by_table_ids(new_docs, top_tables)
            # self.logger.debug(f"Filtered delta docs for question {question_id}: {len(delta_docs)}")
            # self.logger.debug(f"Delta docs: {delta_docs}")
            # retrieved_nodes.extend(delta_docs)
            # self.logger.debug(f"Total retrieved nodes after delta filtering for question {question_id}: {len(retrieved_nodes)}")
        
            # self.logger.debug(f"Total retrieved nodes for question {question_id}: {retrieved_nodes}")


            #### ----------- END ADDITION STEPS ----------------- ###


            # this just extracts table IDs from the retrieved nodes for logging and metrics
            retrieved_table_ids = set()
            for node in retrieved_nodes:
                if isinstance(node, NodeWithScore):
                    retrieved_table_ids.add(node.node.metadata["table_id"])
                elif isinstance(node, Document):
                    retrieved_table_ids.add(node.metadata["table_id"])

            self.logger.debug(f"Retrieved table IDs for question {question_id}: {retrieved_table_ids}")
            if len(retrieved_table_ids) !=3 :
                self.logger.warning("-----------------WARNINGGGGGG------------------")
                self.logger.warning(f"Question {question_id} retrieved only {len(retrieved_table_ids)} tables, which is less than the expected 5. This may affect results.")
            
            final_table_list = list(retrieved_table_ids)


            ### ----------- PRUNING STEP ----------------- ###

            # # # Process with multithreading
            # if retrieved_nodes:
            #     # Transform nodes (assuming this function exists)
            #     transformed_nodes = transform_retrieved_nodes(retrieved_nodes)

            #     # Step 1: Get query-table scores (threaded)
            #     start_time = time.time()
            #     query_scores = get_table_query_scores_threaded(
            #         question, retrieved_nodes, self.rerank, max_workers=self.max_gpu_threads
            #     )
            #     query_time = time.time() - start_time
            #     self.logger.debug(f"Threaded query-table scoring took {query_time:.2f}s")
                
            #     # Step 2: Get table-to-table scores (threaded)
            #     # start_time = time.time()
            #     # table_table_scores = get_table_to_table_scores_threaded(
            #     #     transformed_nodes, self.rerank, query_scores, max_workers=self.max_gpu_threads
            #     # )
            #     start_time = time.time()
            #     table_table_scores = get_table_to_table_scores_threaded_cached(
            #         transformed_nodes, self.rerank, query_scores, 
            #         max_workers=self.max_gpu_threads, similarity_cache=self.similarity_cache
            #     )
            #     table_time = time.time() - start_time
            #     self.logger.debug(f"Threaded table-table scoring took {table_time:.2f}s")
                
            #     # Step 3: Rerank and get final results
            #     final_tables = rerank_by_qt_ttqt(
            #         query_scores, table_table_scores, alpha=0.9, beta=0.1, threshold=0.1
            #     )
            #     final_table_list = [table_id for table_id, _, _ in final_tables]
            # else:
            #     final_table_list = retrieved_table_ids

            ### ----------- END PRUNING STEP ----------------- ###

            # Calculate metrics
            metrics = calculate_metrics_by_id(expected_table_ids, final_table_list, id_to_name_map)
            metrics["question_id"] = question_id
            metrics["question"] = question
            
            # Log detailed metrics
            self.logger.info(f"METRICS - Question {question_id} completed:")
            self.logger.info(f"METRICS - Recall: {metrics['recall']:.4f}")
            self.logger.info(f"METRICS - Precision: {metrics['precision']:.4f}")
            self.logger.info(f"METRICS - F1 Score: {metrics['f1']:.4f}")
            self.logger.info(f"METRICS - Full Recall: {metrics['full_recall']}")
            
            if "expected_table_ids" in metrics and "retrieved_table_ids" in metrics:
                self.logger.info(f"METRICS - Expected Tables: {', '.join(metrics['expected_table_ids'])}")
                self.logger.info(f"METRICS - Retrieved Tables: {', '.join(metrics['retrieved_table_ids'])}")
                if metrics.get('missing_table_ids'):
                    self.logger.info(f"METRICS - Missing Tables: {', '.join(metrics['missing_table_ids'])}")
                if metrics.get('extra_table_ids'):
                    self.logger.info(f"METRICS - Extra Tables: {', '.join(metrics['extra_table_ids'])}")
            
            self.logger.debug(f"Question {question_id} processing complete")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Question {question_id} processing failed: {e}", exc_info=True)
            return None
        finally:
            # Clean up GPU memory after each question
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
    
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'cpu_executor') and self.cpu_executor:
            self.cpu_executor.shutdown(wait=True)

        # Log final cache stats
        final_stats = self.get_cache_statistics()
        self.logger.info(f"Final cache statistics: {final_stats}")

    def get_cache_statistics(self):
        """Get comprehensive cache statistics"""
        return self.similarity_cache.get_cache_stats()
    
        

def run_optimized_single_gpu_discovery(questions_data, retriever, faiss_store, lc_embed_model,
                                     rerank, table_repository, top_k, delta, max_gpu_threads=None):
    """
    Optimized approach with multithreading for GPU operations - replaces the original single GPU function.
    
    Args:
        questions_data: List of question dictionaries
        retriever: Document retriever
        faiss_store: FAISS vector store
        lc_embed_model: Embedding model
        rerank: CrossEncoder reranking model
        table_repository: Table repository
        top_k: Top-k parameter
        delta: Delta parameter
        max_gpu_threads: Maximum number of threads for GPU operations (auto-detected if None)
    """
    
    logger = setup_logging()
    logger.info("=== STARTING OPTIMIZED SINGLE GPU TABLE DISCOVERY (MULTITHREADED) ===")
    
    # System configuration
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    num_cpus = os.cpu_count()
    available_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0
    
    # Auto-detect optimal thread count if not specified
    if max_gpu_threads is None:
        if available_vram < 16:  # Less than 16GB VRAM
            max_gpu_threads = 1
        elif available_vram < 24:  # Less than 24GB VRAM
            max_gpu_threads = 2
        elif available_vram < 40:  # Less than 40GB VRAM
            max_gpu_threads = 3
        else:  # 40GB+ VRAM
            max_gpu_threads = 4
    
    # Conservative settings for single GPU systems
    max_cpu_workers = min(4, num_cpus - 2)  # Leave 2 cores for main process
    max_cpu_workers = max(1, max_cpu_workers)
    
    # Optimize based on available memory (same logic as original)
    memory_per_worker = 45 / (max_cpu_workers + 1)  # +1 for main process
    if memory_per_worker < 4:  # If less than 4GB per worker
        max_cpu_workers = max(1, 45 // 5)  # Ensure at least 5GB per process
        print(f"Reduced workers to {max_cpu_workers} due to memory constraints")
    
    logger.info(f"System Configuration:")
    logger.info(f"- Available GPUs: {num_gpus}")
    logger.info(f"- GPU VRAM: {available_vram:.1f} GB")
    logger.info(f"- Available CPUs: {num_cpus}")
    logger.info(f"- Available RAM: 45 GB")
    logger.info(f"- Using {max_cpu_workers} CPU workers")
    logger.info(f"- GPU Threads: {max_gpu_threads} (auto-detected)")
    logger.info(f"- Estimated RAM per worker: {memory_per_worker:.1f} GB")
    logger.info(f"- Main process handles all GPU operations with threading")
    logger.info(f"- Hybrid join discovery: GPU embeddings + CPU similarities (threaded)")
    
    logger.info(f"Processing Configuration:")
    logger.info(f"- Total questions: {len(questions_data)}")
    logger.info(f"- Top-k parameter: {top_k}")
    logger.info(f"- Delta parameter: {delta}")
    logger.info(f"- Threading enabled: {max_gpu_threads > 1}")
    
    # Initialize optimized processor (updated class name)
    processor = OptimizedHybridTableDiscovery(
        retriever=retriever,
        faiss_store=faiss_store,
        lc_embed_model=lc_embed_model,
        rerank=rerank,
        table_repository=table_repository,
        top_k=top_k,
        max_cpu_workers=max_cpu_workers,
        max_gpu_threads=max_gpu_threads
    )
    
    # Initialize metrics tracking
    overall_metrics = {
        "total_recall": 0.0, "total_precision": 0.0, "total_f1": 0.0,
        "full_recall_count": 0, "avg_recall": 0.0, "avg_precision": 0.0,
        "avg_f1": 0.0, "avg_full_recall": 0.0, "avg_full_precision": 0.0
    }
    all_metrics = []
    
    # Process questions sequentially with threaded GPU operations
    start_time = time.time()
    logger.info(f"Starting optimized single GPU processing at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        with tqdm(total=len(questions_data), desc="Processing Questions") as pbar:
            for i, question_data in enumerate(questions_data):
                
                metrics = processor.process_single_question(question_data, i, delta)
                
                if metrics is not None:
                    # Update overall metrics
                    overall_metrics["total_recall"] += metrics["recall"]
                    overall_metrics["total_precision"] += metrics["precision"]
                    overall_metrics["total_f1"] += metrics["f1"]
                    overall_metrics["full_recall_count"] += metrics["full_recall"]
                    
                    all_metrics.append(metrics)
                    
                    # Log progress every 10 questions (more frequent for single GPU)
                    if len(all_metrics) % 10 == 0:
                        elapsed_time = time.time() - start_time
                        avg_time_per_question = elapsed_time / len(all_metrics)
                        remaining_questions = len(questions_data) - len(all_metrics)
                        estimated_remaining_time = avg_time_per_question * remaining_questions
                        
                        # Check GPU memory usage and embedding cache size
                        if torch.cuda.is_available():
                            gpu_memory_used = torch.cuda.memory_allocated(0) / (1024**3)
                            gpu_memory_cached = torch.cuda.memory_reserved(0) / (1024**3)
                        else:
                            gpu_memory_used = gpu_memory_cached = 0
                        
                        embedding_cache_size = len(processor.embedding_cache)
                        
                        logger.info(f"PROGRESS - {len(all_metrics)}/{len(questions_data)} questions completed")
                        logger.info(f"PROGRESS - Elapsed time: {elapsed_time:.1f}s")
                        logger.info(f"PROGRESS - Avg time per question: {avg_time_per_question:.2f}s")
                        logger.info(f"PROGRESS - Estimated remaining time: {estimated_remaining_time:.1f}s")
                        logger.info(f"PROGRESS - Recent F1 score: {metrics['f1']:.4f}")
                        logger.info(f"PROGRESS - GPU Memory: {gpu_memory_used:.1f}GB used, {gpu_memory_cached:.1f}GB cached")
                        logger.info(f"PROGRESS - Embedding cache: {embedding_cache_size} columns")
                        
                        # Print brief progress to console
                        print(f"Progress: {len(all_metrics)}/{len(questions_data)} - F1: {metrics['f1']:.4f} - GPU: {gpu_memory_used:.1f}GB - Threads: {max_gpu_threads} - Cache: {embedding_cache_size} - ETA: {estimated_remaining_time:.0f}s")
                
                pbar.update(1)
                
                # Aggressive memory management for single GPU (every 5 questions)
                if i % 5 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
    
    finally:
        # Cleanup resources
        processor.cleanup()
    
    end_time = time.time()
    total_time = end_time - start_time
    
    logger.info("=== PROCESSING COMPLETED ===")
    logger.info(f"Total processing time: {total_time:.2f} seconds")
    logger.info(f"Average time per question: {total_time / len(questions_data):.2f} seconds")
    logger.info(f"Questions processed successfully: {len(all_metrics)}/{len(questions_data)}")
    
    # Calculate final metrics
    num_questions = len(all_metrics)
    if num_questions > 0:
        overall_metrics["avg_recall"] = overall_metrics["total_recall"] / num_questions
        overall_metrics["avg_precision"] = overall_metrics["total_precision"] / num_questions
        overall_metrics["avg_f1"] = overall_metrics["total_f1"] / num_questions
        overall_metrics["avg_full_recall"] = overall_metrics["full_recall_count"] / num_questions
        overall_metrics["avg_full_precision"] = overall_metrics["full_recall_count"] / num_questions
    
    # Log final metrics
    logger.info("=== FINAL METRICS ===")
    logger.info(f"Average Recall: {overall_metrics['avg_recall']:.4f}")
    logger.info(f"Average Precision: {overall_metrics['avg_precision']:.4f}")
    logger.info(f"Average F1 Score: {overall_metrics['avg_f1']:.4f}")
    logger.info(f"Average Full Recall: {overall_metrics['avg_full_recall']:.4f}")
    logger.info(f"Questions with Full Recall: {overall_metrics['full_recall_count']}/{len(all_metrics)}")
    
    # Print summary to console
    print("\n=== OVERALL METRICS ===")
    print(f"Average Recall: {overall_metrics['avg_recall']:.4f}")
    print(f"Average Precision: {overall_metrics['avg_precision']:.4f}")
    print(f"Average F1 Score: {overall_metrics['avg_f1']:.4f}")
    print(f"Average Full Recall: {overall_metrics['avg_full_recall']:.4f}")
    print(f"Questions with Full Recall: {overall_metrics['full_recall_count']}/{len(all_metrics)}")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"GPU Memory Efficiency: Optimized for single {available_vram:.0f}GB GPU")
    print(f"Threaded GPU Operations: {max_gpu_threads} parallel threads")
    print(f"Embedding Cache: {len(processor.embedding_cache)} unique columns cached")
    
    # Save results (updated filename to reflect threading)
    # output_file = f"results_single_gpu_threaded_{top_k}_qwen3.json"
    output_file = self.results_file
    with open(output_file, 'w') as f:
        json.dump({
            "overall": overall_metrics,
            "questions": all_metrics,
            "processing_time": total_time,
            "avg_time_per_question": total_time / len(questions_data),
            "system_info": {
                "num_gpus": num_gpus,
                "gpu_vram_gb": available_vram,
                "num_cpus": num_cpus,
                "max_cpu_workers": max_cpu_workers,
                "max_gpu_threads": max_gpu_threads,
                "approach": "hybrid_single_gpu_threaded_optimized"
            }
        }, f, indent=2)
    
    logger.info(f"Results saved to {output_file}")
    logger.info("=== MULTITHREADED TABLE DISCOVERY COMPLETE ===")
    
    print(f"\nDetailed metrics saved to {output_file}")
    print(f"Check logs/ directory for detailed processing logs")
    
    return overall_metrics, all_metrics

# Additional utility functions for advanced threading optimization

def adaptive_batch_size(num_items, max_workers, min_batch_size=1, max_batch_size=50):
    """
    Calculate optimal batch size based on number of items and workers.
    
    Args:
        num_items: Total number of items to process
        max_workers: Maximum number of worker threads
        min_batch_size: Minimum batch size
        max_batch_size: Maximum batch size
    
    Returns:
        Optimal batch size
    """
    if num_items <= max_workers:
        return max(1, num_items // max_workers)
    
    # Calculate ideal batch size
    ideal_batch_size = max(min_batch_size, num_items // (max_workers * 2))
    
    # Cap at maximum batch size
    return min(ideal_batch_size, max_batch_size)

def process_table_comparisons_with_memory_management(table_pairs, table_columns, cross_encoder_model, 
                                                   gpu_lock, max_memory_mb=2048):
    """
    Advanced table comparison processing with GPU memory management.
    
    Args:
        table_pairs: List of table pairs to compare
        table_columns: Dictionary of table columns
        cross_encoder_model: CrossEncoder model
        gpu_lock: Threading lock for GPU access
        max_memory_mb: Maximum GPU memory to use (in MB)
    
    Returns:
        Dictionary of comparison results
    """
    results = {}
    processed_pairs = 0
    
    try:
        for (i, j, table_i, table_j) in table_pairs:
            columns_i = table_columns[table_i]
            columns_j = table_columns[table_j]
            
            # Create all column pairs for this table pair
            pairs = [[col_i, col_j] for col_i in columns_i for col_j in columns_j]
            
            if pairs:
                # Check if we need to process in sub-batches due to memory constraints
                max_pairs_per_batch = max_memory_mb // 4  # Rough estimate: 4MB per comparison
                
                if len(pairs) > max_pairs_per_batch:
                    # Process in smaller batches
                    all_scores = []
                    for batch_start in range(0, len(pairs), max_pairs_per_batch):
                        batch_end = min(batch_start + max_pairs_per_batch, len(pairs))
                        batch_pairs = pairs[batch_start:batch_end]
                        
                        with gpu_lock:
                            batch_scores = cross_encoder_model.predict(batch_pairs, convert_to_tensor=True)
                            if hasattr(batch_scores, 'cpu'):
                                batch_scores = batch_scores.cpu().float().numpy()
                            elif hasattr(batch_scores, 'item'):
                                batch_scores = np.array([score.item() for score in batch_scores])
                        
                        all_scores.extend(batch_scores)
                        
                        # Clear GPU cache between batches
                        torch.cuda.empty_cache()
                    
                    max_score = np.max(all_scores)
                else:
                    # Process all pairs at once
                    with gpu_lock:
                        scores = cross_encoder_model.predict(pairs, convert_to_tensor=True)
                        if hasattr(scores, 'cpu'):
                            scores = scores.cpu().float().numpy()
                        elif hasattr(scores, 'item'):
                            scores = np.array([score.item() for score in scores])
                    
                    max_score = np.max(scores)
                
                results[(i, j)] = max_score
                processed_pairs += 1
                
                # Periodic memory cleanup
                if processed_pairs % 10 == 0:
                    torch.cuda.empty_cache()
    
    except Exception as e:
        logging.getLogger('table_discovery').error(f"Error in memory-managed table comparison: {e}")
        
    return results

def get_table_to_table_scores_advanced(tables_dict, cross_encoder_model, query_table_scores, 
                                      max_workers=4, memory_efficient=True):
    """
    Advanced multithreaded table-to-table scoring with memory management and adaptive batching.
    
    Args:
        tables_dict: Dictionary containing table information with column descriptions
        cross_encoder_model: An instance of a CrossEncoder model
        query_table_scores: Dictionary with table_ids as keys and query similarity scores as values
        max_workers: Number of threads to use for parallel processing
        memory_efficient: Whether to use memory-efficient processing
    
    Returns:
        A dictionary with table_ids as keys and their total weighted similarity scores as values
    """
    
    table_ids = list(tables_dict.keys())
    n_tables = len(table_ids)
    
    # Pre-extract all column descriptions
    table_columns = {table_id: tables_dict[table_id]['column_descriptions'] 
                    for table_id in table_ids}
    
    # Create similarity matrix for table-table similarities
    similarity_matrix = np.zeros((n_tables, n_tables))
    
    # Generate all unique table pairs with their indices
    table_pairs = [(i, j, table_ids[i], table_ids[j]) for i, j in combinations(range(n_tables), 2)]
    
    # Create a GPU lock for thread-safe access
    gpu_lock = threading.Lock()
    
    # Calculate optimal batch size
    batch_size = adaptive_batch_size(len(table_pairs), max_workers, min_batch_size=1, max_batch_size=20)
    pair_batches = [table_pairs[i:i + batch_size] for i in range(0, len(table_pairs), batch_size)]
    
    # Choose processing function based on memory efficiency setting
    process_func = (process_table_comparisons_with_memory_management if memory_efficient 
                   else process_table_pair_batch)
    
    # Process batches in parallel using threads
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        if memory_efficient:
            # Submit tasks with memory management
            future_to_batch = {
                executor.submit(process_func, batch, table_columns, cross_encoder_model, gpu_lock, 2048): batch 
                for batch in pair_batches
            }
        else:
            # Submit tasks with standard processing
            future_to_batch = {
                executor.submit(process_func, batch, table_columns, cross_encoder_model, gpu_lock): batch 
                for batch in pair_batches
            }
        
        # Collect results as they complete
        for future in as_completed(future_to_batch):
            try:
                batch_results = future.result()
                
                # Update similarity matrix with batch results
                for (i, j), max_score in batch_results.items():
                    similarity_matrix[i, j] = max_score
                    similarity_matrix[j, i] = max_score  # Symmetric
                    
            except Exception as e:
                logging.getLogger('table_discovery').error(f"Advanced thread execution failed: {e}")
    
    # Calculate weighted scores for each table
    table_scores = {}
    for i, table_i in enumerate(table_ids):
        weighted_scores = []
        for j, table_j in enumerate(table_ids):
            if i != j:
                # Get table-table similarity
                table_similarity = similarity_matrix[i, j]
                # Get query-table similarity for table_j
                query_score_j = query_table_scores.get(table_j, 0.0)
                # Multiply and collect
                weighted_scores.append(table_similarity * query_score_j)
        # Take the mean (avoid division by zero)
        table_scores[table_i] = np.mean(weighted_scores) if weighted_scores else 0.0

    return table_scores

class ThreadingMonitor:
    """Monitor threading performance and GPU utilization"""
    
    def __init__(self):
        self.start_time = None
        self.thread_times = []
        self.gpu_memory_snapshots = []
        self.lock = threading.Lock()
    
    def start_monitoring(self):
        """Start monitoring"""
        self.start_time = time.time()
        self.thread_times = []
        self.gpu_memory_snapshots = []
    
    def record_thread_completion(self, thread_time):
        """Record completion time for a thread"""
        with self.lock:
            self.thread_times.append(thread_time)
    
    def record_gpu_memory(self):
        """Record current GPU memory usage"""
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated(0) / (1024**3)
            memory_cached = torch.cuda.memory_reserved(0) / (1024**3)
            
            with self.lock:
                self.gpu_memory_snapshots.append({
                    'timestamp': time.time() - self.start_time,
                    'used_gb': memory_used,
                    'cached_gb': memory_cached
                })
    
    def get_statistics(self):
        """Get threading and GPU statistics"""
        if not self.thread_times:
            return {}
        
        return {
            'total_threads': len(self.thread_times),
            'avg_thread_time': np.mean(self.thread_times),
            'max_thread_time': np.max(self.thread_times),
            'min_thread_time': np.min(self.thread_times),
            'thread_time_std': np.std(self.thread_times),
            'total_monitoring_time': time.time() - self.start_time if self.start_time else 0,
            'peak_gpu_memory': max([s['used_gb'] for s in self.gpu_memory_snapshots]) if self.gpu_memory_snapshots else 0,
            'avg_gpu_memory': np.mean([s['used_gb'] for s in self.gpu_memory_snapshots]) if self.gpu_memory_snapshots else 0
        }

# Example usage and configuration helpers

def get_optimal_thread_config(gpu_memory_gb, model_size="base"):
    """
    Get optimal threading configuration based on available GPU memory.
    
    Args:
        gpu_memory_gb: Available GPU memory in GB
        model_size: Size of the model ("small", "base", "large")
    
    Returns:
        Dictionary with optimal configuration
    """
    config = {
        "max_gpu_threads": 2,
        "memory_efficient": True,
        "batch_size_multiplier": 1.0
    }
    
    # Adjust based on GPU memory
    if gpu_memory_gb >= 40:  # High-end GPU
        config["max_gpu_threads"] = 4 if model_size == "small" else 3
        config["memory_efficient"] = False
        config["batch_size_multiplier"] = 2.0
    elif gpu_memory_gb >= 24:  # Mid-range GPU
        config["max_gpu_threads"] = 3 if model_size == "small" else 2
        config["memory_efficient"] = False
        config["batch_size_multiplier"] = 1.5
    elif gpu_memory_gb >= 16:  # Standard GPU
        config["max_gpu_threads"] = 2
        config["memory_efficient"] = True
        config["batch_size_multiplier"] = 1.0
    else:  # Low memory GPU
        config["max_gpu_threads"] = 1
        config["memory_efficient"] = True
        config["batch_size_multiplier"] = 0.5
    
    return config

def benchmark_threading_performance(sample_data, cross_encoder_model, thread_counts=[1, 2, 3, 4]):
    """
    Benchmark different thread configurations to find optimal settings.
    
    Args:
        sample_data: Sample data for benchmarking
        cross_encoder_model: CrossEncoder model
        thread_counts: List of thread counts to test
    
    Returns:
        Dictionary with benchmark results
    """
    results = {}
    
    for thread_count in thread_counts:
        print(f"Benchmarking with {thread_count} threads...")
        
        start_time = time.time()
        monitor = ThreadingMonitor()
        monitor.start_monitoring()
        
        try:
            # Run a sample processing task
            gpu_lock = threading.Lock()
            
            # Create sample table pairs
            table_pairs = [(0, 1, "table1", "table2")] * 10  # Sample pairs
            table_columns = {"table1": ["col1", "col2"], "table2": ["col3", "col4"]}
            
            # Process with current thread count
            with ThreadPoolExecutor(max_workers=thread_count) as executor:
                futures = []
                for i in range(thread_count):
                    future = executor.submit(
                        process_table_pair_batch, 
                        table_pairs[i::thread_count], 
                        table_columns, 
                        cross_encoder_model, 
                        gpu_lock
                    )
                    futures.append(future)
                
                for future in as_completed(futures):
                    thread_time = time.time() - start_time
                    monitor.record_thread_completion(thread_time)
                    monitor.record_gpu_memory()
            
            total_time = time.time() - start_time
            stats = monitor.get_statistics()
            
            results[thread_count] = {
                'total_time': total_time,
                'threading_stats': stats,
                'throughput': len(table_pairs) / total_time
            }
            
        except Exception as e:
            print(f"Benchmark failed for {thread_count} threads: {e}")
            results[thread_count] = {'error': str(e)}
    
    return results

# overall_metrics, all_metrics = run_optimized_single_gpu_discovery(
#     questions_data, retriever, faiss_store, lc_embed_model,
#     rerank, table_repository, discovery, top_k, delta
# )

# overall_metrics, all_metrics = run_optimized_single_gpu_discovery(
#     questions_data, retriever, faiss_store, lc_embed_model, 
#     rerank, table_repository, top_k, delta
# )


results_file = f"results_{top_k}_{delta}_dense_retriever.json"

overall_metrics, all_metrics = run_optimized_single_gpu_discovery(
    questions_data, retriever, faiss_store, lc_embed_model,
    rerank, table_repository, top_k, delta, results_file=results_file
)
# overall_metrics, all_metrics = run_optimized_single_gpu_discovery(
#     questions_data, retriever, None, lc_embed_model,
#     None, None, top_k, delta
# )
