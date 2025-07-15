import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))    

import numpy
import faiss


d = 768 # for sentence transformers, jina embeddings
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



def deepjoin_column_transform(column_name: str, column_values: List[Any], table_title: str, parent_table_desc: str) -> str:
    if not column_values:
        return f"{table_title}.{column_name} contains 0 values (0, 0, 0):"

    lengths = [len(str(val)) for val in column_values if val is not None]
    if not lengths:
        return f"{table_title}.{column_name} contains 0 values (0, 0, 0):"

    n = len(column_values)
    max_len = max(lengths)
    min_len = min(lengths)
    avg_len = round(sum(lengths) / n, 2)
    col_snippet = ", ".join([str(val) for val in column_values[:10]])   

    return f"{table_title}.{column_name} contains {n} values ({max_len}, {min_len}, {avg_len}): {col_snippet}"


import json

# Specify the path to your JSON file
file_path = "./MultiTableQA/himanshu/combined_database_with_desc.json"

# Open and load the JSON file
with open(file_path, "r") as file:
    table_repository = json.load(file)

# Print the loaded data
table_repository

from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    String,
    Integer,
    select,
    column,
)

from tqdm import tqdm

def get_full_data(table_repository):

    documents = []
    count = 0
     # Process the nested table structure
    for table_name, variants in table_repository["tables"].items():
        # Process each variant of this table
        for variant_idx, variant in enumerate(variants):
            table_id = variant.get("id")    
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
                
                for row in content:
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
            count+=1
            if count == 1:
                pbar = tqdm(total=sum(len(v) for v in table_repository["tables"].values()), desc="Processing tables")
            pbar.update(1)
            if count == pbar.total:
                pbar.close()

            documents.append(doc)

    return documents

docs = get_full_data(table_repository)
docs[:5]

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


dataframes = [document_to_dataframe(doc) for doc in docs]
dataframes[:2]

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

table_id_to_df = {df.attrs['table_id']: df for df in dataframes}
table_id_to_df

from sqlalchemy import Table, Column, String, Integer, Float, MetaData, insert
import pandas as pd


engine = create_engine("sqlite:///:memory:", future=True)
metadata_obj = MetaData()

from sqlalchemy import Table, Column, String, Integer, Float, MetaData, insert
import pandas as pd
from tqdm import tqdm

def infer_sqlalchemy_type(series):
    """Infer SQLAlchemy column type from pandas series"""
    if pd.api.types.is_integer_dtype(series):
        return Integer
    elif pd.api.types.is_float_dtype(series):
        return Float
    else:
        # For strings, we'll use a reasonable default length
        # You might want to adjust this based on your data
        max_length = series.astype(str).str.len().max() if len(series) > 0 else 50
        return String(max(max_length + 10, 50))  # Add some buffer

def create_table_from_dataframe(table_name, df, metadata_obj):
    """Create a SQLAlchemy table from a pandas DataFrame"""
    columns = []
    
    # Always add an auto-incrementing primary key column first
    columns.append(Column('row_id', Integer, primary_key=True, autoincrement=True))
    
    # Add all DataFrame columns as regular columns (no primary key constraints)
    for col_name in df.columns:
        col_type = infer_sqlalchemy_type(df[col_name])
        columns.append(Column(col_name, col_type))
    
    table = Table(table_name, metadata_obj, *columns)
    return table

def insert_dataframes_to_sql(data_dict, engine, metadata_obj):
    """Insert all DataFrames from the dictionary into SQL tables"""
    
    # Create all tables first
    tables = {}
    for table_id, df in data_dict.items():
        table = create_table_from_dataframe(table_id, df, metadata_obj)
        tables[table_id] = table
    
    # Create all tables in the database
    metadata_obj.create_all(engine)
    
    count = 0
    # Insert data into each table
    for table_id, df in data_dict.items():
        table = tables[table_id]
        
        # Convert DataFrame to list of dictionaries
        rows = df.to_dict('records')
        
        # Insert each row
        for row in rows:
            # Handle NaN values by converting them to None
            clean_row = {k: (None if pd.isna(v) else v) for k, v in row.items()}
            stmt = insert(table).values(**clean_row)
            with engine.begin() as connection:
                connection.execute(stmt)
        
        count+=1
        if count == 1:
            pbar = tqdm(total=len(data_dict), desc="Inserting tables")
        pbar.update(1)
        if count == pbar.total:
            pbar.close()


insert_dataframes_to_sql(table_id_to_df, engine, metadata_obj)

# Print all table names to verify
print("Created tables:", list(metadata_obj.tables.keys()))

lc_embed_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5"
)
# lc_embed_model = HuggingFaceEmbeddings(
#     model_name="Lajavaness/bilingual-embedding-large",
# )
# lc_embed_model = HuggingFaceEmbeddings(
#     model_name="jinaai/jina-embeddings-v2-base-en",
# )
embed_model = LangchainEmbedding(lc_embed_model)

Settings.embed_model = embed_model
Settings.chunk_size = 5500

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

import re
from difflib import SequenceMatcher
from typing import List, Tuple, Union, Any

def normalize_text(text: str) -> str:
    """Normalize text for comparison"""
    if text is None:
        return ""
    # Convert to lowercase, remove extra spaces, punctuation
    text = str(text).lower().strip()
    text = re.sub(r'[^\w\s]', ' ', text)  # Replace punctuation with space
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
    return text.strip()

def extract_values_from_sql_result(sql_result: List[Tuple]) -> List[str]:
    """Extract all values from SQL result tuples"""
    values = []
    for row in sql_result:
        for item in row:
            if item is not None:
                values.append(str(item))
    return values

def extract_values_from_ground_truth(gt_string: str) -> List[str]:
    """Extract values from ground truth string"""
    # Split by common separators
    values = re.split(r'[,;|]', gt_string)
    # Clean each value
    return [normalize_text(v) for v in values if normalize_text(v)]

def fuzzy_match_score(text1: str, text2: str) -> float:
    """Calculate fuzzy matching score between two texts"""
    norm1 = normalize_text(text1)
    norm2 = normalize_text(text2)
    return SequenceMatcher(None, norm1, norm2).ratio()

def number_match(val1: str, val2: str, tolerance: float = 0.01) -> bool:
    """Check if two values match as numbers within tolerance"""
    try:
        num1 = float(re.sub(r'[^\d.-]', '', val1))
        num2 = float(re.sub(r'[^\d.-]', '', val2))
        return abs(num1 - num2) <= tolerance * max(abs(num1), abs(num2))
    except (ValueError, ZeroDivisionError):
        return False

def answer_match_basic(sql_result: List[Tuple], ground_truth: str, 
                      fuzzy_threshold: float = 0.8) -> dict:
    """
    Basic answer matching - checks if any SQL result values match ground truth
    """
    sql_values = extract_values_from_sql_result(sql_result)
    if isinstance(ground_truth, dict):
        ground_truth = str(ground_truth.get('data', ''))
    gt_values = extract_values_from_ground_truth(ground_truth)
    
    matches = []
    for sql_val in sql_values:
        for gt_val in gt_values:
            # Exact match after normalization
            if normalize_text(sql_val) == normalize_text(gt_val):
                matches.append(('exact', sql_val, gt_val, 1.0))
            # Fuzzy match
            elif fuzzy_match_score(sql_val, gt_val) >= fuzzy_threshold:
                score = fuzzy_match_score(sql_val, gt_val)
                matches.append(('fuzzy', sql_val, gt_val, score))
            # Number match
            elif number_match(sql_val, gt_val):
                matches.append(('numeric', sql_val, gt_val, 1.0))
    
    return {
        'matches': matches,
        'has_match': len(matches) > 0,
        'best_score': max([m[3] for m in matches]) if matches else 0.0,
        'sql_values': sql_values,
        'gt_values': gt_values
    }

def answer_match_advanced(sql_result: Union[List[Tuple], str], ground_truth: str,
                         fuzzy_threshold: float = 0.7,
                         require_all_gt_values: bool = False) -> dict:
    """
    Advanced answer matching with better handling of multi-value answers
    """
    
    # Handle string input
    if isinstance(sql_result, str):
        # Parse string into list of values
        parsed_values = re.split(r'[,\n\r]+', sql_result.strip())
        parsed_values = [v.strip() for v in parsed_values if v.strip()]
        sql_result = [(val,) for val in parsed_values] if parsed_values else [("",)]
    
    if isinstance(ground_truth, dict):
        ground_truth = str(ground_truth.get('data', ''))

    sql_values = extract_values_from_sql_result(sql_result)
    gt_values = extract_values_from_ground_truth(ground_truth)


    
    # Create normalized versions
    norm_sql = [normalize_text(v) for v in sql_values]
    norm_gt = [normalize_text(v) for v in gt_values]
    
    matches = []
    matched_gt_indices = set()
    
    for i, sql_val in enumerate(sql_values):
        best_match = None
        best_score = 0
        
        for j, gt_val in enumerate(gt_values):
            if j in matched_gt_indices:
                continue
                
            # Try different matching strategies
            if normalize_text(sql_val) == normalize_text(gt_val):
                score = 1.0
                match_type = 'exact'
            elif number_match(sql_val, gt_val):
                score = 1.0
                match_type = 'numeric'
            else:
                score = fuzzy_match_score(sql_val, gt_val)
                match_type = 'fuzzy'
            
            if score > best_score and score >= fuzzy_threshold:
                best_match = (match_type, sql_val, gt_val, score, j)
                best_score = score
        
        if best_match:
            matches.append(best_match[:-1])  # Remove index
            matched_gt_indices.add(best_match[-1])
    
    # Calculate overall match quality
    if require_all_gt_values:
        has_match = len(matched_gt_indices) == len(gt_values)
    else:
        has_match = len(matches) > 0
    
    coverage = len(matched_gt_indices) / len(gt_values) if gt_values else 0
    
    return {
        'matches': matches,
        'has_match': has_match,
        'coverage': coverage,  # Fraction of ground truth values matched
        'best_score': max([m[3] for m in matches]) if matches else 0.0,
        'avg_score': sum([m[3] for m in matches]) / len(matches) if matches else 0.0,
        'sql_values': sql_values,
        'gt_values': gt_values,
        'unmatched_gt': [gt_values[i] for i in range(len(gt_values)) if i not in matched_gt_indices]
    }


import os
from datetime import datetime
# Set up comprehensive logging
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
    logger = logging.getLogger('sql_evaluation')
    logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # File handler for all logs (detailed)
    file_handler = logging.FileHandler(f'logs/sql_evaluation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    
    # File handler for results only
    results_handler = logging.FileHandler(f'logs/sql_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    results_handler.setLevel(logging.INFO)
    results_handler.setFormatter(simple_formatter)
    
    # Add filter to only log results
    class ResultsFilter(logging.Filter):
        def filter(self, record):
            return 'RESULT' in record.getMessage()
    
    results_handler.addFilter(ResultsFilter())
    logger.addHandler(results_handler)
    
    # Console handler for important messages only
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # Show INFO and above to console
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    # Progress handler
    progress_handler = logging.FileHandler(f'logs/sql_progress_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    progress_handler.setLevel(logging.INFO)
    progress_handler.setFormatter(simple_formatter)
    
    class ProgressFilter(logging.Filter):
        def filter(self, record):
            return 'PROGRESS' in record.getMessage()
    
    progress_handler.addFilter(ProgressFilter())
    logger.addHandler(progress_handler)
    
    return logger

import os
retrieved_tables_json_data = "./results_single_gpu_threaded_5_Splade_Hybrid.json"
if os.path.exists(retrieved_tables_json_data):
    with open(retrieved_tables_json_data, 'r') as f:
        retrieved_tables_json = json.load(f)

description_data_file = "./MultiTableQA/himanshu/combined_database_with_desc.json"
if os.path.exists(description_data_file):
    with open(description_data_file, 'r') as f:
        description_data = json.load(f)



def get_table_id_descriptions(json_data):
    """
    Extract a mapping of table IDs to their descriptions from the JSON structure.
    
    Args:
        json_data (dict): The loaded JSON data containing tables information
        
    Returns:
        dict: Mapping of table_id -> table_description
    """
    table_id_to_description = {}
    
    # Access the tables dictionary
    tables = json_data.get("tables", {})
    
    # Iterate through each table name and its variants
    for table_name, table_variants in tables.items():
        # Each table can have multiple variants (different table IDs)
        for variant in table_variants:
            table_id = variant.get("id")
            table_description = variant.get("table_description", "")
            
            if table_id and table_description:
                table_id_to_description[table_id] = table_description
    
    return table_id_to_description


def get_filtered_table_descriptions(json_data, table_ids):
    """
    Get table ID to description mapping for only the specified table IDs.
    
    Args:
        json_data (dict): The loaded JSON data containing tables information
        table_ids (list): List of table IDs to include
        
    Returns:
        dict: Mapping of table_id -> table_description for specified IDs only
    """
    # Get all descriptions first
    all_descriptions = get_table_id_descriptions(json_data)
    
    # Filter to only include requested table IDs
    filtered_descriptions = {
        table_id: description 
        for table_id, description in all_descriptions.items() 
        if table_id in table_ids
    }
    
    return filtered_descriptions

from llama_index.core.prompts import PromptTemplate

text_to_sql_prompt = PromptTemplate(
    """You are a SQL expert analyzing a multi-table database. Write a SQL query to answer the question.

ANALYSIS PROCESS:
1. Identify what information is needed to answer the question
2. Find which tables contain this information (ignore irrelevant tables)
3. Determine join relationships using common column names 
4. Write an efficient query using only the necessary tables

RULES:
- Only JOIN tables that are actually needed for the answer
- Use table aliases (T1, T2, etc.) for readability

If you cannot answer the question, return an empty result as the answer. And be concise and to the point
In your output just give the SQL query, nothing else. Don't include any explanation or additional text.

Schema: {schema}

Question: {query_str}

SQL Query:"""
)



questions_file = "./MultiTableQA/himanshu/merged_questions.json"
with open(questions_file, 'r') as f:
    questions_data = json.load(f)

# questions_data = questions_data[:300]  # Limit to first 200 questions for testing
# questions_data=questions_data[:20]
results_file=None

# from llama_index.llms.gemini import Gemini
# # current_key = keys.get()
# GOOGLE_API_KEY="AIzaSyCRq5vEOQNyzB0NqN9PfUESo3uCU-PCJZQ"

# genai.configure(api_key=GOOGLE_API_KEY)


# model = llm = Gemini(
#     model="models/gemini-2.0-flash",
#     api_key=GOOGLE_API_KEY,
#     max_tokens=1000000
# )
# Settings.llm = model


from llama_index.llms.deepseek import DeepSeek
from llama_index.llms.gemini import Gemini
from llama_index.llms.openai import OpenAI


def get_llm(provider: str = "deepseek", api_key: str | None = None):
    """Return an LLM instance for the given provider."""
    provider = provider.lower()

    if provider == "deepseek":
        if not api_key:
            api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError(
                "DeepSeek API key not provided. Set the `DEEPSEEK_API_KEY` environment variable or pass the key explicitly."
            )
        return DeepSeek(
            model="deepseek-ai/DeepSeek-V3-0324",
            api_key=api_key,
            api_base="https://api.kluster.ai/v1",
        )

    if provider == "gemini":
        if not api_key:
            api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Gemini API key not provided. Set the `GOOGLE_API_KEY` environment variable or pass the key explicitly."
            )
        genai.configure(api_key=api_key)
        return Gemini(model="models/gemini-2.0-flash", api_key=api_key, max_tokens=1_000_000)

    if provider == "openai":
        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OpenAI API key not provided. Set the `OPENAI_API_KEY` environment variable or pass the key explicitly."
            )
        return OpenAI(model="gpt-3.5-turbo", api_key=api_key)

    raise ValueError(f"Unsupported LLM provider: {provider}")


provider = os.environ.get("LLM_PROVIDER", "deepseek")
llm = get_llm(provider)

# You might also want to set deepseek as your default llm
# from llama_index.core import Settings
# Settings.llm = llm

# message = ChatMessage(role="user", content="Tell me a joke")
# resp = llm.chat([message])
# print(resp)


retrieved_tables = get_retrieved_tables(retrieved_tables_json, 1805)
retrieved_tables

table_desc = get_filtered_table_descriptions(description_data, retrieved_tables)
table_desc


from llama_index.core import SQLDatabase
from llama_index.core.query_engine import NLSQLTableQueryEngine


sql_database = SQLDatabase(engine, include_tables=retrieved_tables)
sql_query_engine = NLSQLTableQueryEngine(
    sql_database,
    text_to_sql_prompt=text_to_sql_prompt,
    llm = llm,
    context_query_kwargs = table_desc
)

sql_query_engine.sql_retriever._get_prompts()

answer = sql_query_engine.query("Which department currently headed by a temporary acting manager has the largest number of employees, and how many employees does it have?")
answer

final_answer = answer.source_nodes[0].node.metadata['result']
final_answer

import time
import logging
import os
from llama_index.core import SQLDatabase
from llama_index.core.query_engine import NLSQLTableQueryEngine
from sqlalchemy import text

# Initialize logging
logger = setup_logging()
logger.info("=== SQL EVALUATION STARTED ===")

accuracy = 0
total_questions = len(questions_data)
c = 0
top_k = 5
delta = 3
result_file = f"./results_ground_truth_{provider}.json"

# Check for existing results and load completed question IDs
completed_question_ids = set()
if os.path.exists(result_file):
    logger.info(f"Found existing results file: {result_file}")
    try:
        with open(result_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        result_data = json.loads(line)
                        if 'question_id' in result_data:
                            completed_question_ids.add(result_data['question_id'])
                            if 'has_match' in result_data and result_data.get('has_match') and result_data.get('coverage', 0) == 1.0:
                                accuracy += 1
                    except json.JSONDecodeError:
                        continue
        logger.info(f"Resuming from existing results. Found {len(completed_question_ids)} completed questions with {accuracy} correct answers")
    except Exception as e:
        logger.error(f"Error reading existing results file: {str(e)}")
        completed_question_ids = set()

logger.info(f"PROGRESS - Starting evaluation with {total_questions} questions")
logger.info(f"Results will be saved to: {result_file}")
logger.info(f"Skipping {len(completed_question_ids)} already completed questions")

for question in tqdm(questions_data):
    question_id = question['question_id']
    
    # Skip if question already completed
    if question_id in completed_question_ids:
        c += 1
        logger.debug(f"Skipping already completed question {question_id}")
        continue
    
    question_text = question['question']
    try:
        gt = question['answer']
    except KeyError:
        logger.error(f"Ground truth not found for question ID {question_id}")
        print(f"Ground truth not found for question ID {question_id}")
        continue
    
    logger.debug(f"Processing question {question_id}: {question_text}")
    logger.info(f"PROGRESS - Processing question {c+1}/{total_questions} (ID: {question_id})")
    
    # Get the table IDs from the question data
    retrieved_table_ids = [table['id'] for table in question['tables']]
    
    if not retrieved_table_ids:
        logger.warning(f"No retrieved tables for question ID {question_id}")
        print(f"No retrieved tables for question ID {question_id}")
        continue
    
    logger.debug(f"Retrieved tables for question {question_id}: {retrieved_table_ids}")
    table_desc = get_filtered_table_descriptions(description_data, retrieved_table_ids)
    sql_database = SQLDatabase(engine, include_tables=retrieved_table_ids)
    sql_query_engine = NLSQLTableQueryEngine(
            sql_database,
            text_to_sql_prompt=text_to_sql_prompt,
            llm = llm,
            context_query_kwargs=table_desc,
        )

    # Rate limit handling
    max_retries = 2
    retry_count = 0
    answer = None
    while answer is None and retry_count < max_retries:
        try:
            answer = sql_query_engine.query(question_text)
            logger.debug(f"Successfully got answer for question {question_id}")

        except Exception as e:
            if "max_tokens" in str(e).lower() or "400" in str(e).lower():
                logger.error(f"Max tokens exceeded for question {question_id}. Retrying with a new key...")
                print(f"Max tokens exceeded for question {question_id}. Retrying with a new key...")
                answer = None
                retry_count = max_retries
                continue
            logger.error(f"Error querying SQL for question {question_id}: {str(e)}")
            print(f"Error querying SQL for question {question_id}: {str(e)}")
            retry_count += 1
            if retry_count < max_retries:
                logger.info(f"Retrying ({retry_count}/{max_retries})...")
                time.sleep(60)
                continue
        
    if answer is None:
        logger.error(f"Failed to get answer for question ID {question_id}")
        print(f"Failed to get answer for question ID {question_id}")
        continue
    try:
        # Try to get the SQL query instead of the result
        if hasattr(answer, 'metadata') and 'sql_query' in answer.metadata:
            sql_query = answer.metadata['sql_query']
            # Execute the SQL directly with a timeout
            with engine.connect() as conn:
                result = conn.execute(text(sql_query)).fetchall()
                final_answer = str(result)
        else:
            # Fallback to string representation of the answer
            final_answer = str(answer)
        logger.info("got the final answer")
    except Exception as e:
        logger.error(f"Error extracting final answer for question {question_id}: {str(e)}")
        print(f"Error extracting final answer: {str(e)}")
        # Use the raw answer as fallback
        final_answer = str(answer)
    # try:
    #     result = answer_match_advanced(final_answer, gt)
    # except Exception as e:
    #     logger.error(f"Error matching answer for question {question_id}: {str(e)}")
    #     print(f"Error matching answer: {str(e)}")
    #     continue

    final_answer = str(answer)
    try:
        # Handle ground truth format
        if isinstance(gt, dict):
            gt_str = str(gt.get('data', ''))
        else:
            gt_str = str(gt)
        
        # Strict matching function
        def strict_match(answer_str: str, gt_str: str) -> dict:
            answer_norm = normalize_text(answer_str)
            # Split gt by common separators
            gt_vals = re.split(r'[,;|\n]', gt_str)
            gt_vals = [normalize_text(v) for v in gt_vals if normalize_text(v)]
            
            matches = [gt_val for gt_val in gt_vals if gt_val in answer_norm]
            all_matched = len(matches) == len(gt_vals)  # Require ALL values
            
            return {
                'has_match': all_matched,  # TRUE only if ALL gt values found
                'coverage': len(matches) / len(gt_vals) if gt_vals else 0,
                'matches': matches,
                'best_score': 1.0 if all_matched else 0.0
            }
        
        result = strict_match(final_answer, gt_str)
    except Exception as e:
        logger.error(f"Error normalizing ground truth for question {question_id}: {str(e)}")
        print(f"Error normalizing ground truth: {str(e)}")
        continue
    

    if result['has_match'] and result["coverage"] == 1.0:
        accuracy += 1
        logger.info(f"RESULT - Question ID: {question_id} - CORRECT")
        print(f"Question ID: {question_id} - Correct")
    else:
        logger.info(f"RESULT - Question ID: {question_id} - INCORRECT")
        logger.info(f"RESULT - SQL: {final_answer} | GT: {gt}")
        print(f"Question ID: {question_id} - Incorrect")
        print(f"SQL Result: {final_answer}")
        print(f"Ground Truth: {gt}")
        print(f"Matches: {result['matches']}")

    with open(result_file, 'a') as f:
        json.dump({
            "question_id": question_id,
            "question": question_text,
            "ground_truth": gt,
            "sql_result": final_answer,
            "matches": result['matches'],
            "coverage": result.get('coverage', 0.0),
            "has_match": result['has_match'],
            "best_score": result['best_score'],
        }, f)
        f.write("\n")

    c+=1

current_accuracy = accuracy/total_questions
logger.info(f"PROGRESS - Evaluation completed. Final accuracy: {accuracy}/{total_questions} = {current_accuracy:.2%}")
print(f"Accuracy: {accuracy}/{total_questions} = {current_accuracy:.2%}")

if result_file:
    with open(result_file, 'a') as f:
        json.dump({
            "accuracy": current_accuracy,
            "total_questions": total_questions,
        }, f, indent=4)

import json

def calculate_accuracy_from_file(file_path):
    """
    Calculate accuracy from a results file containing JSON lines
    
    Args:
        file_path (str): Path to the results file
        
    Returns:
        dict: Accuracy statistics
    """
    total_questions = 0
    correct_answers = 0
    results = []
    
    try:
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    result = json.loads(line)
                    
                    # Skip summary lines that don't have question_id
                    if 'question_id' not in result:
                        continue
                        
                    total_questions += 1
                    
                    # Check if answer is correct
                    has_match = result.get('has_match', False)
                    coverage = result.get('coverage', 0)
                    
                    # Consider correct if has_match is True and coverage is 1.0
                    if has_match>= 1.0:
                        correct_answers += 1
                        is_correct = True
                    else:
                        is_correct = False
                    
                    results.append({
                        'question_id': result['question_id'],
                        'is_correct': is_correct,
                        'has_match': has_match,
                        'coverage': coverage,
                        'best_score': result.get('best_score', 0)
                    })
                    
                except json.JSONDecodeError as e:
                    print(f"Error parsing line {line_num}: {e}")
                    continue
                    
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None
    
    accuracy = correct_answers / total_questions if total_questions > 0 else 0
    
    # Calculate additional statistics
    has_match_count = sum(1 for r in results if r['has_match'])
    avg_coverage = sum(r['coverage'] for r in results) / len(results) if results else 0
    avg_score = sum(r['best_score'] for r in results) / len(results) if results else 0
    
    return {
        'total_questions': total_questions,
        'correct_answers': correct_answers,
        'accuracy': accuracy,
        'accuracy_percent': accuracy * 100,
        'has_match_count': has_match_count,
        'has_match_rate': has_match_count / total_questions if total_questions > 0 else 0,
        'average_coverage': avg_coverage,
        'average_best_score': avg_score,
        'results': results
    }

def print_accuracy_report(stats):
    """Print a formatted accuracy report"""
    if stats is None:
        return
        
    print("=" * 50)
    print("ACCURACY REPORT")
    print("=" * 50)
    print(f"Total Questions: {stats['total_questions']}")
    print(f"Correct Answers: {stats['correct_answers']}")
    print(f"Accuracy: {stats['accuracy']:.4f} ({stats['accuracy_percent']:.2f}%)")
    print(f"Has Match Rate: {stats['has_match_rate']:.4f} ({stats['has_match_rate']*100:.2f}%)")
    print(f"Average Coverage: {stats['average_coverage']:.4f}")
    print(f"Average Best Score: {stats['average_best_score']:.4f}")
    print("=" * 50)


stats = calculate_accuracy_from_file("results_ground_truth.json")
stats
