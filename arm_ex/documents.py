import json
from typing import List, Any, Dict
from llama_index.core import Document


def deepjoin_column_transform(
    column_name: str,
    column_values: List[Any],
    table_title: str,
    parent_table_desc: str,
) -> str:
    """Summarize a column's values for joinability checks."""
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

    return (
        f"{table_title}.{column_name} contains {n} values "
        f"({max_len}, {min_len}, {avg_len}): {col_snippet}"
    )


def create_llama_index_documents(database_path: str) -> List[Document]:
    """Convert a database JSON file into LlamaIndex documents."""
    with open(database_path, "r") as f:
        database = json.load(f)

    documents = []
    for table_name, table_variants in database["tables"].items():
        for variant_idx, table_data in enumerate(table_variants):
            table_id = table_data.get("id", f"unknown_{table_name}_{variant_idx}")
            columns = table_data.get("columns", [])
            description = table_data.get("table_description", "")

            content = table_data.get("content", table_data.get("data", []))

            schema_text = f"Table ID: {table_id}\n\n"
            schema_text += f"Table Name: {table_name}\n\n"
            schema_text += f"Columns: {', '.join(columns)}\n\n"
            schema_text += f"Description: {description}\n\n"

            if content:
                schema_text += "First 10 rows of data:\n"
                header_row = " | ".join(columns)
                separator = "-" * len(header_row)
                schema_text += f"{header_row}\n{separator}\n"
                for row in content[:10]:
                    row_values = [str(v) if v is not None else "NULL" for v in row]
                    row_values = row_values[: len(columns)]
                    while len(row_values) < len(columns):
                        row_values.append("NULL")
                    schema_text += " | ".join(row_values) + "\n"
            else:
                schema_text += "Data: No rows available for this table.\n"

            doc = Document(
                text=schema_text,
                metadata={
                    "table_name": table_name,
                    "table_id": table_id,
                    "variant_index": variant_idx,
                    "total_variants": len(table_variants),
                    "columns": columns,
                    "row_count": len(content),
                    "description": description,
                },
            )
            documents.append(doc)
    return documents


def create_documents_from_table_repo(json_path: str) -> List[Document]:
    """Create column-level documents from a table repository JSON."""
    documents = []
    with open(json_path, "r") as f:
        repo = json.load(f)

    for table_name, table_variants in repo.get("tables", {}).items():
        for variant_idx, table_info in enumerate(table_variants):
            table_id = table_info.get("id", "")
            columns = table_info.get("columns", [])
            data = table_info.get("data", table_info.get("content", []))
            table_description = table_info.get(
                "table_description", "No description available"
            )

            table_title = (
                f"{table_name} (Variant {variant_idx + 1}/{len(table_variants)})"
                if len(table_variants) > 1
                else table_name
            )

            for col_idx, col_name in enumerate(columns):
                column_values = [row[col_idx] for row in data if col_idx < len(row)]
                col_description = deepjoin_column_transform(
                    column_name=col_name,
                    column_values=column_values,
                    table_title=table_title,
                    parent_table_desc=table_description,
                )
                metadata = {
                    "table_id": table_id,
                    "table_name": table_name,
                    "column_name": col_name,
                    "variant_index": variant_idx,
                    "total_variants": len(table_variants),
                    "row_count": len(data),
                    "description": table_description,
                }
                documents.append(Document(text=col_description, metadata=metadata))
    return documents


def transform_retrieved_nodes(retrieved_nodes: List[Document]) -> Dict[str, Dict[str, Any]]:
    """Convert retrieved nodes into a simpler dictionary format."""
    result: Dict[str, Dict[str, Any]] = {}
    for node_with_score in retrieved_nodes:
        node = getattr(node_with_score, "node", node_with_score)
        table_name = node.metadata.get("table_name", "Unknown")
        table_id = node.metadata.get("table_id", "Unknown")
        columns = node.metadata.get("columns", [])

        table_title = f"{table_name}_({table_id})"
        text = node.text
        rows = []
        data_start = text.find("First 10 rows of data:")
        if data_start != -1:
            data_section = text[data_start:].split("\n")
            if len(data_section) > 3:
                for line in data_section[3:]:
                    if line.strip() and "|" in line:
                        rows.append([val.strip() for val in line.split("|")])

        column_descriptions = []
        for col_idx, col_name in enumerate(columns):
            column_values = [row[col_idx] for row in rows if col_idx < len(row)]
            description = deepjoin_column_transform(
                col_name,
                column_values,
                table_title,
                parent_table_desc=node.metadata.get("description", ""),
            )
            column_descriptions.append(description)

        result[table_id] = {
            "table_name": table_name,
            "column_descriptions": column_descriptions,
        }
    return result
