# ARM_Ex

This repository contains utilities for building a FAISS-based index from a JSON database.

Use `pipeline.py` to create the index:

```bash
python pipeline.py path/to/database.json
```

The underlying document creation logic lives in `arm_ex/documents.py`.

`main_sql.py` can work with multiple language models. Choose a provider by
setting the `LLM_PROVIDER` environment variable to `deepseek`, `gemini`, or
`openai` (default `deepseek`). For each provider, supply the matching API key
via `DEEPSEEK_API_KEY`, `GOOGLE_API_KEY`, or `OPENAI_API_KEY`. If the key is not
found, the script will exit with an error.
