# ARM_Ex

This repository contains utilities for building a FAISS-based index from a JSON database.

Use `pipeline.py` to create the index:

```bash
python pipeline.py path/to/database.json
```

The underlying document creation logic lives in `arm_ex/documents.py`.
