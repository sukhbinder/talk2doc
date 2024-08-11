# talk2doc
A tool to ask questions to documents in a set of PDFs

This works with ollama.

## Installation

To install, run:

```bash
pip install talk2doc
```
You may need to adjust the version of `langchain_community` based on the actual library requirement.

## Usage

Run:

```bash
talk2doc [model_name] [-p PDF_FILES] [-s CHUNK_SIZE] [-o CHUNK_OVERLAP] [-k TOP_K]
```

Replace `[model_name]` with the name of an LLM model (e.g., "mistral", "gemma").
Replace `PDF_FILES` with a list of paths to PDF files.

Optional arguments:
* `-s CHUNK_SIZE`: Chunk size. Default: 500.
* `-o CHUNK_OVERLAP`: Chunk overlap. Default: 50.
* `-k TOP_K`: Top K docs to return. Default: 6.

Example usage:

```bash
talk2doc mistral -p /path/to/pdfs.pdf -s 1000 -o 75 -k 10
```

This will use the "mistral" model, load PDF files from `/path/to/pdfs.pdf`, split each document into chunks of size 1000 with overlap 75, and return up to 10 top matches for each question.

