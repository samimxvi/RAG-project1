# RAG Pipeline

**Retrieval-Augmented Generation (RAG)** pipeline that runs **fully locally**. No API keys, no cloud, no cost. Ask questions about your own `.txt` and `.md` files and get answers grounded in them.

Built with [Ollama](https://ollama.com) (local LLM) and [sentence-transformers](https://www.sbert.net) (local embeddings). The whole pipeline is one readable Python file.

## How it works

```
docs/*.txt, *.md
      │
      ▼
 1. Load      read your files
 2. Chunk     split into ~150-word overlapping pieces
 3. Embed     turn each chunk into a vector (all-MiniLM-L6-v2)
 4. Store     keep vectors in memory (NumPy array)
      │
Question ──► 5. Retrieve   find the top 3 most similar chunks (cosine similarity)
                   │
                   ▼
             6. Generate   send chunks + question to the local LLM (llama3.2)
                   │
                   ▼
                Answer (with the source files shown)
```

## Requirements

- Python 3.10 or newer
- [Ollama](https://ollama.com/download)
- About 3 GB of free disk space (LLM + embedding model + PyTorch)
- 8 GB RAM recommended

## Setup (Windows, Command Prompt)

**1. Install Ollama and pull a model**

```
ollama pull llama3.2
```

**2. Clone the repo and enter it**

```
git clone https://github.com/<your-username>/simple-local-rag.git
cd simple-local-rag
```

**3. Create a virtual environment and install dependencies**

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**4. Run**

```
python rag.py
```

The first run downloads the embedding model (about 90 MB), so it takes a little longer.

<details>
<summary>macOS / Linux</summary>

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python rag.py
```

</details>

## Usage

Put your `.txt` or `.md` files in the `docs/` folder and start the script. A sample handbook is included so you can try it straight away:

```
Indexed 4 chunks from 1 file(s). Ask away (Ctrl+C to quit).

Q: How many days of annual leave do I get?

Retrieved from: sample.txt (0.62), sample.txt (0.31), sample.txt (0.22)
A: Full-time employees receive 24 days of paid annual leave per year.
```

Each answer prints which files and similarity scores the context came from, so you can check whether retrieval worked.

## Configuration

Edit the constants at the top of `rag.py`:

| Setting | Default | What it does |
|---|---|---|
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | Embedding model (any sentence-transformers model) |
| `LLM_MODEL` | `llama3.2` | Any model you have pulled with Ollama |
| `CHUNK_WORDS` | `150` | Words per chunk |
| `OVERLAP_WORDS` | `30` | Words shared between neighbouring chunks |
| `TOP_K` | `3` | Number of chunks sent to the LLM |

## Troubleshooting

| Problem | Fix |
|---|---|
| `'ollama' is not recognized` | Close and reopen Command Prompt. If it persists, restart your PC. |
| `Connection refused` / Ollama errors | Make sure the Ollama app is running (system tray), or run `ollama serve` in another window. |
| `model 'llama3.2' not found` | Run `ollama pull llama3.2`. |
| `'python' is not recognized` | Reinstall Python with "Add Python to PATH" ticked, or try `py` instead of `python`. |
| Wrong or vague answers | Look at the retrieved sources first. Most problems come from retrieval. Try a different `CHUNK_WORDS` or `TOP_K`. |
| Slow answers | Use a smaller model, for example `llama3.2:1b`. |