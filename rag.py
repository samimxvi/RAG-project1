from pathlib import Path

import numpy as np
import ollama
from sentence_transformers import SentenceTransformer

DOCS_DIR = Path("docs")
EMBED_MODEL = "all-MiniLM-L6-v2"   # small, fast, runs on CPU
LLM_MODEL = "llama3.2"             # any model you've pulled with Ollama
CHUNK_WORDS = 150                  # words per chunk
OVERLAP_WORDS = 30                 # words shared between neighbouring chunks
TOP_K = 3                          # how many chunks to give the LLM


# 1. Load 
def load_documents(folder):
    docs = []
    for path in sorted(folder.glob("**/*")):
        if path.suffix.lower() in {".txt", ".md"}:
            docs.append((path.name, path.read_text(encoding="utf-8")))
    return docs


# 2. Chunk 
def chunk_text(text, size=CHUNK_WORDS, overlap=OVERLAP_WORDS):
    words = text.split()
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        piece = words[start:start + size]
        if piece:
            chunks.append(" ".join(piece))
        if start + size >= len(words):
            break
    return chunks


# 3. Embed + store
def build_index(docs, embedder):
    chunks = []  # list of (source_name, chunk_text)
    for name, text in docs:
        for c in chunk_text(text):
            chunks.append((name, c))
    vectors = embedder.encode(
        [c for _, c in chunks], normalize_embeddings=True, show_progress_bar=False
    )
    return chunks, np.asarray(vectors)


# 4. Retrieve
def retrieve(question, chunks, vectors, embedder, k=TOP_K):
    q = embedder.encode([question], normalize_embeddings=True)[0]
    scores = vectors @ q  # cosine similarity (vectors are normalised)
    top = np.argsort(scores)[::-1][:k]
    return [(chunks[i][0], chunks[i][1], float(scores[i])) for i in top]


# 5. Generate 
def answer(question, hits):
    context = "\n\n".join(f"[{src}]\n{text}" for src, text, _ in hits)
    prompt = (
        "Answer the question using ONLY the context below. "
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    stream = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    for part in stream:
        print(part["message"]["content"], end="", flush=True)
    print()


def main():
    docs = load_documents(DOCS_DIR)
    if not docs:
        raise SystemExit(f"No .txt or .md files found in {DOCS_DIR}/")

    print("Loading embedding model and indexing documents...")
    embedder = SentenceTransformer(EMBED_MODEL)
    chunks, vectors = build_index(docs, embedder)
    print(f"Indexed {len(chunks)} chunks from {len(docs)} file(s). Ask away (Ctrl+C to quit).\n")

    while True:
        try:
            question = input("Q: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not question:
            continue
        hits = retrieve(question, chunks, vectors, embedder)
        print("\nRetrieved from:", ", ".join(f"{s} ({sc:.2f})" for s, _, sc in hits))
        print("A: ", end="")
        answer(question, hits)
        print()


if __name__ == "__main__":
    main()