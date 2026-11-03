# Agency RAG Project Plan (Ollama + Gemma)

A step-by-step plan to build a small, fully local RAG chatbot about an agency, and to understand how each part works.

## 1. Goal and success criteria

**Goal:** Ask questions about a fictional agency ("Bright Studio") and get answers grounded in your own documents, using only local tools.

**Done when:**

- You can ask 10 test questions and at least 8 get correct, source-cited answers.
- The bot says "I don't know" for questions that are not in the documents.
- You can explain, in your own words, what chunking, embedding, retrieval, and generation each do.

## 2. Architecture

**Ingestion (run once):** documents → split into chunks → embed each chunk → store in ChromaDB.

**Query (every question):** question → embed → find top-k similar chunks → put chunks in a prompt → Gemma (via Ollama) writes the answer.

| Part | Tool | Runs where |
| --- | --- | --- |
| LLM | Gemma 12B via Ollama | Local |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` | Local |
| Vector store | ChromaDB (persistent) | Local folder `db/` |
| Language | Python 3.10+ | Local |

## 3. Project structure

```
agency-rag/
├── docs/               # your knowledge base (.txt files)
├── db/                 # created by ingest.py
├── ingest.py           # build the index
├── ask.py              # chat loop
├── evaluate.py         # run the test questions
├── test_questions.json # questions with expected answers
└── notes.md            # your experiment log
```

## 4. Phase 1: Environment setup

- [ ] Install Ollama and confirm it runs: `ollama list`
- [ ] Note the exact Gemma tag from `ollama list` (for example `gemma4:12b`)
- [ ] Create the project and virtual environment:

```bash
mkdir agency-rag && cd agency-rag
python -m venv venv && source venv/bin/activate
pip install sentence-transformers chromadb ollama
mkdir docs
```

**Check:** `python -c "import ollama; print(ollama.list())"` prints your models without errors.

## 5. Phase 2: Build the knowledge base

Create 5 files in `docs/`. Give each file **different facts** so you can tell whether retrieval picked the right one.

| File | Contents |
| --- | --- |
| `services.txt` | SEO, branding, social media, web design: what each includes |
| `pricing.txt` | Three packages with monthly prices and what is included |
| `team.txt` | 6 people: name, role, specialty |
| `case_studies.txt` | 3 past clients with goals and measurable results |
| `faq.txt` | Onboarding steps, timelines, contract length, cancellation |

Aim for 300 to 800 words per file. Include a few **similar-sounding but different** facts (for example two packages with similar names) to make retrieval more interesting.

**Check:** you can answer 10 questions from the files by hand. Write them down now (see Phase 6).

## 6. Phase 3: Write the test questions first

Creating the test set before the code keeps you honest. Save as `test_questions.json`:

```json
[
  {"q": "How much does the social media package cost?", "expect": "EXACT PRICE FROM YOUR FILE"},
  {"q": "Who should I talk to about branding?", "expect": "NAME FROM YOUR FILE"},
  {"q": "What was the result for the first case study?", "expect": "RESULT FROM YOUR FILE"},
  {"q": "What is the cheapest package and who handles that work?", "expect": "needs pricing.txt and team.txt"},
  {"q": "Does the agency offer legal services?", "expect": "don't know"}
]
```

Extend it to 10 questions: 5 direct facts, 2 fuzzy wording, 2 multi-file, 1 or 2 traps (not in the documents).

## 7. Phase 4: Ingestion (`ingest.py`)

```python
import glob
import chromadb
from sentence_transformers import SentenceTransformer

CHUNK_SIZE = 500
OVERLAP = 100

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="db")
col = client.get_or_create_collection("agency")

def chunk(text, size=CHUNK_SIZE, overlap=OVERLAP):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks

for path in glob.glob("docs/*.txt"):
    text = open(path, encoding="utf-8").read()
    pieces = chunk(text)
    col.upsert(
        ids=[f"{path}-{i}" for i in range(len(pieces))],
        documents=pieces,
        embeddings=model.encode(pieces).tolist(),
        metadatas=[{"source": path}] * len(pieces),
    )
    print(f"Indexed {path}: {len(pieces)} chunks")
```

Run: `python ingest.py`

**Check:** each file reports a sensible chunk count (a 600-word file is roughly 8 to 10 chunks).

**Concept:** chunking exists because embedding models and prompts work best on small, focused pieces. Overlap stops a fact from being cut in half at a boundary.

## 8. Phase 5: Question answering (`ask.py`)

```python
import ollama
import chromadb
from sentence_transformers import SentenceTransformer

LLM_MODEL = "gemma4:12b"   # exact name from `ollama list`
TOP_K = 5

model = SentenceTransformer("all-MiniLM-L6-v2")
col = chromadb.PersistentClient(path="db").get_collection("agency")

def retrieve(question, k=TOP_K):
    result = col.query(query_embeddings=model.encode([question]).tolist(), n_results=k)
    return list(zip(result["documents"][0], result["metadatas"][0]))

def ask(question):
    hits = retrieve(question)
    context = "\n\n".join(f"[{m['source']}]\n{doc}" for doc, m in hits)

    response = ollama.chat(
        model=LLM_MODEL,
        options={"temperature": 0, "num_ctx": 8192},
        keep_alive="30m",
        messages=[
            {
                "role": "system",
                "content": (
                    "You answer questions about Bright Studio using ONLY the context provided. "
                    "If the answer isn't in the context, say you don't know. "
                    "Mention which source file you used."
                ),
            },
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )
    return response["message"]["content"], hits

if __name__ == "__main__":
    while True:
        q = input("\nAsk about the agency (or 'quit'): ")
        if q.lower() == "quit":
            break
        answer, hits = ask(q)
        print("\n--- Retrieved chunks ---")
        for doc, m in hits:
            print(f"[{m['source']}] {doc[:80]}...")
        print("\n--- Answer ---")
        print(answer)
```

Run: `python ask.py`

**Check:** the "Retrieved chunks" printout appears before each answer. Always read it first when debugging.

## 9. Phase 6: Evaluate (`evaluate.py`)

```python
import json
from ask import ask

tests = json.load(open("test_questions.json", encoding="utf-8"))
for i, t in enumerate(tests, 1):
    answer, hits = ask(t["q"])
    sources = sorted({m["source"] for _, m in hits})
    print(f"\n#{i} {t['q']}")
    print(f"Expected: {t['expect']}")
    print(f"Sources retrieved: {sources}")
    print(f"Answer: {answer}")
```

Read the output and mark each question correct or incorrect in `notes.md`. For every miss, decide which kind of failure it is:

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Right chunk is not in the retrieved list | Retrieval problem | Change chunk size, raise `k`, rephrase docs |
| Right chunk is there, answer is wrong | Generation problem | Tighten the prompt, lower context size |
| Answer ignores the context entirely | Context cut off | Raise `num_ctx` |
| Invents an answer to a trap question | Weak grounding | Strengthen the system prompt, add "quote the sentence you used" |

## 10. Phase 7: Experiments

Change **one variable at a time**, re-run `evaluate.py`, and log the score. For chunk-size changes, delete `db/` and re-run `ingest.py` first.

| # | Experiment | What you learn |
| --- | --- | --- |
| 1 | Chunk size 200 vs 500 vs 1000 | Precision versus context |
| 2 | `k` = 1, 3, 5, 8 | Recall versus noise |
| 3 | Overlap 0 vs 100 | Whether facts get split |
| 4 | Remove "ONLY the context" from the prompt | How much grounding the prompt provides |
| 5 | Swap the embedding model (`nomic-embed-text` via Ollama) | Embedding quality matters (re-ingest required) |
| 6 | Put conflicting prices in two files | How the model handles contradictions |
| 7 | `num_ctx` 2048 vs 8192 | Silent truncation |

**Log template for `notes.md`:**

```
Experiment: chunk size 200
Score: 6/10
What changed: ...
Why I think so: ...
```

## 11. Phase 8: Improvements (optional)

- Load PDFs or scraped web pages instead of `.txt` files
- Print similarity distances and filter out weak matches (answer "I don't know" when nothing is close)
- Add **hybrid search** (keyword plus vector) and **reranking**
- Add chat history so follow-up questions work
- Build a Streamlit or Gradio interface
- Split on paragraphs or headings instead of fixed character counts

## 12. Troubleshooting

- **`ConnectionError`:** Ollama is not running. Start it with `ollama serve`.
- **`model not found`:** `LLM_MODEL` must match `ollama list` exactly, including the tag.
- **Slow answers:** run `ollama ps`. If the PROCESSOR column shows a CPU/GPU split, lower `num_ctx` to 4096.
- **Odd reasoning text in answers:** print only `response["message"]["content"]` and check whether the model has a separate thinking field.
- **Empty or weird retrieval after changing models:** delete `db/` and re-run `ingest.py`. Vectors from different embedding models are not comparable.

## 13. Suggested schedule

| Session | Work |
| --- | --- |
| 1 | Phases 1 and 2: setup and write the documents |
| 2 | Phases 3 and 4: test questions and ingestion |
| 3 | Phases 5 and 6: chatbot and first evaluation |
| 4 | Phase 7: experiments and notes |
| 5 | Phase 8: pick one improvement |

## 14. What you should be able to explain at the end

1. Why we split documents into chunks, and what overlap does.
2. What an embedding is and why similar meanings end up close together.
3. The difference between a retrieval failure and a generation failure.
4. Why the system prompt and `num_ctx` affect answer quality.
5. Which setting (chunk size, `k`, model) changed your score the most, and why.
