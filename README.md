# RAG Research Assistant with Citations

Drop PDFs into a folder, ingest them, and ask questions. Every answer is a list of claims, each citing the **exact file and page** it came from. If the PDFs don't contain the answer, the app says so instead of guessing.

> **Provenance:** a clean reimplementation of a project I originally built; the original code was unavailable, so this was rebuilt from the original design. The unit tests below pass, but I have **not** yet run the full pipeline (Postgres, embeddings, reranker, Claude) end to end or measured retrieval quality, and the repo reports no accuracy numbers. The evaluation script is included so you can measure it on your own PDFs.

## How it works
```
PDFs → PyMuPDF (per page) → ~900-char chunks (150 overlap) → local bge-small embeddings → Postgres + pgvector
query → [pgvector cosine] + [Postgres full-text]  → Reciprocal Rank Fusion → local cross-encoder rerank
      → Claude with a forced JSON schema → claims with chunk ids → code drops any citation that was not retrieved
```

Design decisions:
- **Hybrid search**: vectors catch paraphrases, keyword search catches exact terms and numbers; RRF merges them without comparing raw scores.
- **Local embeddings and reranker**: the only paid API is answer generation.
- **Citation validation in code**: the model returns chunk ids; any id it wasn't shown is discarded, and claims left with no valid source are dropped.
- **Refuse before generating**: if the best rerank score is below a threshold, the app refuses without calling the model. Tune `REFUSE_BELOW` on your own answerable vs. unanswerable questions.

## Quick start
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                  # add ANTHROPIC_API_KEY
docker compose up -d                  # Postgres + pgvector

python -m rag.ingest data/pdfs        # put PDFs in data/pdfs first
python -m rag.cli "What pH range is optimal for Chlorella?"
# or: uvicorn rag.api:app --reload --app-dir src   →  http://localhost:8000/docs
```
Scanned PDFs have no text layer: run `ocrmypdf in.pdf out.pdf` first (ingestion prints a warning for near-empty pages).

## Evaluation
Write `eval/questions.json` from your own PDFs (about 35 answerable questions with gold file and page, and 15 that the PDFs can't answer), then:
```bash
python eval/run_eval.py
```
It reports recall@6 for vector-only, hybrid, and hybrid + rerank, plus correct and false refusals. Re-ingest with chunk sizes 500, 900 and 1400 and keep the numbers in this README.

## Tests
```bash
pytest        # chunking, rank fusion, citation validation (no database or model downloads needed)
```

## Roadmap
- Streamlit page showing the PDF page image next to each citation
- Filename filters, embedding cache
- Use as the literature tool for an algae-cultivation monitoring project
