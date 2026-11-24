"""Ingest PDFs: extract per page, chunk, embed locally, store in Postgres.

    python -m rag.ingest data/pdfs
"""
import sys, pathlib, fitz
from rag.db import init
from rag.chunking import chunk_text

_emb = None

def get_embedder():
    """Loaded lazily so importing this module (and running the unit tests) never downloads a model."""
    global _emb
    if _emb is None:
        from sentence_transformers import SentenceTransformer
        _emb = SentenceTransformer('BAAI/bge-small-en-v1.5')
    return _emb

def ingest_pdf(conn, path: pathlib.Path) -> int:
    pdf = fitz.open(path)
    row = conn.execute('INSERT INTO docs (filename, n_pages) VALUES (%s, %s) '
                       'ON CONFLICT (filename) DO NOTHING RETURNING id', (path.name, len(pdf))).fetchone()
    if row is None: return 0                        # already ingested
    doc_id, rows = row[0], []
    for pno, page in enumerate(pdf, start=1):
        text = page.get_text()
        if len(text.strip()) < 20:
            print(f'warning: {path.name} p.{pno} has almost no text (scanned?). Run ocrmypdf first.')
        for i, ch in enumerate(chunk_text(text)):
            rows.append((doc_id, pno, i, ch))
    vecs = get_embedder().encode([r[3] for r in rows], batch_size=64, normalize_embeddings=True)
    with conn.cursor() as cur:
        cur.executemany('INSERT INTO chunks (doc_id, page, idx, text, embedding) VALUES (%s,%s,%s,%s,%s)',
                        [(*r, v) for r, v in zip(rows, vecs)])
    return len(rows)

if __name__ == '__main__':
    conn = init()
    for p in sorted(pathlib.Path(sys.argv[1]).glob('*.pdf')):
        print(p.name, ingest_pdf(conn, p), 'chunks')
