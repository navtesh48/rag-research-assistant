"""Hybrid retrieval: pgvector cosine + Postgres full-text, merged with RRF, then cross-encoder reranking."""
from rag.fusion import rrf
from rag.ingest import get_embedder

QUERY_PREFIX = 'Represent this sentence for searching relevant passages: '   # bge query instruction
_rerank = None

def get_reranker():
    global _rerank
    if _rerank is None:
        from sentence_transformers import CrossEncoder
        _rerank = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    return _rerank

def retrieve(conn, query: str, n_each: int = 30, n_final: int = 6, mode: str = 'hybrid_rerank'):
    qv = get_embedder().encode(QUERY_PREFIX + query, normalize_embeddings=True)
    vec = [r[0] for r in conn.execute('SELECT id FROM chunks ORDER BY embedding <=> %s LIMIT %s', (qv, n_each))]
    kw = [r[0] for r in conn.execute(
        "SELECT id FROM chunks WHERE tsv @@ websearch_to_tsquery('english', %s) "
        "ORDER BY ts_rank_cd(tsv, websearch_to_tsquery('english', %s)) DESC LIMIT %s", (query, query, n_each))]
    if mode == 'vector': ids = vec
    elif mode == 'hybrid': ids = rrf([vec, kw])
    else: ids = rrf([vec, kw])[:n_each]
    rows = {r[0]: r for r in conn.execute(
        'SELECT c.id, d.filename, c.page, c.text FROM chunks c JOIN docs d ON d.id=c.doc_id WHERE c.id = ANY(%s)', (ids,))}
    cands = [rows[i] for i in ids if i in rows]
    if mode != 'hybrid_rerank':
        return [dict(id=r[0], filename=r[1], page=r[2], text=r[3], score=None) for r in cands[:n_final]]
    scores = get_reranker().predict([(query, r[3]) for r in cands])
    ranked = sorted(zip(cands, scores), key=lambda t: t[1], reverse=True)[:n_final]
    return [dict(id=r[0], filename=r[1], page=r[2], text=r[3], score=float(s)) for r, s in ranked]
