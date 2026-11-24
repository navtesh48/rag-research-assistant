-- 384 = embedding size of bge-small-en-v1.5
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS docs (
  id SERIAL PRIMARY KEY, filename TEXT UNIQUE NOT NULL, n_pages INT,
  added_at TIMESTAMPTZ DEFAULT now());
CREATE TABLE IF NOT EXISTS chunks (
  id SERIAL PRIMARY KEY,
  doc_id INT REFERENCES docs(id) ON DELETE CASCADE,
  page INT NOT NULL, idx INT NOT NULL, text TEXT NOT NULL,
  tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
  embedding vector(384));
CREATE INDEX IF NOT EXISTS chunks_tsv ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_emb ON chunks USING hnsw (embedding vector_cosine_ops);
