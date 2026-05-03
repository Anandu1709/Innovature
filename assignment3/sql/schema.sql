-- Requires PostgreSQL 14+ and pgvector extension installed on server.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id SERIAL PRIMARY KEY,
    doc_name TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    embedding vector(384) NOT NULL,
    UNIQUE (doc_name, chunk_index)
);
