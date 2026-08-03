# Database setup

Both retrievers read from a single PostgreSQL table, so the database has to
support two different kinds of search:

- **dense** — cosine similarity over embeddings, via `pgvector`
- **lexical (BM25)** — `ts_rank` over a `tsvector`, via PostgreSQL full-text search

## 1. Create the database

```sql
CREATE DATABASE legalchatbot2;
\c legalchatbot2
CREATE EXTENSION IF NOT EXISTS vector;
```

Then point `.env` at it (`PG_HOST`, `PG_PORT`, `PG_DATABASE`, `PG_USER`, `PG_PASSWORD`).

## 2. Build the vector index

```bash
python src/indexing/index_builder_512.py
```

This chunks the corpus at 512 tokens, embeds with `intfloat/multilingual-e5-base`
(768 dimensions) and writes to PostgreSQL.

A note on table naming that costs people time: LlamaIndex's `PGVectorStore` takes
a *base* table name and creates the physical table with a `data_` prefix. So
`VECTOR_TABLE=laws_vector_table_v2_512` becomes `public.data_laws_vector_table_v2_512`,
which is the name the raw SQL in the BM25 scripts refers to.

## 3. Add the full-text index for BM25

The lexical retriever needs a `tsvector` column and a GIN index on it. Run this
**after** indexing, since the index builder creates the table:

```sql
ALTER TABLE data_laws_vector_table_v2_512
  ADD COLUMN text_tsv tsvector
  GENERATED ALWAYS AS (to_tsvector('greek', text)) STORED;

CREATE INDEX idx_text_tsv
  ON data_laws_vector_table_v2_512 USING GIN (text_tsv);
```

`greek` is a stock PostgreSQL text-search configuration. Confirm it is available:

```sql
SELECT cfgname FROM pg_ts_config;
```

Check the result with the smoke test, which runs a few Greek queries straight
against the index:

```bash
python src/retrieval/test_bm25.py
```

## Restoring from a dump instead

If you have a `pg_dump` of an already-populated database, restoring it skips both
the indexing run and the step above:

```bash
pg_dump    -U "$PG_USER" -d legalchatbot2 -Fc -f legal_chatbot_dump.dump   # produce
pg_restore -U "$PG_USER" -d legalchatbot2 --no-owner --no-privileges \
           --clean --if-exists legal_chatbot_dump.dump                     # restore
```

The dump for this project is roughly 4 GB, so it is not distributed here.
