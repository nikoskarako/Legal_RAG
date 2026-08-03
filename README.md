# Greek Legal RAG

A Retrieval-Augmented Generation system for **Greek legislation**, built for a diploma
thesis at the National Technical University of Athens.

The project does two things: it answers legal questions in Greek over a corpus of
greek legal documents, and it builds the evaluation framework
needed to measure how well that actually works — an LLM-generated, human-reviewed
question bank, four retrieval systems, and RAGAS scoring across all of them.

---

## Results

118 human-accepted Greek legal questions, scored with [RAGAS](https://docs.ragas.io)
(judge: `gpt-4.1-mini`). All four systems answer with the same LLM
(`deepseek-chat-v3-0324`, temperature 0) and differ **only** in how context is retrieved.

| System | Retrieval | Faithfulness | Answer Relevancy | Context Relevance |
|---|---|---|---|---|
| Baseline | none (closed-book) | 0.116 | 0.800 | n/a |
| Dense | pgvector, `multilingual-e5-base` | 0.659 | 0.711 | 0.816 |
| BM25 | PostgreSQL `tsvector`, Greek config | **0.805** | 0.605 | 0.691 |
| **Hybrid** | Reciprocal Rank Fusion of both | 0.799 | **0.875** | **0.932** |

<sub>Context Relevance does not apply to the closed-book baseline, which retrieves nothing.</sub>

Reproduce the table from the committed scores — no API keys or database needed:

```bash
python src/analysis/stats.py     # means, medians, paired Wilcoxon tests
python src/analysis/oracle.py --hybrid results/ragas_scores_hybrid.json
```

**What the numbers say.** Retrieval is what makes the system trustworthy: the
closed-book baseline scores 0.116 on faithfulness, meaning it answers fluently
(0.800 answer relevancy) while being almost entirely ungrounded — confident
Greek legal prose with no basis in any actual statute. Retrieval raises
faithfulness by 5-7x.

Dense and lexical retrieval fail on *different* questions. Their per-question
faithfulness correlates at only r=0.18; BM25 wins 42% of questions, dense wins
27%, and an oracle that picked the better system per question would reach 0.885
faithfulness. Greek legal queries often hinge on exact statutory terms and
numbers, which lexical matching handles well and embeddings blur. Fusing the two
captures most of that complementarity: hybrid retrieval matches BM25's
faithfulness (p=0.59, i.e. no significant difference) while significantly
improving answer relevancy (p<1e-6) and reaching the best context relevance of
any system.

---

## How the pipeline fits together

Four stages. Stages communicate through PostgreSQL tables and JSON files on disk
rather than direct function calls, so the arrows below are the real dependencies.

```
   harvester_full.db  ── scraped ΦΕΚ / Government Gazette (SQLite, 859 MB, not in repo)
          │
          ▼
┌─ 1. INDEXING ─────────────────────────────────────────────────────────────┐
│  src/indexing/index_builder_512.py    chunk 512 → laws_vector_table_v2_512        │
│  embeddings: intfloat/multilingual-e5-base (768-dim) → PostgreSQL + pgvector      │
└──────────────────────────────────────────────────────────────────────────┘
          │                                    │
          ▼                                    ▼
┌─ 2. QUESTION BANK ──────────────┐  ┌─ 3. ANSWERING ─────────────────────┐
│ clustering.py                   │  │ rag/chat_openrouter.py             │
│   Louvain over cosine k-NN      │  │   dense retrieval + Greek prompt   │
│   → data/clusters/              │  │   ├─ rag/interface.py  (Tkinter)   │
│                                 │  │   └─ api/main.py       (FastAPI)   │
│ clustering_qa_harvester_*.py    │  │                                    │
│   representative doc per cluster│  │ rag/chat_cli_ragas_builder.py      │
│   2-stage gpt-4.1-mini:         │  │   + BGE cross-encoder reranking    │
│     questions_prompt → 3 Qs     │  │                                    │
│     answers_prompt   → A + meta │  │ retrieval/run_bm25_eval.py         │
│   Pydantic-validated            │  │   lexical tsvector retrieval       │
│                                 │  │                                    │
│ review_sheet.py                 │  │ retrieval/run_hybrid_eval.py       │
│   → Google Sheets, human        │  │   RRF fusion of dense + BM25       │
│     accept/reject               │  │                                    │
│   → data/qa_pairs/qa_review.json│  │ evaluation/generate_baseline_*.py  │
│     253 questions, 118 accepted │  │   closed-book, no retrieval        │
└─────────────────────────────────┘  └────────────────────────────────────┘
          │                                    │
          └──────────────┬─────────────────────┘
                         ▼
┌─ 4. EVALUATION ───────────────────────────────────────────────────────────┐
│  each system writes data/datasets/<System>_Dataset/{id}.json               │
│      {id, question, answer, context: {"1": ...}, doc_ids}                  │
│                                                                           │
│  evaluation/evaluate_ragas.py  → results/ragas_scores_<system>.json        │
│      faithfulness · answer relevancy · context relevance                   │
│                                                                           │
│  analysis/stats.py          means, medians, paired Wilcoxon                │
│  analysis/oracle.py         best-of-both ceiling, complementarity          │
│  analysis/question_type.py  breakdown by type and difficulty               │
│  analysis/error_analysis.py failure categories                            │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Repository layout

```
src/
  paths.py            shared project paths; also lets a script in one stage
                      import a module from another (see "A note on imports")
  indexing/           SQLite → embeddings → PostgreSQL + pgvector
  qa_generation/      clustering, 2-stage LLM Q-A generation, review upload
  rag/                dense retrieval + answering; Tkinter desktop client
  api/                FastAPI service wrapping the RAG pipeline
  retrieval/          BM25 and hybrid-RRF retrieval experiments
  evaluation/         dataset builders and the RAGAS scorer
  analysis/           statistics, oracle bound, error analysis
data/
  datasets/           118 scored questions per system (dense/bm25/hybrid/baseline)
  qa_pairs/           generated pairs and the human-reviewed question bank
  clusters/           36 Louvain clusters over the corpus
results/              RAGAS scores per system
docs/                 database setup, Q-A generation process, the LLM prompts used
```

### Per-file map

**Indexing**
| File | Role |
|---|---|
| `index_builder_512.py` | Chunks at 512 tokens → `laws_vector_table_v2_512` + `storage_512/`; used by the paper's experiments |

**Question bank**
| File | Role |
|---|---|
| `clustering.py` | Louvain communities over the embedding k-NN graph → `data/clusters/` |
| `clustering_qa_harvester_openai_10.py` | Picks a representative per cluster, generates 3 Q-A pairs with metadata |
| `QA_Generation_Manual.py` | Same generation for a fixed document list, skipping clustering |
| `review_sheet.py` | Pushes pairs to Google Sheets for manual accept/reject |

**Answering**
| File | Role |
|---|---|
| `chat_openrouter.py` | Core dense RAG: retrieve → Greek prompt → OpenRouter; resolves law titles and full text for citations |
| `interface.py` | Tkinter desktop client with source citations and law-text popups |
| `main.py` | FastAPI service — `GET /health`, `POST /query` |
| `chat_cli_ragas_builder.py` | CLI variant adding BGE cross-encoder reranking and dynamic chunk selection |

**Retrieval experiments**
| File | Role |
|---|---|
| `run_bm25_eval.py` | Lexical retrieval via PostgreSQL `ts_rank` with the Greek text-search config |
| `run_hybrid_eval.py` | Reciprocal Rank Fusion over dense + BM25 candidate lists (k=60) |
| `test_bm25.py` | Smoke test for the tsvector index |

**Evaluation and analysis**
| File | Role |
|---|---|
| `batch_ragas_export.py` | Runs accepted questions through the dense pipeline → `RAGAs_Dataset/` |
| `generate_baseline_dataset.py` | Closed-book answers, no retrieval → `RAGAS_baseline_dataset/` |
| `evaluate_ragas.py` | RAGAS scoring for any dataset folder → `results/` |
| `oracle.py` | Best-of-both upper bound, win split, complementarity correlation |
| `stats.py` | Means, medians and paired Wilcoxon tests between systems |
| `question_type.py` | Scores broken down by question type and difficulty |
| `error_analysis.py` | Failure categories across the three retrieval systems |
| `merge_hybrid_item.py` | Folds a re-scored item into an existing score file |

---

## The question bank

Two files describe the same 253 reviewed questions, and they agree exactly (all
253 question texts, types, difficulties and source documents match):

- `data/qa_pairs/Questions_for_Evaluation.csv` — what the retrieval and analysis
  scripts read. A question's **1-based row number is its id**, and that id is the
  filename in every per-system dataset.
- `data/qa_pairs/qa_review.json` — the same bank as JSON, as exported from the
  manual review.

The CSV is comma-delimited. This matters more than it sounds: the Greek question
mark is the semicolon character, so *every* question ends in `;` and a
semicolon-delimited file survives only through careful quoting.

---

## Stack

LlamaIndex · PostgreSQL + pgvector · `intfloat/multilingual-e5-base` ·
BGE cross-encoder reranker · OpenRouter (`deepseek-chat-v3-0324`) ·
OpenAI (`gpt-4.1-mini`, for Q-A generation and RAGAS judging) · RAGAS ·
python-louvain · FastAPI · Tkinter

## Licence

The code in this repository is released under the [MIT Licence](LICENSE).

The Greek legal texts it processes are public Greek government documents
and are not covered by that licence. The generated question bank and the
evaluation datasets under `data/` are derived from those public documents by
LLMs and reviewed by hand; they are shared for reproducibility.
