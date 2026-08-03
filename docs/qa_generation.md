# How the evaluation question bank was built

Evaluating a RAG system over Greek legislation needs questions whose answers are
genuinely grounded in specific statutes. There is no off-the-shelf Greek legal
QA benchmark, so the question bank was generated from the corpus itself and then
filtered by hand.

The pipeline is implemented in
[`src/qa_generation/clustering_qa_harvester_openai_10.py`](../src/qa_generation/clustering_qa_harvester_openai_10.py).

## 1. Cluster the corpus

Each law is stored in PostgreSQL both as full text and as chunk embeddings. The
chunk embeddings of a single law are averaged into one document-level vector, a
cosine k-NN graph is built over those vectors (k=10, edge weights = cosine
similarity), and Louvain community detection (resolution γ=1.0) is applied to the
graph's largest connected component. This produced the **35 thematic clusters**
the benchmark samples from, corresponding to major legal domains — civil,
criminal, administrative law and so on.

Clustering matters because sampling documents uniformly would over-represent
whatever topic happens to be most common in the Gazette. Sampling 2–3
representatives per cluster — those closest to the cluster centroid, subject to a
minimum length threshold that filters out trivial fragments — spreads the
questions across distinct areas of law. That gave **85 source documents**.

The community assignments exported by the clustering run are in
[`data/clusters/`](../data/clusters) (36 files, `cluster_0`–`cluster_35`).

## 2. Generate questions (LLM call 1)

The representative document's **full text** is sent to `DeepSeek Chat v3` with
[`prompts/questions_prompt.txt`](prompts/questions_prompt.txt), asking for three
questions of varying type and difficulty. Three questions across 85 documents
gives the **254 candidates** that go to review.

(The generator model is configurable — `clustering_qa_harvester_openai_10.py`
carries `gpt-4.1-mini` as its default, which is the *evaluation* judge, not the
model used to build the benchmark.)

## 3. Generate answers and metadata (LLM call 2)

A second call sends the same full text plus the three questions, with
[`prompts/answers_prompt.txt`](prompts/answers_prompt.txt). It returns, for each
question: the answer, the question type (factual / analytical / inferential), a
difficulty rating, the relevant excerpt of the law, and a short justification.
Responses are validated against a Pydantic schema before being written out.

Splitting generation into two calls keeps the model from writing questions it
has already decided how to answer, which tends to produce questions that are
trivially answerable from their own phrasing.

See [`prompts/qa_example_with_law_text.json`](prompts/qa_example_with_law_text.json)
for a complete worked example including the source law text.

## 4. Human review

Generated pairs were uploaded to a Google Sheet via
[`src/qa_generation/review_sheet.py`](../src/qa_generation/review_sheet.py) and
reviewed manually. The result is recorded twice, identically: as
[`Questions_for_Evaluation.csv`](../data/qa_pairs/Questions_for_Evaluation.csv)
(what the scripts read) and
[`qa_review.json`](../data/qa_pairs/qa_review.json).

A single reviewer applied a five-criterion accept/reject guide. A question is
accepted only if it is answerable from the source document alone, specific rather
than open-ended, and grounded in what the text explicitly states.

| | Count |
|---|---|
| Candidates reviewed | 254 |
| **Accepted** | **118** |
| Rejected | 136 |

An acceptance rate of 46.5% is worth noting: more than half of what the generator
produced was not good enough to evaluate against. The high rejection rate comes
mostly from the grounding criterion — many generated questions smuggle in
background knowledge or framing that the source document never states. The 118
accepted questions are what every system in the results table is scored on.

Distribution of the accepted set, and of the 253 rows present in the exported
files:

| | Exported rows (253) | | Accepted (118) |
|---|---|---|---|
| factual | 183 | | 96 |
| analytical | 42 | | 15 |
| inferential | 28 | | 7 |
| easy | 84 | | 46 |
| medium | 144 | | 52 |
| hard | 25 | | 20 |

<sub>The exported CSV and JSON carry 253 rows rather than 254 — one rejected
question did not make it into the export. Only rejected questions are affected;
the 118 accepted questions, which are the entire basis of the results, are
complete and identical in both files.</sub>

Review was also selective in a way that shifts the mix. Inferential questions
survived least often (7 of 28 accepted, 25%) while hard questions survived most
often (20 of 25, 80%) — so the accepted set is harder but less inferential than
what was generated. That matters when reading the per-type breakdown from
[`question_type.py`](../src/analysis/question_type.py): the inferential bucket
holds only 7 questions, so its scores move a lot per question.

A question's **1-based row number in the CSV is its id**, and that id is the
filename used for it in every per-system dataset under `data/datasets/`.

---

## Original description (Greek)

Περιγραφή της διαδικασίας του `clustering_qa_harvester.py`:

**1. Συλλογή δεδομένων**
Κάθε νόμος στη βάση υπάρχει και σαν πλήρες κείμενο και σαν embeddings. Τα embeddings
ενός ενιαίου/ολόκληρου νόμου συνδυάζονται σε έναν μέσο όρο για να έχουμε μία συνολική
αναπαράσταση. Παράλληλα, κρατάμε το πλήρες κείμενο του νόμου.

**2. Ομαδοποίηση και επιλογή εκπροσώπου**
Οι νόμοι ομαδοποιούνται σε clusters με βάση την ομοιότητά τους. Από κάθε cluster
επιλέγεται ένας representative νόμος, δηλαδή αυτός που είναι πιο κοντά στο κέντρο
της ομάδας.

**3. Παραγωγή ερωτήσεων (Στάδιο 1)**
Για τον representative νόμο, στέλνουμε στο LLM:
- το πλήρες κείμενο του νόμου,
- μαζί με `question_prompt`, συνοπτικά: «φτιάξε 3 ερωτήσεις πάνω στο κείμενο».

Το LLM επιστρέφει 3 ερωτήσεις διαφορετικής δυσκολίας και τύπου.

**4. Παραγωγή απαντήσεων (Στάδιο 2)**
Στη συνέχεια στέλνουμε στο LLM:
- το ίδιο πλήρες κείμενο,
- τις 3 ερωτήσεις που βγήκαν πριν,
- και `answer_prompt`, συνοπτικά: «δώσε απαντήσεις και metadata για κάθε ερώτηση».

Το LLM επιστρέφει τις απαντήσεις μαζί με πρόσθετες πληροφορίες, όπως το είδος και
τη δυσκολία κάθε ερώτησης, το σχετικό απόσπασμα του νόμου και μια σύντομη εξήγηση.

**5. Τελικό αποτέλεσμα**
Έτσι, για κάθε representative νόμο καταλήγουμε με 3 ζευγάρια ερώτησης–απάντησης,
εμπλουτισμένα με δομημένα metadata, τα οποία αποθηκεύονται σε JSON και μπορούν να
χρησιμοποιηθούν αργότερα για εκπαίδευση ή αξιολόγηση.
