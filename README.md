# Conversational RAG over COVID-19 Doctor Interviews

An end-to-end, **single-turn (stateless) Conversational RAG** system that answers
questions over a corpus of **20 COVID-19 doctor interview transcripts** (US, India,
Germany) — with grounded citations, a fallback safety net, and a **self-running
evaluation pipeline** built on the **RAG Triad**.

Built to match the capstone spec's five components and three access points, and
to run **with zero API keys by default** (local embeddings + extractive answers),
upgrading to a hosted LLM (generation + LLM-as-judge) by setting one env var.

---

## 1. Quick start

```bat
:: Windows one-click (creates venv, installs deps, builds index, launches UI)
run_app.bat
```

Or manually:

```bat
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python build_index.py          :: Access Point 1 (ingest + index)
.venv\Scripts\python -m streamlit run app.py  :: the UI (all 3 access points)
```

Then open **http://localhost:8501**.

> First run downloads a small embedding model (~80 MB, `all-MiniLM-L6-v2`) once.

### Get a public `*.streamlit.app` URL (Streamlit Community Cloud)

1. Push this repo to GitHub — just run **`push_to_github.bat`** (handles login + push).
2. Go to **https://share.streamlit.io** → sign in with the same GitHub account.
3. **Create app** → pick repo `rag-covid-interviews`, branch `main`, main file **`app.py`**.
4. (Advanced settings → Python 3.11+.) Click **Deploy**. You'll get a URL like
   `https://<name>.streamlit.app`. First boot takes a few minutes (installs PyTorch +
   downloads the embedding model); after that it's instant.

The committed index loads automatically, so the app is usable immediately. If the
free tier runs short on memory, set `RAG_EMBEDDING_BACKEND=tfidf` in the app's
**Settings → Secrets** and rebuild from the Ingest tab — that drops the PyTorch
dependency entirely (lighter, faster boot, slightly less semantic nuance).

---

## 2. The five components (spec → implementation)

| # | Spec component | Where | What it does |
|---|---|---|---|
| 1 | Ingestion & Processing | `rag_app/ingestion.py` | Loads PDF/DOCX, cleans, **Q&A-aware chunking**, extracts metadata + topics |
| 2 | Hybrid Search & Retrieval | `rag_app/embeddings.py`, `vectorstore.py`, `retriever.py` | Dense embeddings + BM25, persisted index, **Reciprocal Rank Fusion** |
| 3 | Response Engine | `rag_app/response_engine.py` | Grounded answer + citations, or **fallback with a reason** |
| 4 | Golden Benchmark Dataset | `rag_app/golden_dataset.json` | 18 curated Q&A with **expected answer + expected chunks + expected citations** |
| 5 | Evaluation Pipeline | `rag_app/evaluation.py` | **RAG Triad** + retrieval / citation / latency / fallback KPIs |

### Three access points

| Access point | UI | CLI |
|---|---|---|
| ① Trigger ingestion & update knowledge base | **Ingest** tab → *Build/Rebuild* | `python build_index.py` |
| ② Answer a new user question | **Ask** tab | `python ask.py "..."` |
| ③ Evaluate over the golden dataset | **Evaluate** tab → *Run evaluation* | `python run_eval.py` |

---

## 3. Key design decisions (and why)

### a) Q&A-aware chunking — questions and answers stay together
The transcripts are clean turn-based dialogues (`Interviewer: … Dr. X: …`). Instead
of a fixed character window (which would slice an answer in half), we **split on
`Interviewer:` boundaries** so **each chunk = exactly one question + its full answer**.
Result: 20 documents → **299 self-contained Q&A chunks**, each tagged with doctor,
country, role, source file, turn index, and auto-detected topics.

### b) Hybrid retrieval (dense + BM25 → RRF) — the right fit for this data
- **Dense (semantic) embeddings** catch *paraphrased / conceptual* questions
  (“how did staff cope with stress?” → “moral distress”, “burnout”).
- **BM25 (lexical)** nails *exact clinical terms & acronyms* that embeddings blur
  (**MIS-C**, **SpO₂ 92%**, **mucormycosis**, **dexamethasone**, **Corona-Warn-App**).
- **Reciprocal Rank Fusion** merges the two *rankings* (not raw scores), so the
  differently-scaled signals combine robustly.

This is exactly why a single method is not enough here, and why "hybrid" is the
chosen retrieval method after reading the corpus.

### c) Grounding + Fallback — the two anti-hallucination levers
- **Grounding:** every answer quotes the exact supporting lines and attributes them
  to a doctor / country / file.
- **Fallback:** a relevance gate refuses to answer when nothing relevant is found,
  and **explains why** (e.g. off-topic, or out of scope). See limitations below.

### d) MMR + metadata filtering (retrieval refinements)
Two techniques from the courses, chosen *because they fit this data*:

- **Metadata filtering** — every chunk carries `country / doctor / specialty / topic`,
  so you can scope a query (e.g. *“what did **German** doctors say about vaccines”*).
  Clear precision win; exposed as filters in the Ask tab.
- **MMR (Maximal Marginal Relevance)** — re-ranks the hybrid candidate pool to drop
  **near-duplicate** chunks (e.g. three doctors giving the same vaccine-rollout
  answer at 0.72–0.75 similarity) so the k slots hold *distinct* evidence. Relevance
  in the MMR objective is the **normalised hybrid (fused) score** (keeps BM25 in play);
  diversity is dense cosine. `λ` (default 0.8) trades relevance vs diversity.

**Honest, measured finding:** on our golden benchmark MMR is roughly *neutral*
(MRR 0.839 → 0.844, recall unchanged) — because the hybrid top-5 is already mostly
distinct, and our gold evidence is intentionally topic-clustered, so aggressive
diversity (low λ) actually *hurts* recall. MMR’s value here is **situational
de-duplication** (fewer redundant chunks fed to the LLM, your exact intuition), not
a headline metric jump. **Auto-merging was rejected** for the same data reason — our
short, flat Q&A turns are already the right unit; merging would inject off-topic text.

> Earlier a naive MMR (dense-only relevance + unnormalised scores) *degraded* recall
> by 0.10 and MRR by 0.18 — a good reminder that a retrieval method must be matched to
> the data and the scores must be on a comparable scale.

---

## 4. Evaluation framework

The framework scores the three output stages the spec calls out (4a/4b/4c). Note
these are *areas* — the actual metrics designed for each are:

| Stage | Area | Metrics |
|---|---|---|
| Retrieve | **Quality of retrieval** | **Context Relevance** (RAG Triad), Hit Rate@k, MRR, Context Precision/Recall |
| Answer | **Answer relevance & quality** | **Answer Relevance** (RAG Triad), Answer Correctness (vs ideal) |
| Ground | **Citation accuracy** | **Groundedness / Faithfulness** (RAG Triad), Citation Grounding, Citation Source P/R |
| Ops | **Reliability** | Fallback Correctness, Latency (per stage) |

**The RAG Triad** = *Context Relevance · Groundedness · Answer Relevance* — the
framework from the *Building & Evaluating Advanced RAG* course.

Two judge backends, auto-selected:
- **`math`** (default, no key): embedding-cosine + lexical-overlap proxies for the
  triad; reference-based metrics (Hit Rate, MRR, P/R, citation checks) need no model.
- **`llm`** (if a provider key is set): LLM-as-judge feedback functions, mirroring
  the TruLens approach from the course.

### Latest benchmark (default keyless config: MiniLM + extractive + math judge)

| Metric | Score |
|---|---|
| **RAG Triad (mean)** | **0.63** |
| &nbsp;&nbsp;Context Relevance | 0.49 |
| &nbsp;&nbsp;Groundedness | 0.73 |
| &nbsp;&nbsp;Answer Relevance | 0.67 |
| Hit Rate @5 | **0.93** |
| MRR | 0.84 |
| Context Recall | 0.89 |
| Context Precision @5 | 0.27 \* |
| Citation Grounding | **1.00** |
| Fallback Correctness | 0.89 |
| Latency / question | ~190 ms |

\* **Context Precision is low by design, not by failure.** The golden set lists only
1–3 *ideal* source chunks per question, so precision@5 is capped near 0.2–0.4 even
when the other retrieved chunks are genuinely on-topic. **Hit Rate (0.93) and Recall
(0.89) are the meaningful retrieval signals here.**

---

## 5. Honest limitations (and how the LLM mode addresses them)

The keyless similarity gate cleanly rejects **off-topic** questions (e.g. a biryani
recipe scores 0.09; “capital of France” 0.13). It **cannot**, on similarity alone,
separate a *hard-but-answerable* question from one that is *on-topic but whose
specific fact is absent*:

- *“What is the exact dose of dexamethasone?”* — the drug is discussed, but **no dose
  is ever stated**. Similarity is high (0.44); the truthful answer is “not in the
  corpus”.
- *“COVID mortality rate in Brazil?”* — COVID-adjacent (0.45), but Brazil isn’t covered.

These two cases sit right next to the hardest answerable question (0.46), so any
single threshold either misses them or wrongly rejects real questions. **This is the
exact problem the RAG Triad’s LLM groundedness check is built for**: in `llm` mode the
generator inspects whether the *specific* answer is in the context and returns
`INSUFFICIENT_CONTEXT` if not. The keyless build catches the clearly-off-topic cases
(fallback correctness 0.89); enabling an LLM closes the rest.

---

## 6. Upgrading to a hosted LLM (optional)

```bat
set RAG_LLM_PROVIDER=openai          & rem or anthropic
set OPENAI_API_KEY=sk-...
set RAG_EMBEDDING_BACKEND=openai     & rem optional, defaults to local MiniLM
.venv\Scripts\python -m pip install openai          & rem or: anthropic
```

This switches **generation** to the LLM (answers strictly from context, with a hard
fallback instruction) and the **judge** to LLM-as-judge for the RAG Triad. Everything
else — chunking, hybrid retrieval, golden dataset, metrics — is unchanged.

All knobs live in `rag_app/config.py` (top-k, fusion constant, fallback thresholds,
model names) and are overridable via `RAG_*` environment variables.

---

## 7. Project layout

```
RAG_Capstone_Project/
├─ app.py                     UI — 3 access points as tabs (+ overview)
├─ build_index.py             Access Point 1 (CLI)
├─ ask.py                     Access Point 2 (CLI)
├─ run_eval.py                Access Point 3 (CLI)
├─ run_app.bat                One-click setup + launch (Windows)
├─ requirements.txt
├─ tests_smoke.py             Headless UI test (Streamlit AppTest)
├─ rag_app/
│  ├─ config.py               Central config + provider switch
│  ├─ ingestion.py            Load + Q&A-aware chunking + metadata/topics
│  ├─ embeddings.py           Pluggable embedder (ST / TF-IDF / OpenAI)
│  ├─ vectorstore.py          Persistent hybrid index (dense + BM25)
│  ├─ retriever.py            Hybrid retrieval + Reciprocal Rank Fusion
│  ├─ llm.py                  Optional provider-agnostic LLM client
│  ├─ response_engine.py      Grounded answers + citations + fallback
│  ├─ evaluation.py           RAG Triad + KPIs
│  └─ golden_dataset.json     18 curated benchmark items
├─ storage/                   Persisted index + cached eval results
└─ Interview Transcripts (PDFs, Docx)/   the source corpus
```
