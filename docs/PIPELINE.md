# RAG Pipeline — Current Architecture (flowchart)

This diagram renders automatically on GitHub. It is the **current** configuration
(sentence-transformers embeddings · hybrid retrieval + RRF · extractive generation ·
math judge). Edit the Mermaid block below to keep it in sync as the pipeline evolves.

```mermaid
flowchart TD
  %% ---------------- Ingestion ----------------
  subgraph INGEST["① Ingestion & Processing"]
    A[PDF / DOCX transcripts<br/>20 interviews] --> B[Clean & normalize text]
    B --> C["Q&A-aware chunking<br/>split on 'Interviewer:'<br/>question + answer kept together"]
    C --> D["Extract metadata + topics<br/>doctor · country · specialty · topic"]
  end

  %% ---------------- Index ----------------
  subgraph INDEX["② Hybrid Index (persisted to disk)"]
    D --> E["Dense embeddings<br/>MiniLM · 384-dim"]
    D --> F["BM25 lexical index"]
    E --> G[(Hybrid Store<br/>dense.npy + chunks.json + BM25)]
    F --> G
  end

  %% ---------------- Retrieval ----------------
  Q([User question]) --> R
  subgraph RETRIEVE["③ Hybrid Retrieval Engine"]
    R["Score every chunk:<br/>dense cosine + BM25"] --> S["Reciprocal Rank Fusion (RRF)"]
    S --> X[Top-k chunks + scores]
  end
  G -.loads.-> R

  %% ---------------- Fallback + Response ----------------
  X --> GATE{"Fallback gate:<br/>best similarity ≥ threshold?"}
  GATE -->|no| FB["⚠️ Fallback answer + reason<br/>(anti-hallucination)"]
  GATE -->|yes| GEN
  subgraph RESPOND["④ Response Engine"]
    GEN["Generate answer<br/>extractive (default) · or LLM"] --> CITE["Attach line-level citations<br/>quoted line + literal source line no.<br/>+ view-in-transcript highlight"]
  end
  CITE --> ANS([Grounded answer + citations])
  FB --> ANS

  %% ---------------- Evaluation ----------------
  subgraph EVAL["⑤ Evaluation Pipeline (self-scoring) — 4 MECE sections"]
    GD[(Golden dataset<br/>18 Q&A + expected chunks/citations)] --> EV["Run engine on each question"]
    EV --> JUDGE["Judge: math (default) · or LLM-as-judge"]
    JUDGE --> M1["① Context Relevance <i>(hero)</i><br/>+ Context Recall · Context Precision"]
    JUDGE --> M2["② Groundedness <i>(hero)</i><br/>+ Citation Grounding"]
    JUDGE --> M3["③ Answer Relevance <i>(hero)</i><br/>+ Answer Correctness"]
    JUDGE --> M4["④ Ops<br/>Fallback Correctness · Latency"]
    M1 & M2 & M3 --> TRIAD["RAG Triad = mean of the 3 heroes"]
  end
  ANS -.evaluated by.-> EV

  classDef store fill:#eef4fb,stroke:#1d6fb8,color:#0d3b66;
  classDef gate fill:#fff7ed,stroke:#ea8c00,color:#7a3e00;
  class G,GD store;
  class GATE gate;
```

## Access points (where you trigger each stage)

| Access point | UI | CLI |
|---|---|---|
| ① Ingest & update knowledge base | **Ingest** tab → Build | `python build_index.py` |
| ② Answer a question | **Ask** tab | `python ask.py "..."` |
| ③ Evaluate over golden dataset | **Evaluate** tab → Run | `python run_eval.py` |
