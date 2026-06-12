"""Conversational RAG over COVID-19 Doctor Interviews — Streamlit UI.

Three access points (matching the capstone spec) as tabs:
    ①  Ingest & Process   - trigger the ingestion pipeline, inspect the corpus
    ②  Ask a Question     - live grounded QA with citations + fallback
    ③  Evaluate           - run the golden benchmark, see RAG Triad + KPIs

Run:  streamlit run app.py
"""
from __future__ import annotations

import html as _html
import json
import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from rag_app import config
from rag_app.ingestion import ingest_corpus, corpus_stats
from rag_app.vectorstore import HybridIndex
from rag_app.retriever import HybridRetriever
from rag_app.response_engine import ResponseEngine
from rag_app.evaluation import load_golden, build_judge, run_evaluation

st.set_page_config(page_title="Conversational RAG · COVID Interviews",
                   page_icon="🩺", layout="wide")

# ----------------------------------------------------------------- styling --
st.markdown("""
<style>
  .block-container {padding-top: 1.6rem; max-width: 1300px;}
  .hero {background: linear-gradient(120deg,#0d3b66 0%,#1d6fb8 100%);
         color:#fff; padding:22px 28px; border-radius:14px; margin-bottom:14px;}
  .hero h1 {color:#fff; margin:0 0 4px 0; font-size:1.7rem;}
  .hero p  {color:#dbeafe; margin:0; font-size:.96rem;}
  .pill {display:inline-block; background:#eef4fb; color:#1d4e79; border-radius:999px;
         padding:3px 12px; margin:3px 4px 0 0; font-size:.8rem; font-weight:600;}
  .card {background:#fff; border:1px solid #e6e9ef; border-radius:12px;
         padding:16px 18px; box-shadow:0 1px 3px rgba(16,40,80,.05);}
  .metric-big {font-size:2.0rem; font-weight:700; line-height:1;}
  .metric-lbl {color:#5b6573; font-size:.8rem; text-transform:uppercase; letter-spacing:.04em;}
  .cite {background:#f7faff; border-left:4px solid #1d6fb8; border-radius:6px;
         padding:10px 14px; margin:8px 0; font-size:.92rem;}
  .cite .src {color:#0d3b66; font-weight:700; font-size:.82rem;}
  .fallback {background:#fff7ed; border-left:5px solid #ea8c00; border-radius:8px; padding:14px 18px;}
  .answer-box {background:#f3f8ff; border:1px solid #d6e6fb; border-radius:10px; padding:18px 20px; font-size:1.02rem;}
  .step {background:#f4f7fb;border:1px solid #dde6f1;border-radius:10px;padding:10px 12px;text-align:center;font-size:.84rem;}
  .step b{color:#0d3b66;}
  .flow-arrow{font-size:1.4rem;color:#9bb4cf;text-align:center;}
</style>
""", unsafe_allow_html=True)

GOOD, MID = 0.70, 0.50


def score_color(v: float) -> str:
    if v != v:  # NaN
        return "#9aa4b2"
    return "#1a9850" if v >= GOOD else ("#e08a00" if v >= MID else "#d6604d")


def gauge(value: float, title: str) -> go.Figure:
    v = 0 if value != value else value
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=round(v, 3),
        number={"font": {"size": 30}},
        title={"text": title, "font": {"size": 14}},
        gauge={"axis": {"range": [0, 1]},
               "bar": {"color": score_color(v)},
               "steps": [{"range": [0, MID], "color": "#fde6e0"},
                         {"range": [MID, GOOD], "color": "#fdf2da"},
                         {"range": [GOOD, 1], "color": "#e3f3e1"}]}))
    fig.update_layout(height=200, margin=dict(l=18, r=18, t=44, b=8))
    return fig


def metric_card(label: str, value, sub: str = "", color: str | None = None):
    val = value if isinstance(value, str) else (f"{value:.3f}" if value == value else "n/a")
    col = color or "#0d3b66"
    st.markdown(f"<div class='card'><div class='metric-lbl'>{label}</div>"
                f"<div class='metric-big' style='color:{col}'>{val}</div>"
                f"<div style='color:#7a8493;font-size:.8rem'>{sub}</div></div>",
                unsafe_allow_html=True)


# ------------------------------------------------- metric definitions ------
# Each entry: human name + plain-English definition + the EXACT formula we use
# (math-judge implementation in rag_app/evaluation.py) + value range + how it's
# computed. Surfaced via click-to-open popovers on every metric name.
METRIC_INFO = {
    "rag_triad": {
        "name": "RAG Triad (overall)",
        "definition": "Headline score = the mean of the three RAG-Triad hero metrics "
                      "(Context Relevance · Groundedness · Answer Relevance). The triad "
                      "is the evaluation framework from the *Building & Evaluating "
                      "Advanced RAG* course (TruLens).",
        "formula": r"\text{RAG Triad}=\frac{\text{Context Relevance}+\text{Groundedness}+\text{Answer Relevance}}{3}",
        "range": "0 – 1 (NaNs ignored in the mean).",
    },
    "context_relevance": {
        "name": "Context Relevance",
        "definition": "Average semantic similarity between the question and each of the "
                      "*k* retrieved chunks — are the chunks we pulled actually about what "
                      "was asked? A RAG-Triad metric.",
        "formula": r"\text{Context Relevance}=\frac{1}{k}\sum_{i=1}^{k}\max\!\big(0,\ \cos(\vec q,\ \vec c_i)\big)",
        "range": "0 – 1 (higher = more on-topic context).",
        "source": "Math judge: mean dense-cosine of retrieved chunks vs. query (MiniLM "
                  "embeddings). LLM mode: an LLM rates relevance 0–10.",
    },
    "groundedness": {
        "name": "Groundedness / Faithfulness",
        "definition": "For every sentence *a* in the answer, how close is it to the nearest "
                      "sentence *c* in the retrieved context *C*? Measures whether claims are "
                      "supported by the sources vs. made up. A RAG-Triad metric.",
        "formula": r"\text{Groundedness}=\frac{1}{|A|}\sum_{a\in A}\ \max_{c\in C}\ \cos(\vec a,\ \vec c)",
        "range": "0 – 1 (higher = better supported). NaN on fallback (nothing to ground).",
        "source": "Math judge: max cosine of each answer sentence to any context sentence, "
                  "averaged. LLM mode: an LLM checks every claim is supported.",
    },
    "answer_relevance": {
        "name": "Answer Relevance",
        "definition": "Semantic similarity between the question *q* and the answer *a* — does "
                      "the answer actually address what was asked? A RAG-Triad metric.",
        "formula": r"\text{Answer Relevance}=\max\!\big(0,\ \cos(\vec q,\ \vec a)\big)",
        "range": "0 – 1. NaN on fallback.",
        "source": "Math judge: cosine(question, answer). LLM mode: an LLM rates 0–10.",
    },
    "answer_correctness": {
        "name": "Answer Correctness",
        "definition": "Semantic similarity between the system answer and the **curated ideal "
                      "answer** from the golden dataset — is it actually correct, not just "
                      "on-topic?",
        "formula": r"\text{Answer Correctness}=\max\!\big(0,\ \cos(\vec a_{\text{ideal}},\ \vec a)\big)",
        "range": "0 – 1. NaN on fallback.",
        "source": "Math judge: cosine(ideal answer, system answer). LLM mode: LLM rates "
                  "meaning-match 0–10.",
    },
    "context_recall": {
        "name": "Context Recall @k",
        "definition": "Of the gold (expected) chunks for a question, what fraction appeared "
                      "in the top-*k* retrieved? Coverage of the right context.",
        "formula": r"\text{Context Recall@}k=\frac{|\,\text{retrieved}_{1:k}\,\cap\,\text{gold}\,|}{|\,\text{gold}\,|}",
        "range": "0 – 1 (1 = every gold chunk retrieved).",
        "source": "Reference-based (no model). gold = `expected_source_ids` in the golden dataset.",
    },
    "context_precision": {
        "name": "Context Precision @k",
        "definition": "Of the top-*k* chunks we retrieved, what fraction are gold? "
                      "Signal-vs-noise / purity of the context window.",
        "formula": r"\text{Context Precision@}k=\frac{|\,\text{retrieved}_{1:k}\,\cap\,\text{gold}\,|}{k}",
        "range": "0 – 1. **Capped low here**: each question lists only 1–3 gold chunks, so "
                 "precision@5 maxes near 0.2–0.4 even with perfect retrieval.",
        "source": "Reference-based (no model).",
    },
    "citation_grounding": {
        "name": "Citation Grounding",
        "definition": "Fraction of cited quotes that genuinely occur in the source chunk they "
                      "point to (≥ 80% of the quote's tokens found). Catches fabricated "
                      "citations. *T(·)* = set of normalized tokens.",
        "formula": r"\text{Citation Grounding}=\frac{1}{|\mathcal C|}\sum_{c\in\mathcal C}\mathbf{1}\!\left[\frac{|\,T(c.\text{quote})\cap T(\text{src})\,|}{|\,T(c.\text{quote})\,|}\ \ge\ 0.8\right]",
        "range": "0 – 1 (1 = every citation verified). NaN on fallback / no citations.",
        "source": "Reference-based token overlap (no model).",
    },
    "fallback_correct": {
        "name": "Fallback Correctness",
        "definition": "Fraction of questions where the engine abstained **exactly when it "
                      "should** — refusing the out-of-scope / unanswerable ones and answering "
                      "the answerable ones. The anti-hallucination guard.",
        "formula": r"\text{Fallback Correctness}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}\big[\,\text{abstained}_i=\text{should-abstain}_i\,\big]",
        "range": "0 – 1 (1 = perfect abstention behaviour).",
        "source": "should-abstain = (not answerable) from the golden dataset; abstained = "
                  "engine returned a fallback.",
    },
    "latency_ms_total": {
        "name": "Latency",
        "definition": "Average wall-clock time to retrieve + compose an answer, per question.",
        "formula": r"\text{Latency}=\frac{1}{N}\sum_{i=1}^{N} t_i\quad(\text{ms, end-to-end})",
        "range": "milliseconds (lower = faster).",
        "source": "Measured with a perf timer around the engine call.",
    },
}


def metric_info_popover(label: str, key: str, prefix: str = ""):
    """A click-to-open popover whose trigger IS the metric name; reveals the
    definition, exact formula, range and how we compute it."""
    info = METRIC_INFO.get(key, {})
    with st.popover(f"{prefix}{label}"):
        st.markdown(f"#### {info.get('name', label)}")
        if info.get("definition"):
            st.markdown(info["definition"])
        if info.get("formula"):
            st.latex(info["formula"])
        tail = []
        if info.get("range"):
            tail.append(f"**Range:** {info['range']}")
        if info.get("source"):
            tail.append(f"**How we compute it:** {info['source']}")
        if tail:
            st.caption("  \n".join(tail))


def side_metric(label: str, value, sub: str, key: str, color: str | None = None):
    """A supporting metric card: clickable name (popover) + big value + caption."""
    with st.container(border=True):
        metric_info_popover(label, key)
        val = value if isinstance(value, str) else (f"{value:.3f}" if value == value else "n/a")
        col = color or "#0d3b66"
        st.markdown(f"<div class='metric-big' style='color:{col}'>{val}</div>"
                    f"<div style='color:#7a8493;font-size:.78rem'>{sub}</div>",
                    unsafe_allow_html=True)


def eval_section(title: str, subtitle: str, hero_value: float, hero_label: str,
                 hero_key: str, cards: list[tuple], slots: int = 4):
    """One evaluation dimension: a big 'hero' gauge (a RAG-Triad metric, with a
    clickable name that pops up its definition + formula) + its supporting 'side'
    metrics (each name also click-to-define).

    `cards` is a list of (label, value, sub, metric_key) tuples. A numeric value
    (0–1) is colour-coded by score; a pre-formatted string (e.g. latency) renders
    neutral. Side metrics use `slots` fixed-width columns so card sizing stays
    consistent across sections regardless of how many a section has.
    """
    st.markdown(f"##### {title}")
    if subtitle:
        st.caption(subtitle)
    left, right = st.columns([1.1, 2.6])
    with left:
        st.markdown("<div style='text-align:center;color:#b8860b;font-weight:800;"
                    "font-size:.72rem;letter-spacing:.10em;text-transform:uppercase;'>"
                    "★ Hero metric — click name for definition</div>", unsafe_allow_html=True)
        metric_info_popover(hero_label, hero_key, prefix="📖 ")
        st.plotly_chart(gauge(hero_value, hero_label), width="stretch")
    with right:
        st.write("")
        cc = st.columns(max(slots, len(cards)))
        for col, (lbl, val, sub, key) in zip(cc, cards):
            with col:
                color = score_color(val) if isinstance(val, (int, float)) and not isinstance(val, bool) else None
                side_metric(lbl, val, sub, key, color)


# ------------------------------------------------- line-level source viewer --
def line_label(start, end) -> str:
    if not start:
        return "line n/a"
    return f"line {start}" if (not end or end == start) else f"lines {start}–{end}"


def _transcript_html(raw_text: str, start_line, end_line) -> str:
    """Full transcript with line-number gutter; the cited line range is
    highlighted and auto-scrolled into view (JS runs inside the iframe)."""
    lines = raw_text.split("\n")
    rows = []
    for n, line in enumerate(lines, start=1):
        safe = _html.escape(line) if line.strip() else "&nbsp;"
        hot = bool(start_line) and start_line <= n <= (end_line or start_line)
        anchor = " id='hl'" if (start_line and n == start_line) else ""
        rows.append(f"<tr class='{'hot' if hot else ''}'>"
                    f"<td class='ln'>{n}</td><td{anchor}>{safe}</td></tr>")
    return f"""<!doctype html><html><head><meta charset='utf-8'><style>
      body{{font-family:'Segoe UI',Arial,sans-serif;margin:0;background:#fff;color:#1a2533;}}
      table{{border-collapse:collapse;width:100%;font-size:13px;line-height:1.55;}}
      td.ln{{width:40px;text-align:right;color:#aab3c0;padding:1px 10px 1px 4px;
             user-select:none;border-right:1px solid #eef1f5;}}
      td{{padding:1px 10px;vertical-align:top;white-space:pre-wrap;word-break:break-word;}}
      tr.hot td{{background:#fff3bf;}}
      tr.hot td.ln{{background:#ffe98a;color:#7a5b00;font-weight:700;}}
    </style></head><body><table>{''.join(rows)}</table>
    <script>const e=document.getElementById('hl');if(e){{e.scrollIntoView({{block:'center'}});}}</script>
    </body></html>"""


@st.dialog("📄 Source transcript", width="large")
def show_source_dialog(doc: dict, doctor: str, country: str, quote: str,
                       start_line, end_line):
    if not doc:
        st.warning("Transcript not available — rebuild the index from the ① Ingest tab.")
        return
    st.markdown(f"**{doctor} · {country}**  ·  `{doc['source_file']}`  ·  "
                f"**{line_label(start_line, end_line)}**")
    st.markdown(f"<div class='cite'>“{quote}”</div>", unsafe_allow_html=True)
    components.html(_transcript_html(doc["raw_text"], start_line, end_line),
                    height=460, scrolling=True)


# ------------------------------------------------------- cached resources ---
@st.cache_resource(show_spinner="Loading hybrid index + models …")
def get_engine_and_judge():
    if not HybridIndex.exists():
        return None
    index = HybridIndex.load()
    retriever = HybridRetriever(index)
    engine = ResponseEngine(retriever)
    judge = build_judge(retriever)
    return {"index": index, "retriever": retriever, "engine": engine, "judge": judge}


def load_cached_eval():
    p = config.PROJECT_ROOT / "storage" / "eval_results.json"
    if p.exists():
        return json.load(open(p, encoding="utf-8"))
    return None


# ================================================================= SIDEBAR ==
res = get_engine_and_judge()
with st.sidebar:
    st.markdown("### 🩺 System Status")
    if res is None:
        st.error("No index found. Open the **① Ingest** tab and build it.")
    else:
        idx = res["index"]
        st.success("Index loaded")
        st.markdown(f"<span class='pill'>📚 {len(idx.chunks)} chunks</span>"
                    f"<span class='pill'>🧠 {idx.embedder.name}</span>"
                    f"<span class='pill'>💬 gen: {res['engine'].provider}</span>"
                    f"<span class='pill'>⚖️ judge: {'LLM' if res['judge'].use_llm else 'math'}</span>",
                    unsafe_allow_html=True)
    st.divider()
    st.markdown("#### Configuration")
    st.json(config.active_config(), expanded=False)
    st.caption("Provider-agnostic. Set `RAG_LLM_PROVIDER=openai|anthropic` + an "
               "API key to upgrade to LLM generation & LLM-as-judge. Defaults run "
               "with **no keys** (local embeddings + extractive answers).")

# =================================================================== HERO ===
st.markdown("""
<div class='hero'>
  <h1>Conversational RAG over COVID-19 Doctor Interviews</h1>
  <p>End-to-end pipeline · Hybrid retrieval · Grounded answers with citations ·
     Fallback safety · Self-evaluating with the RAG Triad</p>
</div>
""", unsafe_allow_html=True)

tab_overview, tab_ingest, tab_ask, tab_eval = st.tabs(
    ["🧭 Overview", "①  Ingest & Process", "②  Ask a Question", "③  Evaluate"])

# =============================================================== OVERVIEW ===
with tab_overview:
    st.markdown("#### The pipeline at a glance")
    cols = st.columns([3, 1, 3, 1, 3, 1, 3])
    steps = [
        ("1 · Ingest", "PDF/DOCX → clean → <b>Q&A-aware chunking</b> (question + answer never split) + metadata & topics"),
        ("2 · Index", "Embed each chunk + BM25. Persisted <b>hybrid store</b>"),
        ("3 · Retrieve", "<b>Dense + BM25 → Reciprocal Rank Fusion</b>"),
        ("4 · Respond", "Grounded answer + <b>citations</b>, or <b>fallback</b> if context weak"),
    ]
    positions = [0, 2, 4, 6]
    for i, (pos, (title, body)) in enumerate(zip(positions, steps)):
        with cols[pos]:
            st.markdown(f"<div class='step'><b>{title}</b><br>{body}</div>", unsafe_allow_html=True)
        if pos < 6:
            cols[pos + 1].markdown("<div class='flow-arrow'>➜</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Why these design choices")
        st.markdown("""
- **Q&A-aware chunking.** The transcripts are turn-based dialogues. We split on
  `Interviewer:` boundaries so each chunk is exactly one *question + its answer* —
  never separated. This keeps every chunk self-contained and on-topic.
- **Hybrid retrieval (dense + BM25 → RRF).** Dense embeddings catch *paraphrased*
  questions; BM25 nails *exact clinical terms & acronyms* (MIS-C, SpO₂ 92%,
  mucormycosis, Corona-Warn-App). Rank fusion combines both robustly.
- **Grounding + Fallback.** Every answer quotes its supporting lines and names the
  doctor/country. If nothing relevant is found, the engine *refuses and says why* —
  the two levers that cut hallucination.
        """)
    with c2:
        st.markdown("##### Evaluation = RAG Triad + KPIs")
        st.markdown("""
| Stage | What we measure | Metrics |
|---|---|---|
| **Retrieve** | Quality of retrieval | Context Relevance · Context Recall · Context Precision |
| **Answer** | Relevance & quality | Answer Relevance · Answer Correctness |
| **Ground** | Citation accuracy | Groundedness/Faithfulness · Citation Grounding · Source P/R |
| **Ops** | Reliability | Fallback Correctness · Latency |

The **RAG Triad** = *Context Relevance · Groundedness · Answer Relevance*
(from the *Building & Evaluating Advanced RAG* course).
        """)
    if res:
        cached = load_cached_eval()
        if cached:
            a = cached["aggregate"]; m = a["all_metrics"]
            st.markdown("##### Latest benchmark headline")
            k = st.columns(5)
            with k[0]: metric_card("RAG Triad", a["rag_triad_score"], "mean of 3", score_color(a["rag_triad_score"]))
            with k[1]: metric_card("Context Recall", m["context_recall"], "gold chunks found", score_color(m["context_recall"]))
            with k[2]: metric_card("Citation Grounding", m["citation_grounding"], "quotes verified", score_color(m["citation_grounding"]))
            with k[3]: metric_card("Fallback Correct", m["fallback_correct"], "abstains correctly", score_color(m["fallback_correct"]))
            with k[4]: metric_card("Latency", f"{m['latency_ms_total']:.0f} ms", "per question")

# ============================================================== ① INGEST ===
with tab_ingest:
    st.markdown("#### Access Point ① — Ingestion & Processing Pipeline")
    st.caption("Processes new documents, chunks them (Q&A kept together), extracts "
               "metadata + topics, embeds, and updates the knowledge base.")

    cbtn1, cbtn2 = st.columns([1, 3])
    with cbtn1:
        backend = st.selectbox("Embedding backend",
                               ["sentence-transformers", "tfidf", "auto"], index=0)
    with cbtn2:
        st.write("")
        st.write("")
        rebuild = st.button("🔄 Build / Rebuild knowledge base", type="primary")

    if rebuild:
        prog = st.progress(0.0, "Ingesting transcripts …")
        chunks = ingest_corpus()
        prog.progress(0.4, f"Chunked {len(chunks)} Q&A turns. Embedding + indexing …")
        index = HybridIndex.build(chunks, backend=None if backend == "auto" else backend)
        prog.progress(0.85, "Persisting index …")
        index.save()
        prog.progress(1.0, "Done")
        st.cache_resource.clear()
        st.success(f"Indexed {len(chunks)} chunks with {index.embedder.name}. Reloading …")
        st.rerun()

    if res is None:
        st.info("No knowledge base yet — click **Build** above.")
    else:
        chunks = res["index"].chunks
        stats = corpus_stats(chunks)
        m = st.columns(4)
        with m[0]: metric_card("Documents", str(stats["n_documents"]))
        with m[1]: metric_card("Q&A Chunks", str(stats["n_chunks"]))
        with m[2]: metric_card("Avg chunk size", f"{stats['avg_chunk_chars']} ch", "question + answer together")
        with m[3]: metric_card("Embedding dim", str(res["index"].dense_matrix.shape[1]))

        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown("**Chunks by country**")
            dfc = pd.DataFrame(stats["by_country"].items(), columns=["Country", "Chunks"])
            st.plotly_chart(go.Figure(go.Bar(x=dfc["Country"], y=dfc["Chunks"],
                            marker_color="#1d6fb8")).update_layout(height=300, margin=dict(t=10,b=10)),
                            width="stretch")
        with cc2:
            st.markdown("**Chunks by auto-tagged topic**")
            dft = pd.DataFrame(stats["by_topic"].items(), columns=["Topic", "Chunks"])
            st.plotly_chart(go.Figure(go.Bar(x=dft["Chunks"], y=dft["Topic"], orientation="h",
                            marker_color="#0d3b66")).update_layout(height=300, margin=dict(t=10,b=10),
                            yaxis=dict(autorange="reversed")), width="stretch")

        st.markdown("##### 🔎 Chunk explorer")
        docs = sorted({c["doc_id"] for c in chunks})
        doc = st.selectbox("Document", docs, format_func=lambda d: d.replace("_", " "))
        doc_chunks = [c for c in chunks if c["doc_id"] == doc]
        if doc_chunks:
            meta0 = doc_chunks[0]
            st.markdown(f"**{meta0['doctor']}** · {meta0['country']} · {meta0['role']}  ·  {len(doc_chunks)} chunks")
        for c in doc_chunks:
            with st.expander(f"Turn {c['turn_index']} · {c['id']}  ·  "
                             f"{', '.join(c['topics'])}"):
                if c["question"]:
                    st.markdown(f"**Q:** {c['question']}")
                st.markdown(f"**A:** {c['answer']}")

# ============================================================== ② ASK ======
# Fallback list used only if the golden dataset can't be loaded.
SAMPLES = [
    "What is MIS-C and how was it treated in children?",
    "What is the exact recommended dose of dexamethasone for COVID?",
    "Give me a recipe for chicken biryani.",
]


@st.cache_data(show_spinner=False)
def sample_questions() -> list[tuple[str, str]]:
    """All curated questions from the golden benchmark — (question, tag).
    Answerable ones show their type; fallback demos are clearly marked so the
    Ask tab and the Evaluate tab stay in sync."""
    try:
        items = load_golden()
    except Exception:
        return [(q, "") for q in SAMPLES]
    out = []
    for it in items:
        tag = ("⚠️ fallback demo" if not it.get("answerable", True)
               else it.get("type", "").replace("_", " "))
        out.append((it["question"], tag))
    return out


with tab_ask:
    st.markdown("#### Access Point ② — Response Engine (single-turn QA)")
    st.caption("Retrieve → check relevance → answer grounded in the transcripts with "
               "citations, or fall back with a reason.")

    if res is None:
        st.warning("Build the knowledge base in the ① Ingest tab first.")
    else:
        samples = sample_questions()
        tag_of = dict(samples)

        def _fmt_sample(q: str) -> str:
            if q == "—":
                return "— pick a curated question (or type your own below) —"
            t = tag_of.get(q, "")
            return f"{q}   ·  {t}" if t else q

        sample = st.selectbox("Try a sample question (all curated benchmark questions)",
                              ["—"] + [q for q, _ in samples], format_func=_fmt_sample)
        default_q = "" if sample == "—" else sample
        query = st.text_input("Your question", value=default_q,
                              placeholder="Ask about the COVID-19 doctor interviews …")

        go_btn = st.button("🔍 Answer", type="primary", disabled=not query.strip())

        if go_btn and query.strip():
            with st.spinner("Retrieving + composing grounded answer …"):
                st.session_state["ask_result"] = res["engine"].answer(query)

        # Render from session_state so the per-citation "View source" buttons
        # (which trigger a rerun) keep the answer + citations on screen.
        result = st.session_state.get("ask_result")
        if result is not None:
            # --- confidence + latency strip ---
            t = st.columns([1, 1, 1, 3])
            with t[0]: metric_card("Confidence", result.confidence, "best semantic match",
                                   score_color(result.confidence))
            with t[1]: metric_card("Retrieval", f"{result.timings_ms.get('retrieval',0):.0f} ms")
            with t[2]: metric_card("Total", f"{result.timings_ms.get('total',0):.0f} ms")
            with t[3]:
                st.write(""); st.write("")
                st.caption(f"Generator: **{result.provider}**  ·  "
                           f"fallback threshold {res['engine'].fallback_min_sim:.2f}")

            # --- answer / fallback ---
            if result.is_fallback:
                st.markdown(f"<div class='fallback'>⚠️ <b>Fallback — no confident answer.</b><br><br>"
                            f"{result.answer}</div>", unsafe_allow_html=True)
                st.info("This is the **anti-hallucination** safety net: rather than "
                        "guess, the engine declines and explains why.")
            else:
                st.markdown(f"<div class='answer-box'>{result.answer}</div>",
                            unsafe_allow_html=True)
                st.markdown("##### 📌 Citations (grounding) — click **Source** to see the highlighted line in the transcript")
                docs = res["index"].documents
                for i, c in enumerate(result.citations):
                    col1, col2 = st.columns([7, 1])
                    with col1:
                        st.markdown(
                            f"<div class='cite'><span class='src'>{c.doctor} · {c.country} · "
                            f"{c.source_file}</span> &nbsp; <span style='color:#8a93a0'>"
                            f"({line_label(c.start_line, c.end_line)} · relevance {c.relevance:.2f})"
                            f"</span><br>“{c.quote}”</div>", unsafe_allow_html=True)
                    with col2:
                        doc = docs.get(c.chunk_id.split("::")[0])
                        if c.start_line and doc and st.button("📄 Source", key=f"src_{i}"):
                            show_source_dialog(doc, c.doctor, c.country, c.quote,
                                               c.start_line, c.end_line)

            # --- retrieval transparency ---
            with st.expander("🔬 Retrieval details — how hybrid search ranked the chunks"):
                rows = [{
                    "chunk": rc.id.split("::")[-1],
                    "doctor": rc.chunk["doctor"].replace("Dr. ", ""),
                    "country": rc.chunk["country"],
                    "dense": round(rc.dense_score, 3),
                    "bm25": round(rc.sparse_score, 2),
                    "dense_rank": rc.dense_rank,
                    "bm25_rank": rc.sparse_rank,
                    "fused": round(rc.fused_score, 4),
                } for rc in result.retrieved]
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
                st.caption("**dense** = semantic cosine · **bm25** = lexical score · "
                           "**fused** = Reciprocal Rank Fusion of both rankings. Notice how "
                           "a chunk can rank low on one signal but high after fusion.")

# ============================================================== ③ EVALUATE =
with tab_eval:
    st.markdown("#### Access Point ③ — Evaluation Pipeline (Golden Benchmark)")
    st.caption("Runs the Response Engine over the curated golden dataset and scores "
               "retrieval, answering, grounding & reliability — automatically.")

    if res is None:
        st.warning("Build the knowledge base in the ① Ingest tab first.")
    else:
        golden = load_golden()
        run_cols = st.columns([1, 1, 4])
        with run_cols[0]:
            run_live = st.button("▶️ Run evaluation live", type="primary")
        with run_cols[1]:
            use_cache = st.button("📂 Load last results")

        results = None
        if run_live:
            bar = st.progress(0.0, "Starting …")
            def _p(i, n, qid):
                bar.progress(i / n, f"Evaluated {i}/{n}  ({qid})")
            t0 = time.perf_counter()
            results = run_evaluation(res["engine"], res["judge"], golden, progress=_p)
            bar.progress(1.0, f"Done in {time.perf_counter()-t0:.1f}s")
            results["meta"] = {"embedder": res["index"].embedder.name,
                               "provider": res["engine"].provider,
                               "judge_llm": res["judge"].use_llm}
            json.dump(results, open(config.PROJECT_ROOT/"storage"/"eval_results.json","w",
                      encoding="utf-8"), ensure_ascii=False, indent=2)
        elif use_cache:
            results = load_cached_eval()
            if results is None:
                st.error("No cached results found — run live once.")
        else:
            results = load_cached_eval()

        if results:
            agg = results["aggregate"]; M = agg["all_metrics"]
            meta = results.get("meta", {})
            st.caption(f"Backend: **{meta.get('embedder','?')}** · generator: "
                       f"**{meta.get('provider','?')}** · judge: "
                       f"**{'LLM-as-judge' if meta.get('judge_llm') else 'math (embedding-based)'}** · "
                       f"{agg['n_items']} questions ({agg['n_answerable']} answerable, "
                       f"{agg['n_fallback_expected']} expected-fallback)")

            # ---- 4 MECE dimensions: each hero = one RAG-Triad metric (Ops uses
            #      Fallback Correctness, since the triad has no Ops member).
            #      Every supporting metric belongs to exactly one section. ----
            rt = agg["rag_triad"]

            # Overall headline = mean of the three triad heroes (a roll-up, not a 5th metric).
            oc1, oc2 = st.columns([2.3, 4])
            with oc1:
                st.markdown(
                    f"##### 🎯 RAG Triad (overall): "
                    f"<span style='color:{score_color(agg['rag_triad_score'])};font-weight:800'>"
                    f"{agg['rag_triad_score']:.2f}</span>", unsafe_allow_html=True)
                metric_info_popover("What is the RAG Triad?", "rag_triad", prefix="📖 ")
            with oc2:
                st.caption("Mean of the three hero metrics below (Context Relevance · "
                           "Groundedness · Answer Relevance). Click any **📖 metric name** "
                           "for its definition and exact formula.")
            st.write("")

            # ① RETRIEVE — hero + the two complementary coverage/purity metrics.
            eval_section(
                "① Context Relevance · Retrieval — did we fetch the right context?",
                "Hero metric: **Context Relevance** (RAG Triad — semantic match of retrieved "
                "chunks to the question). Side metrics: **Recall** (did we get the gold chunks?) "
                "and **Precision** (how clean is the top-5?) — the two complementary facets.",
                rt["context_relevance"], "Context Relevance", "context_relevance",
                [("Context Recall", M["context_recall"], "fraction of gold found (coverage)", "context_recall"),
                 ("Context Precision", M["context_precision"], "top-5 that are gold (purity)", "context_precision")])

            st.divider()
            # ② GROUND — faithfulness of the answer to its sources.
            eval_section(
                "② Groundedness · Faithfulness — is every claim backed by the sources?",
                "Hero: **Groundedness** (RAG Triad — is the answer supported by the retrieved "
                "context?). Side metric verifies the literal citations against the source text.",
                rt["groundedness"], "Groundedness", "groundedness",
                [("Citation Grounding", M["citation_grounding"], "quoted lines verified in source", "citation_grounding")])

            st.divider()
            # ③ ANSWER — does the answer serve the question.
            eval_section(
                "③ Answer Relevance · Quality — does the answer address the question?",
                "Hero: **Answer Relevance** (RAG Triad — does the answer actually respond to the "
                "question?). Side metric compares it against the curated ideal answer.",
                rt["answer_relevance"], "Answer Relevance", "answer_relevance",
                [("Answer Correctness", M["answer_correctness"], "vs ideal answer (accuracy)", "answer_correctness")])

            st.divider()
            # ④ OPS — reliability & cost (no triad member ⇒ Fallback Correctness is the hero).
            eval_section(
                "④ Ops · Reliability — does it abstain safely and respond fast?",
                "Hero: **Fallback Correctness** — abstains when the answer isn't in the corpus "
                "instead of hallucinating. Side metric is end-to-end speed.",
                M["fallback_correct"], "Fallback Correctness", "fallback_correct",
                [("Latency", f"{M['latency_ms_total']:.0f} ms", "per question (end-to-end)", "latency_ms_total")])

            st.divider()

            # ---- per-item table ----
            st.markdown("##### 🧾 Per-question results")
            def _num(x):
                return None if (isinstance(x, float) and x != x) else round(x, 2)
            prow = []
            for it in results["per_item"]:
                mm = it["metrics"]
                prow.append({
                    "id": it["id"], "type": it["type"],
                    "question": it["question"][:60] + ("…" if len(it["question"]) > 60 else ""),
                    "fallback": "✓" if it["is_fallback"] else "",
                    "fb_ok": "✅" if it["fallback_correct"] else "❌",
                    "ctx_recall": _num(mm["context_recall"]),
                    "ctx_precision": _num(mm["context_precision"]),
                    "answer_corr": _num(mm["answer_correctness"]),
                    "groundedness": _num(mm["groundedness"]),
                    "cite_grnd": _num(mm["citation_grounding"]),
                })
            st.dataframe(pd.DataFrame(prow), width="stretch", hide_index=True)

            # ---- drilldown ----
            st.markdown("##### 🔍 Inspect a question")
            ids = [it["id"] for it in results["per_item"]]
            pick = st.selectbox("Question", ids,
                                format_func=lambda i: f"{i} · " +
                                next(x['question'] for x in results['per_item'] if x['id'] == i))
            item = next(x for x in results["per_item"] if x["id"] == pick)
            dc1, dc2 = st.columns(2)
            with dc1:
                st.markdown(f"**Question:** {item['question']}")
                st.markdown(f"**Type:** `{item['type']}` · **Answerable:** {item['answerable']}")
                st.markdown("**Ideal answer:**")
                st.info(item["expected_answer"])
            with dc2:
                st.markdown("**System answer:**")
                if item["is_fallback"]:
                    st.warning(item["result"]["answer"])
                else:
                    st.success(item["result"]["answer"])
                if item["result"]["citations"]:
                    st.markdown("**Citations:**")
                    for c in item["result"]["citations"]:
                        loc = line_label(c.get("start_line"), c.get("end_line"))
                        st.markdown(f"<div class='cite'><span class='src'>{c['doctor']} · "
                                    f"{c['country']}</span> <span style='color:#8a93a0'>· {loc}</span>"
                                    f"<br>“{c['quote']}”</div>", unsafe_allow_html=True)
            st.markdown("**Metrics for this question:**")
            st.json({k: (round(v, 3) if isinstance(v, float) and v == v else v)
                     for k, v in item["metrics"].items()})

            # ---- by-type ----
            with st.expander("📈 Breakdown by question type"):
                bt = agg["by_type"]
                st.dataframe(pd.DataFrame([
                    {"type": t, "n": d["n"],
                     "answer_correctness": d["answer_correctness"],
                     "context_recall": d["context_recall"],
                     "fallback_correct": d["fallback_correct"]}
                    for t, d in bt.items()]), width="stretch", hide_index=True)
        else:
            st.info("Click **Run evaluation live** to score the golden benchmark.")
