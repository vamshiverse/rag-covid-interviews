"""Document Ingestion & Processing Pipeline.

Loads PDF/DOCX interview transcripts and turns each into a list of
**Question+Answer chunks**. The single most important design rule for this
corpus (per the spec) is that a chunk must NEVER split an interviewer's
question from the doctor's answer -- they always travel together. Because the
transcripts are clean turn-based dialogues, we chunk on the natural
`Interviewer:` boundaries instead of a fixed character window. Each chunk
therefore equals exactly one full conversational turn.

Each chunk carries rich extracted metadata: doctor, country, role/specialty,
location, source file, turn index, and auto-tagged topics.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from pypdf import PdfReader
import docx  # python-docx

from . import config

COUNTRIES = {"US", "USA", "India", "Germany", "UK", "Italy", "Brazil", "France"}

# Topic -> trigger keywords (lower-cased substring match on chunk text).
TOPIC_KEYWORDS = {
    "Telemedicine & Remote Care": ["telemedicine", "telehealth", "remote monitoring",
                                   "video", "teletriage", "telepediatric", "pulse oxim"],
    "Vaccination & Hesitancy": ["vaccin", "hesitanc", "immuniz", "booster", "dose"],
    "Mental Health & Wellbeing": ["mental health", "anxiety", "depression", "burnout",
                                  "moral distress", "psycholog", "loneliness", "grief"],
    "PPE & Infection Control": ["ppe", "mask", "face shield", "infection control",
                                "isolation room", "n95", "protective"],
    "Testing & Contact Tracing": ["testing", "contact tracing", "quarantine", "rt-pcr",
                                  "swab", "surveillance"],
    "Pediatrics & MIS-C": ["children", "pediatric", "neonat", "infant", "mis-c",
                           "adolescent", "school"],
    "Equity & Vulnerable Groups": ["vulnerable", "inequit", "equity", "undocumented",
                                   "housing", "low-income", "migrant", "rural", "slum"],
    "ICU & Critical Care": ["icu", "ventilator", "intubat", "oxygen", "critical care",
                            "ards", "ecmo", "proning"],
    "Misinformation & Trust": ["misinformation", "myth", "rumor", "rumour", "conspiracy",
                               "trust", "whatsapp"],
    "Policy & Preparedness": ["policy", "preparedness", "stockpile", "public health",
                              "reimbursement", "funding", "lockdown", "government"],
    "Long COVID & Aftercare": ["long covid", "post-viral", "long-term", "aftercare",
                               "rehabilitation", "deferred care", "chronic"],
}


# --------------------------------------------------------------- loaders ----
def load_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def load_docx(path: Path) -> str:
    document = docx.Document(str(path))
    return "\n".join(p.text for p in document.paragraphs)


def load_any(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path)
    if suffix in (".docx", ".doc"):
        return load_docx(path)
    raise ValueError(f"Unsupported file type: {path.name}")


# ---------------------------------------------------------------- cleaning --
def _normalize(text: str) -> str:
    """Collapse the messy whitespace from PDF/DOCX extraction."""
    text = text.replace("\r", "\n")
    # join soft hyphenation at line breaks ("remote-\nmonitoring" -> "remote-monitoring")
    text = re.sub(r"-\n\s*", "-", text)
    # collapse all remaining whitespace runs to single spaces
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _normalize_with_map(raw: str) -> tuple[str, list[int]]:
    """Same normalisation as `_normalize`, but also return idx_map where
    idx_map[j] = index into the (CR-replaced) `raw` string that normalised
    char j came from. Lets us map a cited sentence back to its source line(s).
    `raw.replace('\\r','\\n')` is length-preserving, so the indices line up."""
    s = raw.replace("\r", "\n")
    out: list[str] = []
    idx: list[int] = []
    i, n = 0, len(s)
    while i < n:
        ch = s[i]
        if ch == "-" and i + 1 < n and s[i + 1] == "\n":   # de-hyphenate "-\n\s*"
            out.append("-"); idx.append(i)
            i += 2
            while i < n and s[i].isspace():
                i += 1
            continue
        if ch.isspace():                                   # collapse whitespace run
            out.append(" "); idx.append(i)
            i += 1
            while i < n and s[i].isspace():
                i += 1
            continue
        out.append(ch); idx.append(i)
        i += 1
    # mimic .strip() on the single-space-normalised result
    start, end = 0, len(out)
    while start < end and out[start] == " ":
        start += 1
    while end > start and out[end - 1] == " ":
        end -= 1
    return "".join(out[start:end]), idx[start:end]


def _line_starts(text: str) -> list[int]:
    """Char offsets at which each line begins (text assumed CR-replaced)."""
    starts = [0]
    for i, ch in enumerate(text):
        if ch == "\n":
            starts.append(i + 1)
    return starts


def _line_of(starts: list[int], pos: int) -> int:
    """1-indexed line number containing char offset `pos`."""
    import bisect
    return bisect.bisect_right(starts, pos)


def _parse_header(preamble: str, filename: str) -> dict:
    """Extract doctor / location / role / country from the title line + filename."""
    # Filename: Dr_Sarah_Johnson_US_Detailed_Interview
    stem_tokens = Path(filename).stem.split("_")
    country = next((t for t in stem_tokens if t in COUNTRIES), "Unknown")
    if country == "USA":
        country = "US"
    # name tokens sit between "Dr" and the country token
    name_tokens = []
    for tok in stem_tokens[1:]:
        if tok in COUNTRIES or tok in ("Detailed", "Interview"):
            break
        name_tokens.append(tok)
    name = "Dr. " + " ".join(name_tokens) if name_tokens else "Unknown"
    surname = name_tokens[-1] if name_tokens else ""

    # Title line: "Interview Transcript with Dr. X (Location) - Role, Institution"
    location, role = "", ""
    m = re.search(r"with\s+(Dr\.?\s+[^()]+?)\s*\(([^)]*)\)\s*[-–—]\s*(.+?)(?:Interviewer:|$)",
                  preamble)
    if m:
        name = m.group(1).strip().rstrip(",") or name
        location = m.group(2).strip()
        role = m.group(3).strip().rstrip(". ")
        if not surname:
            surname = name.split()[-1]
    # short specialty = first comma-segment of the role string
    specialty = role.split(",")[0].strip() if role else ""
    return {
        "doctor": name,
        "surname": surname,
        "country": country,
        "location": location,
        "role": role,
        "specialty": specialty,
    }


def _tag_topics(text: str) -> list[str]:
    low = text.lower()
    tags = [topic for topic, kws in TOPIC_KEYWORDS.items()
            if any(kw in low for kw in kws)]
    return tags or ["General"]


# --------------------------------------------------------------- chunking ---
def chunk_transcript(raw_text: str, filename: str) -> list[dict]:
    """Split one transcript into Q&A turn chunks (question + answer kept together).

    Chunking is anchored to the RAW (newline-preserving) text so every chunk
    knows the literal source line where its answer begins -> enables line-level
    citations. The `answer`/`question`/`text` fields are still normalised for
    embedding & search; `raw_answer` keeps the original newlines.
    """
    raw = raw_text.replace("\r", "\n")     # keep newlines; only fold CRs
    starts = _line_starts(raw)
    doc_id = Path(filename).stem

    matches = list(re.finditer(r"Interviewer\s*:", raw))
    preamble = raw[:matches[0].start()] if matches else raw
    meta = _parse_header(_normalize(preamble), filename)

    # The doctor's reply label, e.g. "Dr. Johnson:". Build a tolerant pattern.
    speaker_pat = re.compile(
        rf"Dr\.?\s+{re.escape(meta['surname'])}\s*:" if meta["surname"]
        else r"Dr\.?\s+[A-Z][a-z]+\s*:")

    chunks: list[dict] = []
    turn = 0
    for k, m in enumerate(matches):
        seg_start = m.end()
        seg_end = matches[k + 1].start() if k + 1 < len(matches) else len(raw)
        segment = raw[seg_start:seg_end]

        sm = speaker_pat.search(segment)
        if sm:
            question_raw = segment[:sm.start()]
            answer_raw = segment[sm.end():]
            answer_off = seg_start + sm.end()           # raw offset where answer begins
        else:
            question_raw, answer_raw, answer_off = "", segment, seg_start

        # drop trailing "--- End of Interview ---" marker
        end_m = re.search(r"-{2,}\s*End of Interview\s*-{2,}", answer_raw, re.IGNORECASE)
        if end_m:
            answer_raw = answer_raw[:end_m.start()]
        # advance past leading whitespace so answer_off points at real text
        lead = len(answer_raw) - len(answer_raw.lstrip())
        answer_off += lead
        answer_raw = answer_raw.strip()
        if not answer_raw:
            continue

        turn += 1
        question = _normalize(question_raw)
        answer = _normalize(answer_raw)
        combined = (f"Q: {question}\nA: {answer}" if question else f"A: {answer}")
        chunks.append({
            "id": f"{doc_id}::q{turn:02d}",
            "doc_id": doc_id,
            "source_file": filename,
            "turn_index": turn,
            "question": question,
            "answer": answer,
            "raw_answer": answer_raw,                    # original newlines preserved
            "answer_line_start": _line_of(starts, answer_off),
            "text": combined,             # what we embed / search over (Q + A together)
            "topics": _tag_topics(combined),
            "n_chars": len(combined),
            **{k: meta[k] for k in ("doctor", "country", "location", "role", "specialty")},
        })
    return chunks


# ---------------------------------------------------------- corpus driver ---
def iter_source_files(data_dir: Path | None = None) -> Iterable[Path]:
    data_dir = data_dir or config.DATA_DIR
    for path in sorted(data_dir.rglob("*")):
        if "__MACOSX" in path.parts:
            continue
        if path.name.startswith("._"):
            continue
        if path.suffix.lower() in (".pdf", ".docx", ".doc"):
            yield path


def ingest_corpus_with_docs(data_dir: Path | None = None) -> tuple[list[dict], dict]:
    """Load + chunk every transcript, AND return the raw transcripts (for the
    line-level source viewer). documents: doc_id -> {source_file, raw_text}."""
    all_chunks: list[dict] = []
    documents: dict = {}
    for path in iter_source_files(data_dir):
        try:
            raw = load_any(path)
            all_chunks.extend(chunk_transcript(raw, path.name))
            documents[Path(path.name).stem] = {
                "source_file": path.name,
                "raw_text": raw.replace("\r", "\n"),
            }
        except Exception as exc:  # keep going on a single bad file
            print(f"[ingest] FAILED {path.name}: {exc}")
    return all_chunks, documents


def ingest_corpus(data_dir: Path | None = None) -> list[dict]:
    """Load + chunk every transcript in the data directory."""
    return ingest_corpus_with_docs(data_dir)[0]


def corpus_stats(chunks: list[dict]) -> dict:
    docs = {c["doc_id"] for c in chunks}
    countries: dict[str, int] = {}
    topics: dict[str, int] = {}
    for c in chunks:
        countries[c["country"]] = countries.get(c["country"], 0) + 1
        for t in c["topics"]:
            topics[t] = topics.get(t, 0) + 1
    sizes = [c["n_chars"] for c in chunks] or [0]
    return {
        "n_documents": len(docs),
        "n_chunks": len(chunks),
        "avg_chunk_chars": round(sum(sizes) / len(sizes)),
        "min_chunk_chars": min(sizes),
        "max_chunk_chars": max(sizes),
        "by_country": dict(sorted(countries.items(), key=lambda x: -x[1])),
        "by_topic": dict(sorted(topics.items(), key=lambda x: -x[1])),
    }


if __name__ == "__main__":
    import json
    chunks = ingest_corpus()
    print(json.dumps(corpus_stats(chunks), indent=2))
    print(f"\nSample chunk:\n{json.dumps(chunks[0], indent=2)[:1200]}")
