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
    """Split one transcript into Q&A turn chunks (question + answer kept together)."""
    text = _normalize(raw_text)
    # everything up to the first "Interviewer:" is the header/preamble
    parts = re.split(r"Interviewer\s*:", text)
    preamble = parts[0]
    meta = _parse_header(preamble, filename)
    doc_id = Path(filename).stem

    # The doctor's reply label, e.g. "Dr. Johnson:". Build a tolerant pattern.
    speaker_pat = re.compile(
        rf"Dr\.?\s+{re.escape(meta['surname'])}\s*:" if meta["surname"]
        else r"Dr\.?\s+[A-Z][a-z]+\s*:")

    chunks: list[dict] = []
    turn = 0
    for segment in parts[1:]:
        segment = segment.strip()
        if not segment:
            continue
        split = speaker_pat.split(segment, maxsplit=1)
        if len(split) == 2:
            question, answer = split[0].strip(), split[1].strip()
        else:
            # no detectable doctor label -> treat whole segment as answer
            question, answer = "", segment
        # drop trailing "--- End of Interview ---" markers from the last answer
        answer = re.sub(r"-{2,}\s*End of Interview\s*-{2,}.*$", "", answer,
                        flags=re.IGNORECASE).strip()
        if not answer:
            continue
        turn += 1
        combined = (f"Q: {question}\nA: {answer}" if question else f"A: {answer}")
        chunk_id = f"{doc_id}::q{turn:02d}"
        chunks.append({
            "id": chunk_id,
            "doc_id": doc_id,
            "source_file": filename,
            "turn_index": turn,
            "question": question,
            "answer": answer,
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


def ingest_corpus(data_dir: Path | None = None) -> list[dict]:
    """Load + chunk every transcript in the data directory."""
    all_chunks: list[dict] = []
    for path in iter_source_files(data_dir):
        try:
            raw = load_any(path)
            all_chunks.extend(chunk_transcript(raw, path.name))
        except Exception as exc:  # keep going on a single bad file
            print(f"[ingest] FAILED {path.name}: {exc}")
    return all_chunks


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
