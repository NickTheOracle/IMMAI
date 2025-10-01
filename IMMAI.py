#!/usr/bin/env python3
"""
Immigration Research Chat — single-file Streamlit app (RAG, cites .gov)

- Searches only trusted government domains.
- Extracts pages, ranks relevant passages, composes a grounded answer, shows citations.
- Informational only. Not legal advice. No attorney–client relationship.

Run:
  pip install --upgrade pip
  pip install streamlit duckduckgo-search trafilatura sentence-transformers transformers torch numpy
  # Optional PDF export:
  pip install reportlab
  streamlit run imm_rag_chat.py
"""

from __future__ import annotations
import io
import re
import html
import textwrap
import urllib.parse
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import streamlit as st

# Optional PDF dependency
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    REPORTLAB_OK = True
except Exception:
    REPORTLAB_OK = False

# Search / scrape / vector / generate
from duckduckgo_search import DDGS
import trafilatura
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import numpy as np

# --------------------------- Config ---------------------------

TRUSTED_DOMAINS = [
    "uscis.gov",
    "travel.state.gov",
    "justice.gov",
    "dhs.gov",
    "cbp.gov",
    "ice.gov",
    "congress.gov",
    "federalregister.gov",
    "govinfo.gov",
    "whitehouse.gov",
    "usa.gov",
]

MAX_RESULTS_PER_DOMAIN = 3
GLOBAL_RESULT_CAP = 12
FETCH_TIMEOUT = 20
CHUNK_CHARS = 1200
TOP_CHUNKS = 5
GEN_MAX_NEW_TOKENS = 320

DISCLAIMER = (
    "**Disclaimer:** Informational only. Not legal advice. No attorney–client relationship. "
    "Citations are to government sources. Laws and policy change."
)
REFUSAL = "Requests to lie, use false documents, evade the law, or commit fraud are refused."
INSTRUCTIONS_PROMPT = (
    "You are an immigration information assistant. Answer ONLY using the provided sources. "
    "Be concise, neutral, and high-level. Do not invent facts, fees, timelines, or tactics. "
    "Flag uncertainty explicitly. End with a one-sentence caveat that this is general information, not legal advice."
)
SYSTEM_SUFFIX = " This summary is not legal advice."

# --------------------------- Helpers ---------------------------

def norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def is_trusted(url: str) -> bool:
    try:
        host = urllib.parse.urlsplit(url).netloc.lower().split(":")[0]
    except Exception:
        return False
    return any(host.endswith(dom) for dom in TRUSTED_DOMAINS)

def safe_title(url: str) -> str:
    try:
        parts = urllib.parse.urlsplit(url)
        return f"{parts.netloc}{parts.path or '/'}"
    except Exception:
        return url

# --------------------------- Retrieval ---------------------------

@dataclass
class Hit:
    url: str
    title: str
    snippet: str

@dataclass
class DocChunk:
    url: str
    title: str
    text: str
    score: float

@st.cache_resource(show_spinner=False)
def load_embedder() -> SentenceTransformer:
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=False)
def load_llm():
    name = "google/flan-t5-base"  # small, CPU-friendly
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSeq2SeqLM.from_pretrained(name)
    return tok, model

EMBED = load_embedder()
TOK, LLM = load_llm()

def ddg_trusted_search(query: str) -> List[Hit]:
    hits: List[Hit] = []
    q = norm_space(query)
    try:
        with DDGS() as ddgs:
            for dom in TRUSTED_DOMAINS:
                q_dom = f"site:{dom} {q}"
                for r in ddgs.text(q_dom, max_results=MAX_RESULTS_PER_DOMAIN, region="us-en", safesearch="moderate"):
                    url = r.get("href") or r.get("url") or ""
                    title = r.get("title") or ""
                    body = r.get("body") or ""
                    if not url or not is_trusted(url):
                        continue
                    hits.append(Hit(url=url, title=title, snippet=body))
                    if len(hits) >= GLOBAL_RESULT_CAP:
                        break
                if len(hits) >= GLOBAL_RESULT_CAP:
                    break
    except Exception:
        pass
    # Deduplicate by URL
    uniq, seen = [], set()
    for h in hits:
        if h.url in seen:
            continue
        seen.add(h.url)
        uniq.append(h)
    return uniq

def fetch_and_chunk(url: str, title_fallback: str) -> List[Tuple[str, str, str]]:
    """Return list of (url, title, chunk_text)."""
    try:
        html_doc = trafilatura.fetch_url(url, timeout=FETCH_TIMEOUT)
    except Exception:
        html_doc = None
    if not html_doc:
        return []
    try:
        extracted = trafilatura.extract(html_doc, include_comments=False, include_tables=False)
    except Exception:
        extracted = None
    if not extracted:
        return []
    text = norm_space(extracted)
    if not text:
        return []

    title = title_fallback or safe_title(url)
    chunks: List[Tuple[str, str, str]] = []
    for i in range(0, len(text), CHUNK_CHARS):
        seg = text[i:i + CHUNK_CHARS]
        if i > 0 and len(seg) < 240:
            break
        chunks.append((url, title, seg))
    return chunks

def embed(texts: List[str]) -> np.ndarray:
    return EMBED.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

def rank_chunks(question: str, docs: List[Tuple[str, str, str]]) -> List[DocChunk]:
    if not docs:
        return []
    qv = embed([question])[0]
    tv = embed([d[2] for d in docs])
    sims = (tv @ qv)  # cosine similarity due to normalization
    scored: List[DocChunk] = []
    for (url, title, chunk), s in zip(docs, sims):
        scored.append(DocChunk(url=url, title=title or safe_title(url), text=chunk, score=float(s)))
    scored.sort(key=lambda x: x.score, reverse=True)
    return scored

def format_citations(urls: List[str]) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    idx = 1
    for u in urls:
        if u not in mapping:
            mapping[u] = idx
            idx += 1
    return mapping

def make_prompt(question: str, top_chunks: List[DocChunk], numbered: Dict[str, int]) -> str:
    lines = [INSTRUCTIONS_PROMPT, "", "SOURCES:"]
    for ch in top_chunks:
        i = numbered[ch.url]
        snippet = textwrap.shorten(ch.text, width=800, placeholder=" …")
        lines.append(f"[{i}] {ch.title} :: {ch.url}")
        lines.append(f"Excerpt: {snippet}")
        lines.append("")
    lines.append("QUESTION:")
    lines.append(question)
    lines.append("")
    lines.append("ANSWER:")
    return "\n".join(lines)

def generate_answer(prompt: str) -> str:
    inputs = TOK(prompt, return_tensors="pt", truncation=True, max_length=1024)
    with torch.no_grad():
        out = LLM.generate(**inputs, max_new_tokens=GEN_MAX_NEW_TOKENS, temperature=0.2, do_sample=False)
    return TOK.decode(out[0], skip_special_tokens=True).strip()

def extractive_fallback(chunks: List[DocChunk]) -> str:
    """If LLM fails, return concise extractive bullets from top chunks."""
    bullets: List[str] = []
    for ch in chunks[:3]:
        # first 2 sentences
        sents = re.split(r"(?<=[\.\?\!])\s+", ch.text)
        pick = " ".join(sents[:2])
        bullets.append(textwrap.shorten(pick, width=300, placeholder=" …"))
    if not bullets:
        return "No qualifying primary sources parsed."
    return " • " + "\n • ".join(bullets)

def build_research_answer(question: str) -> Tuple[str, List[Tuple[int, str, str]]]:
    hits = ddg_trusted_search(question)
    if not hits:
        return ("No qualifying primary sources found on trusted domains.", [])

    docs: List[Tuple[str, str, str]] = []
    for h in hits:
        docs.extend(fetch_and_chunk(h.url, h.title or safe_title(h.url))[:3])

    if not docs:
        return ("Sources were found but could not be parsed.", [])

    ranked = rank_chunks(question, docs)[:TOP_CHUNKS]
    numbered = format_citations([c.url for c in ranked])
    prompt = make_prompt(question, ranked, numbered)

    try:
        raw = generate_answer(prompt) or ""
    except Exception:
        raw = ""

    if raw:
        answer = norm_space(raw) + SYSTEM_SUFFIX
    else:
        answer = extractive_fallback(ranked) + SYSTEM_SUFFIX

    citations: List[Tuple[int, str, str]] = []
    for url, idx in sorted(numbered.items(), key=lambda kv: kv[1]):
        title = next((c.title for c in ranked if c.url == url), None) or safe_title(url)
        citations.append((idx, title, url))
    return answer, citations

# --------------------------- UI ---------------------------

st.set_page_config(page_title="Immigration Research Chat", layout="centered")

if "messages" not in st.session_state:
    st.session_state.messages = []      # list[(role, text)]
if "last_citations" not in st.session_state:
    st.session_state.last_citations = []

st.title("Immigration Research Chat")
st.markdown(DISCLAIMER)
st.markdown(REFUSAL)
st.markdown("---")

# INPUT FIRST so submission updates state BEFORE rendering history
with st.form("ask", clear_on_submit=True):
    q = st.text_area("Ask about immigration (general topics, forms, concepts).", height=120)
    submitted = st.form_submit_button("Send")

if submitted:
    user_q = norm_space(q)
    if not user_q:
        st.session_state.messages.append(("System", "Enter a question."))
    else:
        # Append user msg + assistant placeholder, then overwrite in-place after research
        st.session_state.messages.append(("You", user_q))
        st.session_state.messages.append(("Assistant", "Researching trusted sources…"))
        placeholder_index = len(st.session_state.messages) - 1

        with st.spinner("Searching government sources…"):
            ans, cites = build_research_answer(user_q)

        # In-place update for immediate chat refresh
        st.session_state.messages[placeholder_index] = ("Assistant", ans)
        st.session_state.last_citations = cites

# RENDER HISTORY AFTER submission handling so new Q/A appears immediately
for i, (role, text) in enumerate(st.session_state.messages):
    st.markdown(f"**{role}:** {text}", unsafe_allow_html=False)

# Latest citations section
if st.session_state.last_citations:
    st.markdown("**Sources**")
    for idx, title, url in st.session_state.last_citations:
        st.markdown(f"[{idx}] [{html.escape(title)}]({url})")

st.markdown("---")

# Export
col1, col2, col3 = st.columns(3)

def transcript_txt() -> str:
    lines = ["Immigration Research Chat", "", DISCLAIMER, REFUSAL, ""]
    for role, text in st.session_state.messages:
        lines.append(f"{role}: {text}")
    if st.session_state.last_citations:
        lines += ["", "Sources:"]
        for idx, title, url in st.session_state.last_citations:
            lines.append(f"[{idx}] {title} — {url}")
    return "\n".join(lines)

txt_blob = transcript_txt()
col1.download_button(
    "Download transcript (TXT)",
    data=txt_blob.encode("utf-8"),
    file_name="immigration_chat_transcript.txt",
    mime="text/plain",
)

def transcript_pdf() -> Optional[bytes]:
    if not REPORTLAB_OK:
        return None
    buf = io.BytesIO()
    styles = getSampleStyleSheet()
    story = [
        Paragraph("Immigration Research Chat", styles["Title"]),
        Spacer(1, 12),
        Paragraph(DISCLAIMER, styles["Italic"]),
        Paragraph(REFUSAL, styles["Italic"]),
        Spacer(1, 12),
    ]
    for role, text in st.session_state.messages:
        story.append(Paragraph(f"{role}:", styles["Heading4"]))
        story.append(Paragraph(html.escape(text).replace("\n", "<br/>"), styles["Normal"]))
        story.append(Spacer(1, 8))
    if st.session_state.last_citations:
        story.append(Spacer(1, 12))
        story.append(Paragraph("Sources", styles["Heading3"]))
        for idx, title, url in st.session_state.last_citations:
            story.append(Paragraph(f"[{idx}] {html.escape(title)} — {html.escape(url)}", styles["Normal"]))
    doc = SimpleDocTemplate(buf, pagesize=letter)
    doc.build(story)
    return buf.getvalue()

pdf_bytes = transcript_pdf()
if pdf_bytes:
    col2.download_button(
        "Download transcript (PDF)",
        data=pdf_bytes,
        file_name="immigration_chat_transcript.pdf",
        mime="application/pdf",
    )
else:
    col2.write("PDF export unavailable")

# Clear
if col3.button("Clear chat"):
    st.session_state.messages = []
    st.session_state.last_citations = []
    st.rerun()
