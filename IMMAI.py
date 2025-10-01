#!/usr/bin/env python3
"""
IMMigration RAG Chat — single-file Streamlit app

Purpose
- Answer immigration questions with short, high-level summaries grounded ONLY in primary, reputable sources.
- Automatically searches trusted domains (USCIS, State/NVC, EOIR/DOJ, DHS/CBP/ICE, Federal Register, Congress).
- Extracts pages, ranks relevant passages, composes an answer, and shows inline citations [1], [2], … with links.
- Not legal advice. No attorney–client relationship.

Run
  python -m pip install --upgrade pip
  pip install streamlit duckduckgo-search trafilatura sentence-transformers transformers torch
  # Optional (PDF export of the chat):
  pip install reportlab
  streamlit run imm_rag_chat.py
"""

from __future__ import annotations
import os
import re
import io
import math
import time
import html
import textwrap
import urllib.parse
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

import streamlit as st

# Optional PDF dependency
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    REPORTLAB_OK = True
except Exception:
    REPORTLAB_OK = False

# Search and Retrieval
from duckduckgo_search import DDGS
import trafilatura

# Embeddings + Generator
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import numpy as np

# --------------------------- Config ---------------------------

TRUSTED_DOMAINS = [
    "uscis.gov",
    "travel.state.gov",      # NVC/consular
    "justice.gov",           # EOIR (immigration courts)
    "dhs.gov",
    "cbp.gov",
    "ice.gov",
    "congress.gov",          # statute text
    "federalregister.gov",   # rules
    "govinfo.gov",           # CFR/FR archive
    "whitehouse.gov",        # proclamations (stable enough for history)
    "usa.gov",               # general
]

MAX_RESULTS_PER_DOMAIN = 3
GLOBAL_RESULT_CAP = 12
FETCH_TIMEOUT = 20
CHUNK_CHARS = 1200
TOP_CHUNKS = 5
GEN_MAX_NEW_TOKENS = 350

DISCLAIMER = (
    "**Disclaimer:** Informational only. Not legal advice. No attorney–client relationship. "
    "Citations are to government sources. Laws and policy change."
)

REFUSAL = (
    "Requests to lie, use false documents, evade the law, or commit fraud are refused."
)

INSTRUCTIONS_PROMPT = """You are an immigration information assistant.
Answer ONLY using the provided sources. Quote or paraphrase minimally and concisely.
Do not invent facts. Do not give fees, wait times, or step-by-step legal tactics.
Flag uncertainty explicitly. End with a one-sentence caveat that this is general information, not legal advice.
Write in clear plain English."""

SYSTEM_SUFFIX = " This summary is not legal advice."

# --------------------------- Small helpers ---------------------------

def norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def is_trusted(url: str) -> bool:
    try:
        host = urllib.parse.urlsplit(url).netloc.lower()
        # strip subdomains and port
        host = host.split(":")[0]
    except Exception:
        return False
    return any(host.endswith(dom) for dom in TRUSTED_DOMAINS)

def safe_title(url: str) -> str:
    # Derive a short label if no title available
    try:
        host = urllib.parse.urlsplit(url).netloc
        path = urllib.parse.urlsplit(url).path or "/"
        return f"{host}{path}"
    except Exception:
        return url

def batched(iterable, n):
    it = list(iterable)
    for i in range(0, len(it), n):
        yield it[i:i+n]

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
    # Prefer trusted domains via site: filters; fall back to post-filter
    hits: List[Hit] = []
    q = norm_space(query)
    # Issue multiple queries, one per domain, to steer DDG
    with DDGS() as ddgs:
        for dom in TRUSTED_DOMAINS:
            q_dom = f"site:{dom} {q}"
            for r in ddgs.text(q_dom, max_results=MAX_RESULTS_PER_DOMAIN, region="us-en", safesearch="moderate"):
                url = r.get("href") or r.get("url") or ""
                title = r.get("title") or ""
                body = r.get("body") or ""
                if not url:
                    continue
                if not is_trusted(url):
                    continue
                hits.append(Hit(url=url, title=title, snippet=body))
                if len(hits) >= GLOBAL_RESULT_CAP:
                    break
            if len(hits) >= GLOBAL_RESULT_CAP:
                break
    # Deduplicate by URL
    seen, uniq = set(), []
    for h in hits:
        if h.url in seen:
            continue
        seen.add(h.url)
        uniq.append(h)
    return uniq

def fetch_and_chunk(url: str, title_fallback: str) -> List[Tuple[str, str]]:
    html_doc = None
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

    # naive title parse; trafilatura sometimes returns title in metadata, but we keep fallback
    title = title_fallback or safe_title(url)

    # Chunk by characters to keep ordering
    chunks = []
    for i in range(0, len(text), CHUNK_CHARS):
        seg = text[i:i+CHUNK_CHARS]
        if len(seg) < 240 and i > 0:
            break
        chunks.append((title, seg))
    return [(url, title, c) for _, c in [(title, seg) for seg in [ch[1] for ch in [(title, seg) for seg in [t for _, t in chunks]]]]]

def embed(texts: List[str]) -> np.ndarray:
    # Returns (n, d) array
    vecs = EMBED.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return vecs

def rank_chunks(question: str, docs: List[Tuple[str, str, str]]) -> List[DocChunk]:
    if not docs:
        return []
    qv = embed([question])[0]
    texts = [d[2] for d in docs]
    tv = embed(texts)
    sims = (tv @ qv)  # cosine due to normalized vectors
    scored = []
    for (url, title, chunk), s in zip(docs, sims):
        scored.append(DocChunk(url=url, title=title or safe_title(url), text=chunk, score=float(s)))
    scored.sort(key=lambda x: x.score, reverse=True)
    return scored

def make_prompt(question: str, top_chunks: List[DocChunk], numbered_sources: Dict[str, int]) -> str:
    lines = [INSTRUCTIONS_PROMPT, ""]
    lines.append("SOURCES:")
    for i, ch in enumerate(top_chunks, 1):
        idx = numbered_sources[ch.url]
        snippet = textwrap.shorten(ch.text, width=800, placeholder=" …")
        lines.append(f"[{idx}] {ch.title} :: {ch.url}")
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
        out = LLM.generate(
            **inputs,
            max_new_tokens=GEN_MAX_NEW_TOKENS,
            temperature=0.2,
            do_sample=False,
        )
    return TOK.decode(out[0], skip_special_tokens=True).strip()

def format_citations(urls: List[str]) -> Dict[str, int]:
    # Map unique URL -> citation index
    mapping: Dict[str, int] = {}
    i = 1
    for u in urls:
        if u not in mapping:
            mapping[u] = i
            i += 1
    return mapping

def build_research_answer(question: str) -> Tuple[str, List[Tuple[int, str, str]]]:
    # Search
    hits = ddg_trusted_search(question)
    if not hits:
        return ("No qualifying primary sources found on trusted domains. Rephrase and focus the question.", [])

    # Fetch + chunk
    docs: List[Tuple[str, str, str]] = []
    for h in hits:
        chs = fetch_and_chunk(h.url, h.title or safe_title(h.url))
        # Use first few chunks per doc to reduce noise
        docs.extend(chs[:3])

    if not docs:
        return ("Sources were found but could not be parsed. Try a more specific question.", [])

    # Rank
    ranked = rank_chunks(question, docs)
    top = ranked[:TOP_CHUNKS]

    # Number sources by URL
    numbered = format_citations([c.url for c in top])

    # Compose prompt
    prompt = make_prompt(question, top, numbered)

    # Generate
    raw = generate_answer(prompt)
    if not raw:
        return ("Unable to compose a grounded summary from the retrieved sources.", [])

    # Append hard disclaimer
    answer = norm_space(raw) + SYSTEM_SUFFIX

    # Build citation table [(index, title, url)]
    citations = []
    # Keep order by index
    for url, idx in sorted(numbered.items(), key=lambda kv: kv[1]):
        # find a title from our ranked list
        title = next((c.title for c in top if c.url == url), None) or safe_title(url)
        citations.append((idx, title, url))

    return (answer, citations)

# --------------------------- UI ---------------------------

st.set_page_config(page_title="Immigration Research Chat", layout="centered")

if "messages" not in st.session_state:
    st.session_state.messages = []  # list[(role, text)]
if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""
if "last_citations" not in st.session_state:
    st.session_state.last_citations = []

st.title("Immigration Research Chat")
st.markdown(DISCLAIMER)
st.markdown(REFUSAL)
st.markdown("---")

# Chat history
for role, text in st.session_state.messages:
    st.markdown(f"**{role}:** {text}")

# Input
with st.form("ask", clear_on_submit=True):
    q = st.text_area("Ask about immigration (general topics, forms, concepts).", height=120)
    submitted = st.form_submit_button("Send")

if submitted:
    user_q = norm_space(q)
    if not user_q:
        st.session_state.messages.append(("System", "Enter a question."))
    else:
        st.session_state.messages.append(("You", user_q))
        with st.spinner("Researching trusted sources…"):
            ans, cites = build_research_answer(user_q)
        st.session_state.last_answer = ans
        st.session_state.last_citations = cites
        st.session_state.messages.append(("Assistant", ans))

# Latest answer (with citations list)
if st.session_state.last_answer:
    st.markdown("**Sources**")
    for idx, title, url in st.session_state.last_citations:
        st.markdown(f"[{idx}] [{html.escape(title)}]({url})")

st.markdown("---")

# Export
col1, col2, col3 = st.columns(3)

# TXT export
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

# PDF export (optional)
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
    st.session_state.last_answer = ""
    st.session_state.last_citations = []
    st.rerun()
