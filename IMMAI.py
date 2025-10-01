#!/usr/bin/env python3
"""
Immigration Information Chat (single-file, multilingual, with local LLM + KB fallback)

Informational use only. Not legal advice. No attorney–client relationship.
Run:
  pip install streamlit reportlab transformers sentencepiece torch
  streamlit run immigration_chat.py
"""

from __future__ import annotations
import io
import re
import unicodedata
import urllib.parse
from typing import List, Dict, Tuple

import streamlit as st
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ============================== UI STRINGS ==============================
L: Dict[str, Dict[str, str]] = {
    "en": {
        "app_title": "Immigration Information Chat",
        "disclaimer": "**Disclaimer:** Informational only. Not legal advice. No attorney–client relationship.",
        "input_label": "Type your question",
        "send": "Send",
        "clear": "Clear chat",
        "download_pdf": "Download transcript (PDF)",
        "download_txt": "Download transcript (TXT)",
        "mailto_btn": "Open email to send transcript",
        "lang_picker": "Choose language / Elija idioma / Escolha idioma",
        "system_name": "System",
        "you_name": "You",
        "bot_name": "Assistant",
        "fallback": "I provide general information only. Review official USCIS/NVC instructions and consider consulting an accredited representative.",
        "refusal": "I will not assist with lying, false documents, evasion, or fraud. Use official USCIS resources and qualified help.",
        "bad_request": "Enter a question.",
        "footer_note": "Stable references only. No fees, timelines, or evolving policy.",
        "mailto_subject": "Immigration Q&A Transcript",
        "mailto_hint": "If your email client truncates long messages, copy and paste the transcript."
    },
    "es": {
        "app_title": "Chat de Información Migratoria",
        "disclaimer": "**Aviso:** Solo informativo. No es asesoría legal. No crea relación abogado–cliente.",
        "input_label": "Escriba su pregunta",
        "send": "Enviar",
        "clear": "Borrar chat",
        "download_pdf": "Descargar transcripción (PDF)",
        "download_txt": "Descargar transcripción (TXT)",
        "mailto_btn": "Abrir correo para enviar transcripción",
        "lang_picker": "Elija idioma / Choose language / Escolha idioma",
        "system_name": "Sistema",
        "you_name": "Usted",
        "bot_name": "Asistente",
        "fallback": "Brindo información general. Revise las instrucciones oficiales de USCIS/NVC y considere consultar un representante acreditado.",
        "refusal": "No ayudaré con mentir, documentos falsos, evasión o fraude. Use recursos oficiales de USCIS y ayuda calificada.",
        "bad_request": "Escriba una pregunta.",
        "footer_note": "Solo referencias estables. Sin tarifas, tiempos ni políticas cambiantes.",
        "mailto_subject": "Transcripción de Preguntas Migratorias",
        "mailto_hint": "Si su correo recorta mensajes largos, copie y pegue la transcripción."
    },
    "pt": {
        "app_title": "Chat de Informações de Imigração",
        "disclaimer": "**Aviso:** Somente informativo. Não é aconselhamento jurídico. Não cria relação advogado–cliente.",
        "input_label": "Digite sua pergunta",
        "send": "Enviar",
        "clear": "Limpar conversa",
        "download_pdf": "Baixar transcrição (PDF)",
        "download_txt": "Baixar transcrição (TXT)",
        "mailto_btn": "Abrir e-mail para enviar transcrição",
        "lang_picker": "Escolha idioma / Choose language / Elija idioma",
        "system_name": "Sistema",
        "you_name": "Você",
        "bot_name": "Assistente",
        "fallback": "Forneço informações gerais. Revise as instruções oficiais do USCIS/NVC e considere consultar um representante credenciado.",
        "refusal": "Não ajudarei com mentiras, documentos falsos, evasão ou fraude. Use recursos oficiais do USCIS e ajuda qualificada.",
        "bad_request": "Digite uma pergunta.",
        "footer_note": "Apenas referências estáveis. Sem taxas, prazos ou políticas em mudança.",
        "mailto_subject": "Transcrição de Perguntas de Imigração",
        "mailto_hint": "Se o e-mail encurtar mensagens longas, copie e cole a transcrição."
    },
}

# ============================== KNOWLEDGE BASE ==============================
KB: List[Dict] = [
    {
        "tags": ["i-130", "spouse", "marriage", "petitioner", "relative", "family petition", "petición familiar", "petição familiar"],
        "en": "Form I-130 establishes a qualifying family relationship. Approval alone does not grant status. Next steps are Adjustment of Status (inside, if eligible) or Consular Processing (abroad).",
        "es": "El Formulario I-130 establece una relación familiar calificada. La aprobación por sí sola no otorga estatus. Los siguientes pasos son Ajuste de Estatus (dentro, si elegible) o Proceso Consular (en el exterior).",
        "pt": "O Formulário I-130 comprova vínculo familiar qualificado. A aprovação não concede status por si só. Os próximos passos são Ajuste de Status (dentro, se elegível) ou Processamento Consular (no exterior).",
    },
    {
        "tags": ["i-485", "adjustment of status", "aos", "ajuste", "ajuste de estatus"],
        "en": "Adjustment of Status (I-485) is for eligible applicants in the U.S., often requiring a lawful entry. Inadmissibility can require waivers.",
        "es": "El Ajuste de Estatus (I-485) es para solicitantes elegibles dentro de EE. UU., a menudo requiriendo una entrada lícita. La inadmisibilidad puede requerir exenciones.",
        "pt": "O Ajuste de Status (I-485) é para elegíveis nos EUA, geralmente exigindo entrada lícita. A inadmissibilidade pode exigir perdões.",
    },
    {
        "tags": ["consular processing", "nvc", "ds-260", "consulado", "processo consular"],
        "en": "Consular Processing uses NVC, fees, civil documents, and DS-260, then an interview. Unlawful presence or prior orders can require waivers.",
        "es": "El Proceso Consular usa NVC, tarifas, documentos civiles y DS-260, luego una entrevista. La presencia ilegal u órdenes previas pueden requerir exenciones.",
        "pt": "O Processamento Consular usa NVC, taxas, documentos civis e DS-260, seguido de entrevista. Presença ilegal ou ordens prévias podem exigir perdões.",
    },
    {
        "tags": ["i-601a", "provisional waiver", "provisional", "601a"],
        "en": "I-601A provisional waiver is generally for applicants inside the U.S. with a qualifying USC/LPR spouse or parent. Not filed abroad.",
        "es": "La exención provisional I-601A es generalmente para solicitantes dentro de EE. UU. con cónyuge o padre/madre ciudadano o LPR calificador. No se presenta en el exterior.",
        "pt": "O perdão provisório I-601A é geralmente para requerentes nos EUA com cônjuge ou pai/mãe cidadão/LPR qualificado. Não é apresentado no exterior.",
    },
    {
        "tags": ["i-601", "waiver", "extreme hardship", "exención", "perdão"],
        "en": "I-601 can waive certain inadmissibility (including unlawful presence) by showing extreme hardship to a qualifying USC/LPR spouse or parent, usually at the consular stage.",
        "es": "El I-601 puede eximir ciertas causales de inadmisibilidad (incluida la presencia ilegal) demostrando dificultad extrema a cónyuge o padre/madre ciudadano/LPR calificador, usualmente en etapa consular.",
        "pt": "O I-601 pode dispensar certas inadmissibilidades (inclusive presença ilegal) mostrando dificuldade extrema a cônjuge ou pai/mãe cidadão/LPR qualificado, geralmente na fase consular.",
    },
    {
        "tags": ["i-212", "permission to reapply", "removal", "deportation", "reingreso ilegal", "reentrada ilegal"],
        "en": "I-212 requests permission to reapply for admission after removal or certain orders. It may be required in addition to other waivers.",
        "es": "El I-212 solicita permiso para volver a pedir admisión tras una remoción u órdenes. Puede requerirse además de otras exenciones.",
        "pt": "O I-212 solicita permissão para voltar a requerer admissão após remoção ou certas ordens. Pode ser necessário além de outros perdões.",
    },
    {
        "tags": ["212(a)(9)(c)", "permanent bar", "illegal reentry", "barra permanente"],
        "en": "INA 212(a)(9)(C) can trigger a permanent bar for illegal reentry after removal or after >1 year of unlawful presence. Options can be limited; long-term strategies may apply.",
        "es": "INA 212(a)(9)(C) puede generar barra permanente por reingreso ilegal tras remoción o tras >1 año de presencia ilegal. Las opciones pueden ser limitadas; pueden aplicar estrategias a largo plazo.",
        "pt": "INA 212(a)(9)(C) pode gerar barreira permanente por reentrada ilegal após remoção ou após >1 ano de presença ilegal. As opções podem ser limitadas; estratégias de longo prazo podem se aplicar.",
    },
    {
        "tags": ["n-600", "citizenship", "derivation", "derivación", "derivação", "320", "§320"],
        "en": "N-600 documents citizenship acquired or derived; for INA §320 derivation: LPR child, under 18, residing in the U.S., in the legal and physical custody of the citizen parent.",
        "es": "El N-600 documenta ciudadanía adquirida o derivada; para derivación INA §320: menor LPR, menor de 18, residente en EE. UU., bajo custodia legal y física del padre/madre ciudadano.",
        "pt": "O N-600 documenta cidadania adquirida ou derivada; para derivação INA §320: menor LPR, menor de 18, residente nos EUA, sob custódia legal e física do pai/mãe cidadão.",
    },
    {
        "tags": ["crba", "consular report of birth abroad", "passport at post", "ciudadanía al nacer"],
        "en": "CRBA can document citizenship at birth abroad when transmission requirements are met. Often paired with first U.S. passport at post.",
        "es": "El CRBA puede documentar ciudadanía al nacer fuera de EE. UU. cuando se cumplen los requisitos de transmisión. A menudo con el primer pasaporte en el consulado.",
        "pt": "O CRBA pode documentar cidadania ao nascer no exterior quando os requisitos de transmissão são atendidos. Geralmente com o primeiro passaporte no consulado.",
    },
    {
        "tags": ["parole", "inspection", "lawful entry", "entrada lícita", "paroled", "inspected"],
        "en": "Many AOS paths rely on a lawful entry (inspection or parole). Manner of last entry affects eligibility.",
        "es": "Muchas vías de AOS dependen de una entrada lícita (inspección o parole). La forma de la última entrada afecta la elegibilidad.",
        "pt": "Muitas vias de AOS dependem de entrada lícita (inspeção ou parole). A forma da última entrada afeta a elegibilidade.",
    },
    {
        "tags": ["ead", "i-765", "work authorization", "autorización de empleo", "autorização de trabalho"],
        "en": "Work authorization (I-765) may be available when a principal application is pending and the category permits it. Eligibility depends on the underlying basis.",
        "es": "La autorización de empleo (I-765) puede estar disponible cuando una solicitud principal está pendiente y la categoría lo permite. La elegibilidad depende de la base subyacente.",
        "pt": "A autorização de trabalho (I-765) pode estar disponível quando um pedido principal está pendente e a categoria permite. A elegibilidade depende da base subjacente.",
    },
]

# ============================== LLM LOAD (CACHED) ==============================
@st.cache_resource(show_spinner=False)
def load_llm():
    name = "google/flan-t5-base"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSeq2SeqLM.from_pretrained(name)
    return tok, model

TOK, LLM = load_llm()

# ============================== GUARDRAILS ==============================
PROHIBITED_PATTERNS = [
    r"\b(lie|forg(e|ing)|fake|counterfeit|cheat|bypass|evade|bribe|falsif(y|icar|icar)|fraud)\b",
    r"how to.*(misrepresent|hide.*arrest|hide.*marriage|work illegally|overstay)",
]

def normalize(text: str) -> str:
    t = text.lower()
    t = "".join(c for c in unicodedata.normalize("NFKD", t) if not unicodedata.combining(c))
    t = re.sub(r"\s+", " ", t).strip()
    return t

def detect_lang(s: str) -> str:
    s2 = f" {normalize(s)} "
    es_hits = sum(w in s2 for w in [" el ", " la ", " de ", " que ", " usted ", " gracias ", " dónde ", " estoy ", " hijo ", " cónyuge "])
    pt_hits = sum(w in s2 for w in [" de ", " que ", " você ", " voce ", " obrigado ", " onde ", " filho ", " cônjuge ", " conjuge "])
    if es_hits > pt_hits and es_hits >= 2:
        return "es"
    if pt_hits > es_hits and pt_hits >= 2:
        return "pt"
    return "en"

def blocked_request(q: str) -> bool:
    qn = normalize(q)
    return any(re.search(pat, qn) for pat in PROHIBITED_PATTERNS)

def search_kb(question: str, lang: str, k: int = 3) -> List[str]:
    qn = normalize(question)
    scored: List[Tuple[int, Dict]] = []
    for entry in KB:
        score = sum(tag in qn for tag in entry["tags"])
        if score:
            scored.append((score, entry))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [entry[lang] for _, entry in scored[:k]]

def generate_llm_answer(question: str, lang: str) -> str:
    sys_instr = {
        "en": ("You are an immigration information assistant. Provide high-level, non-legal, general guidance only. "
               "No fees, timelines, or evolving policy. Refuse fraud/illegality. Be concise."),
        "es": ("Eres un asistente de información migratoria. Proporciona orientación general y no legal. "
               "Sin tarifas, tiempos ni políticas cambiantes. Rechaza fraude/ilegalidad. Sé conciso."),
        "pt": ("Você é um assistente de informações de imigração. Forneça orientação geral e não jurídica. "
               "Sem taxas, prazos ou políticas em mudança. Recuse fraude/ilegalidade. Seja conciso."),
    }[lang]
    prompt = f"{sys_instr}\n\nQuestion:\n{question}\n\nAnswer:"
    inputs = TOK(prompt, return_tensors="pt", truncation=True, max_length=512)
    out = LLM.generate(**inputs, max_new_tokens=220, temperature=0.2, do_sample=False)
    return TOK.decode(out[0], skip_special_tokens=True).strip()

def compose_answer(question: str, lang: str) -> str:
    if not question or not question.strip():
        return L[lang]["bad_request"]
    if blocked_request(question):
        return L[lang]["refusal"]
    try:
        ans = generate_llm_answer(question, lang)
        if ans:
            return f"{ans} {L[lang]['fallback']}"
    except Exception:
        pass
    hits = search_kb(question, lang)
    if hits:
        return f"{' '.join(hits)} {L[lang]['fallback']}"
    return L[lang]["fallback"]

# ============================== EXPORTS ==============================
def pdf_from_transcript(messages: List[Tuple[str, str]], lang: str) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter)
    styles = getSampleStyleSheet()
    story: List = [
        Paragraph(L[lang]["app_title"], styles["Title"]),
        Spacer(1, 12),
        Paragraph(L[lang]["disclaimer"], styles["Italic"]),
        Spacer(1, 12),
    ]
    for role, text in messages:
        story.append(Paragraph(f"{role}:", styles["Heading4"]))
        story.append(Paragraph(text.replace("\n", "<br/>"), styles["Normal"]))
        story.append(Spacer(1, 8))
    story.append(Spacer(1, 12))
    story.append(Paragraph(L[lang]["footer_note"], styles["Italic"]))
    doc.build(story)
    return buf.getvalue()

def txt_from_transcript(messages: List[Tuple[str, str]], lang: str) -> str:
    lines = [L[lang]["app_title"], "", L[lang]["disclaimer"], ""]
    for role, text in messages:
        lines.append(f"{role}: {text}")
    lines += ["", L[lang]["footer_note"]]
    return "\n".join(lines)

# ============================== APP ==============================
st.set_page_config(page_title="Immigration Chat", layout="centered")

if "lang" not in st.session_state:
    st.session_state.lang = "en"
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of (role_display, text)

lang_choice = st.selectbox(
    L["en"]["lang_picker"],
    ["English", "Español", "Português"],
    index={"en": 0, "es": 1, "pt": 2}[st.session_state.lang],
)
st.session_state.lang = {"English": "en", "Español": "es", "Português": "pt"}[lang_choice]
lang = st.session_state.lang

st.title(L[lang]["app_title"])
st.markdown(L[lang]["disclaimer"])
st.markdown("---")

for role, text in st.session_state.messages:
    st.markdown(f"**{role}:** {text}")

with st.form("chat_form", clear_on_submit=True):
    q = st.text_area(L[lang]["input_label"], height=120, key="q_input")
    submitted = st.form_submit_button(L[lang]["send"])

if submitted:
    user_lang = detect_lang(q) if q.strip() else lang
    if len(st.session_state.messages) == 0:
        lang = user_lang
        st.session_state.lang = user_lang
    if not q.strip():
        st.session_state.messages.append((L[lang]["system_name"], L[lang]["bad_request"]))
    else:
        st.session_state.messages.append((L[lang]["you_name"], q))
        answer = compose_answer(q, lang)
        st.session_state.messages.append((L[lang]["bot_name"], answer))
    st.experimental_rerun()

c1, c2, c3, c4 = st.columns(4)
if c1.button(L[lang]["clear"]):
    st.session_state.messages = []
    st.experimental_rerun()

pdf_bytes = pdf_from_transcript(st.session_state.messages, lang)
c2.download_button(L[lang]["download_pdf"], data=pdf_bytes, file_name="immigration_chat_transcript.pdf", mime="application/pdf")

txt_blob = txt_from_transcript(st.session_state.messages, lang)
c3.download_button(L[lang]["download_txt"], data=txt_blob.encode("utf-8"), file_name="immigration_chat_transcript.txt", mime="text/plain")

subject = urllib.parse.quote(L[lang]["mailto_subject"])
body = urllib.parse.quote(txt_blob[:1800])
mail_href = f"mailto:?subject={subject}&body={body}"
c4.markdown(f"[{L[lang]['mailto_btn']}]({mail_href})")

st.caption(L[lang]["footer_note"])
