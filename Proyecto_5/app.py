# app.py — Chat con PDF (RAG) usando LangChain + OpenAI + FAISS
# RAG (Retrieval-Augmented Generation): combinar LLM con conocimiento propio (PDFs)

# ===============================
# IMPORTS
# ===============================

# import: importa todo el módulo o paquete completo
# from: importa solo lo necesario de un módulo o paquete

import os
import streamlit as st
from typing import List, Any
from PyPDF2 import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document

import constants  # Debe contener OPENAI_API_KEY = "..."

# ===============================
# CONFIGURACIÓN DE LA APP
# ===============================

st.set_page_config(page_title="📚 Chat con tus PDF (OpenAI RAG)", page_icon="📚")
st.header("📚 Chat con tus documentos PDF (OpenAI)")

INDEX_DIR = "faiss_index"  # carpeta local para persistir el índice

# ===============================
# FACTORÍAS CACHEADAS (OpenAI)
# ===============================

@st.cache_resource
def get_embeddings():
    """Crea/recupera embeddings de OpenAI (una sola vez)."""
    api_key = getattr(constants, "OPENAI_API_KEY", None)
    if not api_key:
        st.error("Falta OPENAI_API_KEY en constants.py")
        st.stop()
    # Modelo rápido y económico
    return OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)

@st.cache_resource
def get_llm():
    """Crea/recupera el LLM de OpenAI (una sola vez)."""
    api_key = getattr(constants, "OPENAI_API_KEY", None)
    if not api_key:
        st.error("Falta OPENAI_API_KEY en constants.py")
        st.stop()
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=api_key)

# ===============================
# UTILIDADES PARA PDFs
# ===============================

def extract_text_from_pdfs(uploaded_files: List[Any]) -> List[Document]:
    """Extrae texto por página y devuelve Documents con metadata (source, page)."""
    docs: List[Document] = []
    for pdf in uploaded_files:
        try:
            reader = PdfReader(pdf)
            # Para demo: si quieres, limita páginas: reader.pages[:5]
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    docs.append(
                        Document(
                            page_content=page_text,
                            metadata={"source": pdf.name, "page": i + 1}
                        )
                    )
        except Exception as e:
            st.warning(f"No se pudo leer {getattr(pdf, 'name', 'archivo')}: {e}")
    return docs

def chunk_documents(docs: List[Document]) -> List[Document]:
    """Parte documentos en chunks grandes (menos llamadas a embeddings)."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,      # grande = menos requests
        chunk_overlap=120,    # continuidad razonable
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)

# ===============================
# PROMPT PARA RAG (control de alucinación + citas)
# ===============================

def make_prompt():
    prompt_template = """
Responde SOLO con base en el contexto. Si la respuesta no está en el contexto,
di exactamente: "La respuesta no está disponible en el contexto".
Incluye, al final, citas cortas con (archivo:página).

Contexto:
{context}

Pregunta:
{question}

Respuesta:
"""
    return PromptTemplate(
        template=prompt_template.strip(),
        input_variables=["context", "question"]
    )

def build_qa_chain(retriever):
    """Crea la cadena de QA con recuperación (RetrievalQA) y nuestro prompt."""
    llm = get_llm()
    prompt = make_prompt()
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )

# ===============================
# CARGA / CREACIÓN DE ÍNDICE
# ===============================

def load_vectorstore(index_dir: str, embeddings):
    try:
        if os.path.isdir(index_dir):
            return FAISS.load_local(
                index_dir,
                embeddings,
                allow_dangerous_deserialization=True  # << Solo si confías en ese índice
            )
    except Exception as e:
        st.warning(f"No fue posible cargar el índice existente ({e}). Se creará uno nuevo.")
    return None


def build_and_save_index(docs: List[Document], embeddings, index_dir: str):
    """Crea FAISS a partir de documentos ya fragmentados, persiste y retorna."""
    vs = FAISS.from_documents(docs, embedding=embeddings)
    vs.save_local(index_dir)
    return vs

# ===============================
# UI: CARGA DE PDFs
# ===============================

uploaded_pdfs = st.file_uploader(
    "Sube tus PDFs y luego pregunta 👇",
    accept_multiple_files=True,
    type=["pdf"]
)

embeddings = get_embeddings()
vectorstore = load_vectorstore(INDEX_DIR, embeddings)

if uploaded_pdfs:
    with st.spinner("📖 Extrayendo texto y creando el índice..."):
        raw_docs = extract_text_from_pdfs(uploaded_pdfs)
        if not raw_docs:
            st.warning("No se pudo extraer texto. (Si tus PDFs son escaneados, requieren OCR).")
        else:
            chunks = chunk_documents(raw_docs)
            vs = build_and_save_index(chunks, embeddings, INDEX_DIR)
            st.session_state["vectorstore"] = vs
            st.success(f"Índice creado con {len(chunks)} fragmentos.")

# ===============================
# UI: PREGUNTA Y RESPUESTA
# ===============================

st.markdown("---")
user_q = st.text_input("Pregunta sobre tus documentos:")

if st.button("Preguntar"):
    vs = st.session_state.get("vectorstore") or vectorstore
    if not vs:
        st.error("Primero sube y procesa tus PDFs para crear el índice.")
    elif not user_q.strip():
        st.warning("Escribe una pregunta.")
    else:
        with st.spinner("🧠 Buscando en tus documentos..."):
            retriever = vs.as_retriever(search_kwargs={"k": 4})
            qa = build_qa_chain(retriever)
            out = qa({"query": user_q})

        st.subheader("Respuesta")
        st.markdown((out.get("result", "") or "").strip())

        # Mostrar fuentes con snippet
        srcs = out.get("source_documents") or []
        if srcs:
            st.markdown("**Fuentes:**")
            for i, s in enumerate(srcs, start=1):
                meta = s.metadata or {}
                snippet = (s.page_content or "").strip().replace("\n", " ")
                st.caption(f"{i}. {meta.get('source')} (pág. {meta.get('page')}) — “{snippet[:180]}...”")

