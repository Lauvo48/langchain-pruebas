import os
import streamlit as st
from PyPDF2 import PdfReader


from langchain.text_splitter import RecursiveCharacterTextSplitter # partir texto
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain.vectorstores import FAISS # guardar y cargar embeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

import google.generativeai as genai
import constants # archivo con la clave de API

## configuracion incial api
st.set_page_config(page_title="Chat con PDF", page_icon="📚")
st.header("📚 Chat con tus documentos PDF")


## carpeta indices
INDEX_DIR = "faiss_index"

## funcion incial embeddings, transforma texto a vectores numericos
def get_embeddings():
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=constants.GOOGLE_API_KEY)

## modelo llm
def get_llm():
    return GoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.3, google_api_key=constants.GOOGLE_API_KEY)


### Carga FAISS 
def load_vectorstore(index_dir: str, embeddings: GoogleGenerativeAIEmbeddings):
    if os.path.isdir(index_dir):
        return FAISS.load_local(
            index_dir, embeddings, allow_dangerous_deserialization=True)
    return None

## procesamiento pdf, esta funcion extrae el texto de los PDFs subidos y los concatena
## no funciona si el pdf es solo imagenes
def get_pdf_text(uploaded_files) -> str:
    text_parts = []
    for pdf in uploaded_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                text_parts.append(page_text)
    return "\n".join(text_parts)


def get_text_chunks(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=180,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_text(text)

### indexar o actualizar el indice, crea un nuevo indice o actualiza el existente
def build_or_update_index(chunks, embeddings, index_dir: str):
    vs = FAISS.from_texts(chunks, embedding=embeddings)
    vs.save_local(index_dir)
    return vs

## prompt y cadena de conversación (define el prompt y la cadena de preguntas y respuestas)

def make_prompt():
    prompt_template = """
Eres un asistente que responde con base EXCLUSIVA en el contexto suministrado.
- Si la respuesta no está en el contexto, di claramente: "La respuesta no está disponible en el contexto".
- No inventes datos.

Contexto:
{context}

Pregunta:
{question}

Respuesta (clara y con citas breves de ser útil):
"""
    return PromptTemplate(
        template=prompt_template.strip(),
        input_variables=["context", "question"]
    )


## Extrae textos mas relevantes y genera respuesta
def get_qa_chain(retriever):
    llm = get_llm()
    prompt = make_prompt()

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )

## Cargue de PDFs

uploaded_pdfs = st.file_uploader(
    "Sube tus PDFs y luego pregunta 👇", accept_multiple_files=True, type=["pdf"]
)

## preparacion embeddings y vectorstore
embeddings = get_embeddings()


## cargue indice
vectorstore = load_vectorstore(INDEX_DIR, embeddings)

# Indexación (si hay PDFs nuevos)
if uploaded_pdfs:
    with st.spinner("📖 Extrayendo texto y creando el índice..."):
        raw_text = get_pdf_text(uploaded_pdfs)
        if not raw_text.strip():
            st.warning("No se pudo extraer texto de los PDFs. Revisa que no sean solo imágenes.")
        else:
            chunks = get_text_chunks(raw_text)
            vs = build_or_update_index(chunks, embeddings, INDEX_DIR)
            st.session_state["vectorstore"] = vs  
            st.success(f"Índice creado con {len(chunks)} fragmentos.")



# caja de pregunta y respuesta (boton)

with st.form("qa_form"):
    user_q = st.text_input("Pregunta sobre tus documentos:", key="q")
    ask = st.form_submit_button("Preguntar")

if ask:
    vs = st.session_state.get("vectorstore")          # <<<<< lee del estado
    if not vs:
        st.error("Primero sube y procesa tus PDFs para crear el índice.")
    elif not user_q.strip():
        st.warning("Escribe una pregunta.")
    else:
        with st.spinner("🧠 Buscando en tus documentos..."):
            retriever = vs.as_retriever(search_kwargs={"k": 4})
            qa = get_qa_chain(retriever)
            out = qa({"query": user_q})

        st.subheader("Respuesta")
        st.markdown((out.get("result", "") or "").strip())
