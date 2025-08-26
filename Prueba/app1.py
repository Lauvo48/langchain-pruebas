
### Evaluador automático de solicitudes de crédito

# Este script implementa una mini-aplicación web (con Streamlit) que:

# 1 Recibe cartas de solicitantes (TXT o PDF).
# 2 Usa un LLM ( OpenAI) para EXTRAER variables, estructuradas (ingresos, monto solicitado, edad).....
# 3 Aplica 8 reglas de negocio y decide: APROBADO o RECHAZADO.

# Diseño elegido:

# - Streamlit para  rápida subida de archivos y visualización.

# - PyPDF2 para extraer texto de PDFs (sin OCR; si el PDF es imagen,
#   no habrá texto).

# - LangChain para orquestar prompts y el cliente del modelo .

# - OpenAI como proveedor del LLM (modelo gpt-4o-mini) 

import re          # Expresiones regulares JSON
import json        # cargar estructuras JSON 
import os          # utilidades 
import streamlit as st             # framework APIS
from PyPDF2 import PdfReader       # Lector de PDF para extraer texto
from langchain.prompts import PromptTemplate  # Plantillas de prompts 
from langchain_openai import ChatOpenAI      # LangChain para modelos de chat de OpenAI
import constants                   # key

## Configurar

# Configuración de la página de Streamlit:
# ------------------------------------------------------------
st.set_page_config(page_title="Evaluador de crédito", page_icon="✅")

# Título grande que aparece arriba de la app
st.header("✅ Evaluador automático de solicitudes de crédito")


# LLM

def get_llm():
    """
    Crea y devuelve una instancia del modelo de lenguaje (LLM) a través de LangChain.

    ¿Por qué así?
    - Centralizamos la inicialización del LLM en una función: más limpio y testeable.
    - Leemos la API Key desde 'constants' para no quemarla en el código fuente.
    - Si falta la clave, detenemos la app de forma amigable con un mensaje (st.stop()).

    Parámetros clave:
    - model="gpt-4o-mini": modelo de OpenAI con buena relación costo/latencia.
    - temperature=0: buscamos respuestas deterministas (sin "creatividad") para
      minimizar alucinaciones y asegurar que devuelva JSON estable.
    """
    api_key = getattr(constants, "OPENAI_API_KEY", None)  # lee la API key desde constants.py
    if not api_key:
        # Mensaje de error visible en UI y se detiene la ejecución para evitar fallos posteriores
        st.error("⚠️ Falta OPENAI_API_KEY en constants.py")
        st.stop()
    # Devuelve un wrapper de LangChain que sabe hablar con OpenAI Chat Completions
    return ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)

# Prompt 

# PromptTemplate define una plantilla con un "hueco" {texto}.
# Le pedimos al LLM que:
# - Extraiga campos específicos.
# - Devuelva SOLO un JSON válido (sin explicaciones, sin bloques ```json).
# Esto facilita 

LLM_PROMPT = PromptTemplate.from_template("""
Extrae del texto los siguientes campos y responde SOLO con un JSON **válido**:

{{
  "income": number|null,
  "requested": number|null,
  "age": number|null,
  "experience": number|null,
  "owns_business": true|false|null,
  "delinquency": true|false|null,
  "active_credits": number|null,
  "credit_rating": "excelente"|"buena"|"regular"|"mala"|null,
  "rejections": number|null
}}

No incluyas explicaciones, ni texto extra, ni ```json. SOLO el JSON plano.

TEXTO:
\"\"\"{texto}\"\"\"
""")


# FUNCIONES NECESARIAS PARA APP

def read_text(upload) -> str:
    """
    Extrae el contenido textual de un archivo subido por el usuario.

    - Si es .txt: se decodifica como UTF-8 (ignorando errores de codificación).
    - Si es .pdf: se usa PyPDF2 para concatenar el texto de todas las páginas.
      *Limitación*: PyPDF2 no hace OCR; si el PDF es una imagen escaneada, no habrá texto.

    Retorna:
      str con el texto completo (o cadena vacía si no se pudo).
    """
    name = upload.name.lower()
    if name.endswith(".txt"):
        return upload.read().decode("utf-8", errors="ignore")
    if name.endswith(".pdf"):
        reader = PdfReader(upload)
        # p.extract_text() devuelve None si la página no tiene capa de texto
        return "\n".join([p.extract_text() or "" for p in reader.pages])
    return ""

def extract_json_safe(raw: str) -> dict:
    """
    Intenta aislar el primer bloque JSON válido de la respuesta del modelo.

    ¿Por qué es necesario?
    - A veces (aunque lo pedimos) los LLMs devuelven texto extra antes o después del JSON.
    - Primero intentamos cargar 'raw' entero con json.loads().
    - Si falla, usamos una regex (re.search) para capturar el PRIMER bloque {...}
      y lo volvemos a intentar con json.loads().

    - El flag re.DOTALL permite que '.' en la regex capture también saltos de línea,
      necesario para JSON multi-línea.

    Devuelve:
      dict parsed (si se pudo) o {} si no encontramos JSON válido.
    """
    try:
        return json.loads(raw)  # Camino feliz: el modelo cumplió y devolvió JSON puro
    except:
        match = re.search(r"\{.*\}", raw, re.DOTALL)  # Busca el primer bloque que empiece con { y termine con }
        if match:
            return json.loads(match.group())
    return {}

def eval_rules(data: dict) -> bool:
    """
    Aplica las 8 reglas de negocio al dict con las variables extraídas.

    Estructura esperada en 'data':
      - income (float o int): ingresos mensuales en COP.
      - requested (float o int): monto solicitado en COP.
      - age (int): edad en años.
      - experience (float o int): años de experiencia laboral.
      - owns_business (bool): True si tiene negocio propio.
      - delinquency (bool): True si tuvo mora en los últimos 6 meses.
      - active_credits (int): número de créditos activos.
      - credit_rating (str): "excelente" | "buena" | "regular" | "mala".
      - rejections (int): rechazos en los últimos 12 meses.

    Regresa:
      True si CUMPLE TODAS las reglas (APROBADO), False en caso contrario (RECHAZADO).

    Nota:
      - data.get(...) evita excepciones si el campo no existe (None en vez de KeyError).
      - El uso de 'and' en cada check fuerza que haya valor y que cumpla la condición.
    """
    checks = []
    # Regla 1: Ingresos > 1.000.000 COP
    checks.append(data.get("income") and data["income"] > 1_000_000)
    # Regla 2: NO tener mora en últimos 6 meses (delinquency == False)
    checks.append(data.get("delinquency") is False)
    # Regla 3: Edad mínima 21 años
    checks.append(data.get("age") and data["age"] >= 21)
    # Regla 4: Monto solicitado <= 30% de ingresos
    checks.append(data.get("requested") and data.get("income") and data["requested"] <= 0.3 * data["income"])
    # Regla 5: >= 1 año de experiencia O negocio propio
    checks.append((data.get("experience") and data["experience"] >= 1) or data.get("owns_business"))
    # Regla 6: No más de 2 créditos activos
    checks.append(data.get("active_credits") is not None and data["active_credits"] <= 2)
    # Regla 7: Calificación "Buena" o "Excelente"
    checks.append(data.get("credit_rating") in ["buena", "excelente"])
    # Regla 8: No más de 2 rechazos últimos 12 meses
    checks.append(data.get("rejections") is not None and data["rejections"] <= 2)
    # all(checks) => True solo si TODAS las condiciones son True
    return all(checks)

# INTERFAZ

# Componente de subida de archivos:

uploads = st.file_uploader("Sube cartas (TXT o PDF)", type=["txt", "pdf"], accept_multiple_files=True)

# Botón importante
if st.button("Evaluar solicitudes"):
    if not uploads:
        # Validación básica de UX: nada que procesar
        st.error("⚠️ Sube al menos un archivo.")
    else:
        # inicializamos el LLM una sola vez
        llm = get_llm()
        # Procesamos cada carga
        for up in uploads:
            st.markdown("---")
            st.subheader(f"📄 {up.name}")

            # extraer texto del archivo no puede ser imgen
            text = read_text(up)
            if not text.strip():
                st.warning("No se pudo extraer texto.")
                continue

            # 2 Llamada al LLM para extraer variables en JSON
            with st.spinner("🤖 Extrayendo variables..."):
                try:

                    out = llm.invoke(LLM_PROMPT.format(texto=text))
                    raw = out.content.strip()   # Texto crudo devuelto por el modelo
                    data = extract_json_safe(raw)  # Intento robusto de parsear a dict
                except Exception as e:
                    # Cualquier problema de red/parsing levanta un error legible
                    st.error(f"Error procesando {up.name}: {e}")
                    continue

            # 3 Validación de que obtuvimos algo parseable
            if not data:
                st.error("No se pudo interpretar JSON válido.")
                continue

            # 4 Mostrar las variables extraídas
            st.write("**Variables extraídas:**")
            st.json(data)

            # 5 Evaluar reglas y mostrar decisión final
            aprobado = eval_rules(data)
            if aprobado:
                st.success("🎉 DECISIÓN: **APROBADO**")
            else:
                st.error("⛔ DECISIÓN: **RECHAZADO**")
