
### API BOLIVAR - evaluador automático de solicitudes de credito

#APP con Streamlit

# 1 Recibe cartas de solicitantes (TXT o PDF).
# 2 Usa un LLM en este caso OpenAI) para EXTRAER variables, estructuradas (ingresos, monto solicitado, edad).....
# 3 Aplica 8 reglas de negocio y decide: APROBADO o RECHAZADO.

#  se usa:

# - Streamlit para  rápida subida de archivos y visualización.
# - PyPDF2 para extraer texto de PDFs 
# - LangChain para orquestar prompts y el cliente del modelo .

import re          # JSON
import json        # JSON 
import os          # utilidades 
import streamlit as st             # framework APIS
from PyPDF2 import PdfReader       # Lector de PDF para extraer texto
from langchain.prompts import PromptTemplate  # Plantillas de prompts 
from langchain_openai import ChatOpenAI      # LangChain para modelos de chat de OpenAI
import constants                   # key

## Configurar

# configuracion Streamlit:

st.set_page_config(page_title="Evaluador de crédito", page_icon="✅")

# Título grande que aparece arriba de la app
st.header("✅ Evaluador automático de solicitudes de crédito")


# LLM
# programar la IA

def get_llm():
    """
    Crea y devuelve una instancia del modelo de lenguaje (LLM) a través de LangChain.

    POROCESO
    - Centralizamos la inicialización del LLM en una función: más limpio y testeable.
    - Leemos la API Key desde 'constants' para no quemarla en el código fuente y no se vea en el repo.
    - Si falta la clave, detenemos la app de forma amigable con un mensaje (st.stop()).

    Parámetros clave:
    - model="gpt-4o-mini": modelo de OpenAI con buena relación costo/latencia.
    - temperature=0: buscamos respuestas deterministas para minimizar errotres y JSON estable.
    """


    api_key = getattr(constants, "OPENAI_API_KEY", None)  # lee la API key desde constants.py
    if not api_key:

        # error visible en la interfaz y se detiene la ejecución para evitar fallos.
        st.error("⚠️ Falta OPENAI_API_KEY en constants.py")
        st.stop()

    # devuelve un adaptador de LangChain que sabe hablar con openai
    return ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)

# Prompt 

# PromptTemplate define una plantilla {texto}.
# le estoy diciendo: del siguiente texto, extrae estas variables específicas y devuelve en un JSON


LLM_PROMPT = PromptTemplate.from_template("""
 lea la carta y te devuelva la información en el formato JSON.

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

TEXTO:
\"\"\"{texto}\"\"\"
""")


# FUNCIONES NECESARIAS PARA APP

## leer el texto de los archivos, extraerlo 

def read_text(upload) -> str:
    """
    extrae el contenido textual 

    - Si es .txt: se decodifica como UTF-8, ignorando errores de codificación.
    - Si es .pdf: se usa PyPDF2 para concatenar el texto de todas las páginas.
    - si el PDF es una imagen escaneada, no habrá texto.

    Retorna:
      str con el texto completo (o cadena vacía si no se pudo).
    """

    name = upload.name.lower()
    if name.endswith(".txt"): # abre text
        return upload.read().decode("utf-8", errors="ignore")
    if name.endswith(".pdf"): # abre pdf
        reader = PdfReader(upload)
        # devuelve None si la página no tiene capa de texto
        return "\n".join([p.extract_text() or "" for p in reader.pages])
    return ""

## FUN. JSON
## Su tarea es asegurarse de que la salida del modelo siempre termine siendo un JSON valido, 
# incluso si el LLM suministra texto extra. 

def extract_json_safe(raw: str) -> dict:
    """
    Esta función se asegura de que la salida del modelo
    siempre se pueda convertir en un JSON válido.
    """
    try:
        return json.loads(raw)  #el modelo cumplio y devolvi0 JSON puro
    except:
        match = re.search(r"\{.*\}", raw, re.DOTALL)  # busca el primer bloque que empiece con { y termine con }
        if match:
            return json.loads(match.group())
    return {}


## corazon de mi sistema, evalua las reglas....

# revisa una por una todas las reglas de negocio y decide si el cliente aprueba. 
# Usa la lista checks para guardar los resultados de cada regla, 
# y con all(checks) exige que absolutamente todas se cumplan para aprobar.


def eval_rules(data: dict) -> bool: ## organizar info
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

    checks = [] ##lista 

    # Regla 1: Ingresos > 1.000.000 COP
    checks.append(data.get("income") and data["income"] > 1_000_000)

    # Regla 2: NO tener mora en ultimos 6 meses 
    checks.append(data.get("delinquency") is False)

    # Regla 3: Edad min 21 años
    checks.append(data.get("age") and data["age"] >= 21)

    # Regla 4: Monto <= 30% de ingresos
    checks.append(data.get("requested") and data.get("income") and data["requested"] <= 0.3 * data["income"])

    # Regla 5: >= 1 año de experiencia O negocio propio...
    checks.append((data.get("experience") and data["experience"] >= 1) or data.get("owns_business"))

    # Regla 6: No mas de 2 creditos activos
    checks.append(data.get("active_credits") is not None and data["active_credits"] <= 2)

    # Regla 7: Calificación "Buena" o "Excelente"
    checks.append(data.get("credit_rating") in ["buena", "excelente"])

    # Regla 8: No más de 2 rechazos ultimos 12 meses
    checks.append(data.get("rejections") is not None and data["rejections"] <= 2)

    # all(checks) => True solo si TODASSS todasss las condiciones son True
    return all(checks)


# INTERFAZ

# Componente de subida de archivos:

uploads = st.file_uploader("Sube cartas (TXT o PDF)", type=["txt", "pdf"], accept_multiple_files=True)

# Boton importante
if st.button("Evaluar solicitudes"):
    if not uploads:
        # Validación básica de UX: nada que procesar
        st.error("⚠️ Sube al menos un archivo.")
    else:
        # inicializamos el LLM una sola vez esto espara no cargar la red cada vez
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

            #  Llamada al LLM para extraer variables en JSON
            with st.spinner("🤖 Extrayendo variables..."):
                try:

                    out = llm.invoke(LLM_PROMPT.format(texto=text))
                    raw = out.content.strip()   # texto crudo devuelto por el modelo
                    data = extract_json_safe(raw)  # intento robusto de parsear a dict
                except Exception as e:
                    # cualquier problema de redlevanta un error legible
                    st.error(f"Error procesando {up.name}: {e}")
                    continue

            # 3 no valido
            if not data:
                st.error("No se pudo interpretar JSON válido.")
                continue

            # 4 mostrar las variables extraidas
            st.write("**Variables extraídas:**")
            st.json(data)

            # 5 decisión final
            aprobado = eval_rules(data)
            if aprobado:
                st.success("🎉 DECISIÓN: **APROBADO**")
            else:
                st.error("⛔ DECISIÓN: **RECHAZADO**")
