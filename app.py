# Chat Bot Laura
import os # leer variables de entorno
from dotenv import load_dotenv # cargar variables de entorno desde un archivo .env
import streamlit as st # framework para crear apps web
import constants # archivo con la llave de openai

from langchain.llms import OpenAI # modelo chat moderno

load_dotenv() ## toma la llave de openai (variables de entorno), es mas seguro usar un archivo .env

## inicializar la app con streamlit (arrancar la app)
st.set_page_config(page_title="Chat Bot Laura")
st.header("LangChain - Chat Bot Laura")

## crear modelo solo una vez

@st.cache_resource # mejora el rendimiento al evitar la recreación del modelo en cada interacción
def get_llm():

    api_key = constants.OPENAI_API_KEY 
    if not api_key:
        st.error("Falta la variable OPENAI_API_KEY en tu archivo .env")
        st.stop()
    return OpenAI(api_key=api_key, model="gpt-3.5-turbo-instruct", temperature=0.6)


llm = get_llm()

## crear una función para que el modelo OpenAI responda
def get_openai_response(question: str) -> str:

    return llm.invoke(question)

# entrada y acción (interfaz de usuario y botón)
user_input = st.text_input("Input:", key="input")
submit = st.button("Pregunta a Laura")


# acción cuando se presiona el botón
if submit:
    response = get_openai_response(user_input)
    st.subheader("Tu respuesta es:")
    st.write(response)