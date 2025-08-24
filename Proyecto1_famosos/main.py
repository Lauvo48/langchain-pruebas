## Ejemplo 1: mi primera API con OpenAI

## 1. Integrar codigo con con la API de OpenAI
import os
from Proyecto1_famosos.constants import openai_key
from langchain_openai import OpenAI

import streamlit as st

os.environ["OPENAI_API_KEY"] = openai_key

# Streamlit framework 

st.title("Mi primera API con OpenAI")
input_text = st.text_input("Escribe tu pregunta")

## OPENAI LLMS
llm = OpenAI(temperature=0.8)


if input_text:
    st.write(llm.invoke(input_text))