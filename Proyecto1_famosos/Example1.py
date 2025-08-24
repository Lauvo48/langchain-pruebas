##  Integrar codigo con con la API de OpenAI y Langchain
import os
from Proyecto1_famosos.constants import openai_key
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain #unir el modelo con el prompt
from langchain.chains import SequentialChain #unir varios chains

import streamlit as st

os.environ["OPENAI_API_KEY"] = openai_key

# Streamlit framework(permite construir apps web de forma sencilla)
st.title("Busqueda de Famosos con LAU")
input_text = st.text_input("Busca una persona famosa!!")

#1. Prompt Templates (Plantilla de instrucciones para LLM)
first_input_prompt = PromptTemplate(
    input_variables=["name"],
    template="Cuentame algo sobre {name} en tres lineas",
)

#2. Prompt Templates (Plantilla de instrucciones para LLM)
second_input_prompt = PromptTemplate(
    input_variables=['persona'],
    template="¿Cuando nació {persona}? Responde solo con la fecha de nacimiento si la conoces",
)

#3. Prompt Templates (Plantilla de instrucciones para LLM)
thir_input_prompt = PromptTemplate(
    input_variables=['fecha'],
    template="¿Que paso el día {fecha}? Responde con un acontecimiento historico que paso ese día",
)


## OPENAI LLMS - Modelo de lenguaje
llm = OpenAI(temperature=0.8)

# Crear los chains
chain = LLMChain(llm=llm, prompt=first_input_prompt, verbose=True, output_key='persona')
chain2 = LLMChain(llm=llm, prompt=second_input_prompt, verbose=True, output_key='fecha')
chain3 = LLMChain(llm=llm, prompt=thir_input_prompt, verbose=True, output_key='acontecimiento')

# unir los chains en uno solo
parent_chain = SequentialChain(
    chains=[chain, chain2, chain3], input_variables = ['name'], output_variables = ['persona', 'fecha', 'acontecimiento'], verbose=True)



# solo ejecuta si hay input (escrito por el usuario)
if input_text:

    # pasa el nombre escrito por el usuario a la cadena completa
    result = parent_chain.invoke({"name": input_text})

    # muestra lo que encontro de esa persona
    st.write("Descripción:", result.get("persona"))

    # muestra la fecha de nacimiento
    st.write("Fecha de Nacimiento:", result.get("fecha"))

    # muestra el acontecimiento historico en esa fecha
    st.write("Acontecimiento Historico:", result.get("acontecimiento"))