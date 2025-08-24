## ChatBot conversacional con LangChain, tiene memoria de la conversación

import streamlit as st # interfaz

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
import constants #


## Configuración Página
st.set_page_config(page_title="ChatBot Conversacional", page_icon="🤖")
st.header("🤖 ChatBot Conversacional")

## Modelo  

def get_llm():
    api_key = constants.OPENAI_API_KEY
    if not api_key:
        st.error("Falta la variable OPENAI_API_KEY en constants.py")
        st.stop()
    return ChatOpenAI(api_key=api_key, model="gpt-4o-mini", temperature=0.5)

chat = get_llm()

## sistema de mensajes amigable
if 'flowmessages' not in st.session_state:
    st.session_state['flowmessages'] = [
        SystemMessage(content="Eres un asistente servicial y amigable.")]


## recibe el texto del usuario
def get_openai_response(question: str) -> str:
    """Envía la pregunta al modelo y actualiza el historial en session_state."""
    st.session_state["flowmessages"].append(HumanMessage(content=question))
    answer = chat.invoke(st.session_state["flowmessages"])
    st.session_state["flowmessages"].append(AIMessage(content=answer.content))
    return answer.content


user_input = st.text_input("Haz una pregunta al ChatBot", placeholder="Escribe tu pregunta aquí...",
    label_visibility="hidden"
)

## Botón de envío
submit = st.button("Haz tu pregunta")

if submit:
    if not user_input.strip():
        st.warning("Por favor escribe una pregunta antes de enviar.")
    else:
        try:
            response = get_openai_response(user_input)
            st.subheader("Respuesta del ChatBot")
            st.write(response)
        except Exception as e:
            st.error(f"Ocurrió un error: {e}")
