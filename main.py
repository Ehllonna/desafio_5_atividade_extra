# main.py

import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
import zipfile
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from typing import List

# Funções dos outros arquivos
from agent_brain import invoke_agent_executor 
from data_handler import load_and_validate_csv

# --- CONFIGURAÇÃO INICIAL ---
load_dotenv()
st.set_page_config(
    layout="wide", 
    page_title="Agente Autônomo de Dados Ehllonna",
    initial_sidebar_state="expanded" 
)

# --- LÓGICA DA SESSÃO ---
def initialize_session_state():
    """Inicializa as variáveis de estado da sessão."""
    if 'messages' not in st.session_state:
        st.session_state['messages'] = [
            {"role": "assistant", "content": "Olá! Por favor, faça o **upload de um arquivo CSV** ou um arquivo ZIP para começar a análise."}
        ]
    if 'df' not in st.session_state:
        st.session_state['df'] = None
    if 'last_image_path' not in st.session_state:
        st.session_state['last_image_path'] = None

    if not os.path.exists("plots"):
        os.makedirs("plots")

# --- PROCESSAMENTO DE UPLOAD ---
def process_uploaded_file(uploaded_file):
    """Processa o arquivo CSV ou ZIP carregado."""
    file_path = None
    temp_dir = "temp_data"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    if uploaded_file.name.endswith('.csv'):
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
            
    elif uploaded_file.name.endswith('.zip'):
        try:
            with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
                csv_files = [name for name in zip_ref.namelist() if name.endswith('.csv')]
                if csv_files:
                    zip_ref.extract(csv_files[0], temp_dir)
                    file_path = os.path.join(temp_dir, csv_files[0])
                else:
                    st.error("O arquivo ZIP não contém arquivos CSV.")
                    return
        except Exception as e:
            st.error(f"Erro ao processar o arquivo ZIP: {e}")
            return
            
    if file_path:
        df = load_and_validate_csv(file_path)
        if df is not None:
            st.session_state['df'] = df
            st.session_state.messages.append({"role": "assistant", "content": "Dados carregados com sucesso! O que você gostaria de analisar? Experimente: 'Gere um histograma para a coluna Amount' ou 'Quais são os tipos de dados?'"})

# --- LÓGICA DO CHAT ---
def handle_chat_input(prompt: str):
    """Lida com a entrada do usuário, invoca o agente e processa a resposta."""
    st.session_state.messages.append({"role": "user", "content": prompt})

    chat_history_list: List[BaseMessage] = []
    # Pula a primeira mensagem de boas-vindas para não poluir o histórico do LLM
    for msg in st.session_state.messages[1:]:
        if msg["role"] == "user":
            chat_history_list.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant" and isinstance(msg["content"], str):
            chat_history_list.append(AIMessage(content=msg["content"]))
            
    df = st.session_state.get('df') 
    
    with st.spinner("O Agente Ehllonna está analisando..."):
        result = invoke_agent_executor(
            prompt=prompt,
            chat_history=chat_history_list,
            df=df
        )

    final_answer = result["final_answer"]
    new_image_path = result.get("tool_output_image_path")

    st.session_state.messages.append({"role": "assistant", "content": final_answer})
    
    # Se um NOVO gráfico foi gerado, atualiza o caminho na sessão.
    if new_image_path:
        st.session_state.last_image_path = new_image_path

# --- INTERFACE DO USUÁRIO (UI) ---
initialize_session_state()

st.title(" Agente de Análise de Dados Ehllonna")

# Sidebar para Upload
with st.sidebar:
    st.header("Upload de Dados")
    uploaded_file = st.file_uploader(
        "Carregue seu arquivo CSV ou ZIP",
        type=["csv", "zip"]
    )
    if uploaded_file and st.session_state.get('df') is None:
        process_uploaded_file(uploaded_file)
        st.rerun() 

# Exibição do Histórico de Chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Exibição da Imagem (se existir)
# Esta lógica agora é executada após a exibição do texto da mensagem.
if st.session_state.get('last_image_path') and os.path.exists(st.session_state.last_image_path):
    # Verifica se a última mensagem foi do assistente e continha a palavra "gráfico"
    last_message = st.session_state.messages[-1]
    if last_message["role"] == "assistant" and "gráfico" in last_message["content"].lower():
         st.image(st.session_state.last_image_path, caption="Gráfico gerado pela análise", use_column_width=True)

# Visão Geral dos Dados
if st.session_state.df is not None:
    df = st.session_state.df
    with st.expander("📊 Visão Geral do DataFrame"):
        st.markdown(f"**Dimensões:** {df.shape[0]} linhas, {df.shape[1]} colunas.")
        st.dataframe(df.head())

# Área de Input do Chat
if st.session_state.df is not None:
    if prompt := st.chat_input(f"Faça uma pergunta sobre seus dados..."):
        handle_chat_input(prompt)
        st.rerun()