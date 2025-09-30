# agent_brain.py

import os
import re
from typing import Dict, Any, List, Optional
from functools import partial 

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor
# 🚨 MUDANÇA CRÍTICA: Importar StructuredTool
from langchain.tools import Tool, StructuredTool 
from langchain_core.pydantic_v1 import BaseModel, Field

import pandas as pd
import streamlit as st 

# Importa as funções de ferramentas atualizadas
from tools.custom_tools import (
    get_descriptive_stats, 
    generate_plot, 
    get_correlation,
    get_df_info,
    run_python_calculation
) 

# --- SISTEMA DE PROMPT (SEM MUDANÇAS) ---
SYSTEM_PROMPT = """
Você é o Agente de Análise de Dados Ehllonna, um assistente especializado em Pandas e Python.
Sua função é guiar o usuário na exploração e análise do DataFrame (df) fornecido.
---
REGRAS CRÍTICAS:
1. SEMPRE use as ferramentas disponíveis para análise de dados. NUNCA invente uma resposta se uma ferramenta for necessária.
2. Ao ser solicitado um gráfico, SEMPRE use a ferramenta 'generate_plot'.
3. **MAPEAMENTO DE COLUNAS:** TENTE MAPEAR nomes informais para os nomes reais do DataFrame ('Time', 'V1'...'V28', 'Amount', 'Class').
4. **TIPOS DE PLOTAGEM:** Ao usar a ferramenta de gráfico, use apenas os tipos aceitos: **'hist', 'box', 'bar'**.
"""
class StatsToolSchema(BaseModel):
    """Esquema de argumentos para get_descriptive_stats."""
    column: str = Field(description="O nome da coluna para a qual calcular estatísticas. Ex: 'Amount' ou 'Time'.")

class PlotToolSchema(BaseModel):
    """Esquema de argumentos para generate_plot."""
    column: str = Field(description="O nome da coluna para visualização. Ex: 'Amount', 'Class', 'Time'.")
    plot_type: str = Field(description="O tipo de gráfico a ser gerado: 'hist', 'box' ou 'bar'.")

class CorrelationToolSchema(BaseModel):
    """Esquema de argumentos para get_correlation."""
    column1: str = Field(description="O nome da primeira coluna numérica. Ex: 'V1'.")
    column2: str = Field(description="O nome da segunda coluna numérica. Ex: 'V2'.")

class PythonCalcSchema(BaseModel):
    """Esquema de argumentos para run_python_calculation."""
    expression: str = Field(description="A expressão matemática Python a ser avaliada. Ex: '25 * 4'.")

def create_agent_executor(df: pd.DataFrame) -> AgentExecutor:
    """
    Cria e configura o AgentExecutor, injetando o DataFrame nas ferramentas.
    """
    # MUDANÇA CRÍTICA: Uso de StructuredTool para argumentos nomeados
    tools = [
        # get_df_info: Não precisa de StructuredTool pois só tem a injeção do df
        Tool(
            name=get_df_info.__name__,
            description="Use esta ferramenta para obter uma visão geral dos dados, incluindo tipos, contagem de nulos e resumo estatístico.",
            func=partial(get_df_info, df)
        ),
         # get_descriptive_stats: Atribui StatsToolSchema
        StructuredTool(
            name=get_descriptive_stats.__name__,
            description="Use para obter estatísticas resumidas...",
            func=partial(get_descriptive_stats, df),
            args_schema=StatsToolSchema # 🚨 NOVO!
        ),
        # generate_plot: Atribui PlotToolSchema
        StructuredTool(
            name=generate_plot.__name__,
            description="Gera um gráfico ('hist', 'box', 'bar')...",
            func=partial(generate_plot, df),
            args_schema=PlotToolSchema # 🚨 NOVO!
        ),
        # get_correlation: Atribui CorrelationToolSchema
        StructuredTool(
            name=get_correlation.__name__,
            description="Calcula a correlação de Pearson...",
            func=partial(get_correlation, df),
            args_schema=CorrelationToolSchema # 🚨 NOVO!
        ),
        # run_python_calculation: Tool simples, mas vamos usar o schema para robustez
        Tool(
            name=run_python_calculation.__name__,
            description="Use para executar cálculos matemáticos...",
            func=run_python_calculation,
            args_schema=PythonCalcSchema # 🚨 NOVO! (Para esta ferramenta simples, Tool também aceita schema)
        )
    ]

    api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Chave da API GEMINI_API_KEY não encontrada.")

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=api_key, temperature=0)
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent_runnable = create_tool_calling_agent(llm, tools, prompt_template)

    return AgentExecutor(
        agent=agent_runnable,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
    )

def invoke_agent_executor(
    prompt: str, 
    chat_history: List[BaseMessage],
    df: Optional[pd.DataFrame]
) -> Dict[str, Any]:
    """
    Cria um executor com o DataFrame atual e invoca o agente.
    """
    if df is None:
        return {"final_answer": "Por favor, faça o upload de um arquivo CSV primeiro.", "tool_output_image_path": None}

    try:
        agent_executor = create_agent_executor(df)
    except ValueError as e:
        return {"final_answer": f"Erro de Configuração: {e}", "tool_output_image_path": None}

    agent_input = {
        "input": prompt,
        "chat_history": chat_history,
    }

    try:
        result = agent_executor.invoke(agent_input)
        final_answer = result.get("output", "Não foi possível gerar uma resposta.")
        
        last_image_path = None
        if "intermediate_steps" in result:
            for action, observation in result["intermediate_steps"]:
                if action.tool == "generate_plot":
                    match = re.search(r"O arquivo foi salvo em: (plots[/\\][\w\d_.-]+)", str(observation))
                    if match:
                        last_image_path = match.group(1).strip()
        
        return {
            "final_answer": final_answer,
            "tool_output_image_path": last_image_path
        }

    except Exception as e:
        print(f"ERRO CRÍTICO NO AGENT EXECUTOR: {type(e).__name__}: {e}")
        error_message = f"Ocorreu uma falha inesperada ({type(e).__name__}). Por favor, verifique os logs e tente novamente."
        return {"final_answer": error_message, "tool_output_image_path": None}