# tools/custom_tools.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import time
import numexpr as ne
import streamlit as st # Garante que o Streamlit é importado

# Dicionário de sinônimos (SEM MUDANÇAS)
MAPPINGS = {
    'quantia': 'Amount',
    'quantidade': 'Amount',
    'valor': 'Amount',
    'custa': 'Amount',
    'fraude': 'Class',
    'fraudes': 'Class',
    'classe': 'Class', 
    'tempo': 'Time',
    'segundos': 'Time'
}

def _map_column_name(column: str, df_columns: pd.Index) -> str:
    """Mapeia sinônimos comuns para nomes de colunas reais."""
    if not column:
        return column
    
    norm_col = column.lower().strip()
    
    n = len(norm_col)
    if n > 4 and n % 2 == 0 and norm_col[:n//2] == norm_col[n//2:]:
        norm_col = norm_col[:n//2]
        
    mapped_name = MAPPINGS.get(norm_col, norm_col)
    
    if mapped_name in df_columns:
        return mapped_name
    if column in df_columns:
        return column
        
    return column

# --- 1. get_df_info (CORRIGIDA COM *args e **kwargs) ---
def get_df_info(df: pd.DataFrame, *args, **kwargs) -> str:
    """
    Retorna o resumo das colunas, incluindo tipos de dados (dtypes) e valores não nulos.
    *args captura o argumento posicional extra do LangChain.
    """
    try:
        output_buffer = StringIO()
        df.info(buf=output_buffer, verbose=True)
        df_info = output_buffer.getvalue()
        
        desc_stats = df.describe(include='all').to_markdown()

        result = (
            "--- Tipos de Dados e Info ---\n\n"
            f"Shape do DataFrame: {df.shape[0]} linhas, {df.shape[1]} colunas.\n\n"
            f"{df_info}\n\n"
            "--- Resumo Estatístico ---\n\n"
            f"{desc_stats}"
        )
        return result
    except Exception as e: 
        return f"Erro Crítico ao gerar resumo dos dados: {type(e).__name__}: {e}"

# --- 2. get_descriptive_stats (CORRIGIDA COM *args e **kwargs) ---
def get_descriptive_stats(df: pd.DataFrame, column: str, *args, **kwargs) -> str:
    """
    Retorna estatísticas descritivas (mean, min, max, std) para uma coluna específica.
    """
    if not column:
        return "Erro: O nome da coluna é obrigatório (e.g., Time, Amount, V1). Por favor, forneça o nome."

    original_column = column
    column = _map_column_name(column, df.columns)
    
    if column not in df.columns:
        available_cols = ", ".join(df.columns.tolist())
        return f"Erro: A coluna '{original_column}' não foi encontrada. Colunas disponíveis: {available_cols}"

    try:
        if pd.api.types.is_numeric_dtype(df[column]):
            stats = df[column].describe().to_markdown()
            return f"Estatísticas Descritivas para a coluna '{column}':\n\n{stats}"
        else:
            value_counts = df[column].value_counts().head(10).to_markdown()
            return f"A coluna '{column}' é categórica. Principais 10 contagens de valores:\n\n{value_counts}"
    except Exception as e:
        return f"Erro ao gerar estatísticas para '{column}': {type(e).__name__}: {e}"

# --- 3. generate_plot (CORRIGIDA COM *args e **kwargs) ---
def generate_plot(df: pd.DataFrame, column: str, plot_type: str = 'hist', *args, **kwargs) -> str:
    """
    Gera um gráfico do tipo e coluna especificados e salva a imagem em 'plots/'.
    """
    original_column = column
    column = _map_column_name(column, df.columns)
    
    if column not in df.columns:
        return f"Erro: Coluna '{original_column}' não encontrada. Colunas disponíveis: {df.columns.tolist()}"

    is_amount_column = column == 'Amount'
    is_numeric = pd.api.types.is_numeric_dtype(df[column])

    plt.figure(figsize=(10, 6))

    try:
        if plot_type == 'bar' and is_numeric:
            plt.close()
            return f"Erro: Gráfico de barras ('bar') não é adequado para a coluna numérica '{column}'. Use 'hist' ou 'box'."
        
        if plot_type == 'hist' and is_numeric:
            if is_amount_column:
                data_to_plot = df[column].apply(lambda x: np.log1p(x)).dropna()
                sns.histplot(data_to_plot, kde=True)
                plt.title(f'Histograma de Log(1 + {column})')
                plt.xlabel(f'Log(1 + {column})')
            else:
                sns.histplot(df[column].dropna(), kde=True)
                plt.title(f'Histograma de {column}')
        
        elif plot_type == 'box' and is_numeric:
            sns.boxplot(y=df[column])
            plt.title(f'Box Plot de {column}')
            if is_amount_column:
                plt.yscale('log')
        
        elif plot_type == 'bar' and not is_numeric:
            sns.countplot(y=df[column], order=df[column].value_counts().index)
            plt.title(f'Contagem de {column}')
        
        else:
            plt.close()
            return f"Erro: Tipo de plotagem '{plot_type}' não é adequado para a coluna '{column}'."
            
    except Exception as e:
        plt.close()
        return f"Erro Crítico ao gerar o gráfico de '{column}': {type(e).__name__}: {e}."

    output_dir = "plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    timestamp = int(time.time())
    file_path = os.path.join(output_dir, f"plot_{column}_{timestamp}.png")
    
    plt.tight_layout()
    try:
        plt.savefig(file_path)
    except Exception as e:
        plt.close() 
        return f"Erro ao salvar o gráfico: {e}."
    
    plt.close()
    return f"Gráfico gerado com sucesso. O arquivo foi salvo em: {file_path}"

# --- 4. get_correlation (CORRIGIDA COM *args e **kwargs) ---
def get_correlation(df: pd.DataFrame, column1: str, column2: str, *args, **kwargs) -> str:
    """
    Calcula o coeficiente de correlação de Pearson entre duas colunas numéricas.
    """
    col1_map = _map_column_name(column1, df.columns)
    col2_map = _map_column_name(column2, df.columns)
    
    if col1_map not in df.columns or col2_map not in df.columns:
        return f"Erro: Uma ou ambas as colunas ('{column1}', '{column2}') não foram encontradas."

    if not pd.api.types.is_numeric_dtype(df[col1_map]) or not pd.api.types.is_numeric_dtype(df[col2_map]):
        return "Erro: A correlação só pode ser calculada entre colunas numéricas."
        
    try:
        correlation = df[[col1_map, col2_map]].corr().loc[col1_map, col2_map]
        return f"O coeficiente de correlação (Pearson) entre '{col1_map}' e '{col2_map}' é: **{correlation:.4f}**"
    except Exception as e:
        return f"Erro ao calcular a correlação: {e}."

# --- 5. run_python_calculation (SEM MUDANÇAS) ---
def run_python_calculation(expression: str, **kwargs) -> str:
    """
    Executa uma expressão matemática simples em Python usando numexpr.
    """
    try:
        result = ne.evaluate(expression).item()
        return f"Resultado do cálculo '{expression}': {result}"
    except Exception as e:
        return f"Erro ao executar a expressão Python: {e}"
