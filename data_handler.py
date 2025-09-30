import pandas as pd
import streamlit as st
from typing import Optional # Opcional, mas boa prática para indicar retorno None

def load_and_validate_csv(filepath: str) -> Optional[pd.DataFrame]:
    """
    Carrega o DataFrame a partir do caminho do arquivo CSV (filepath) e 
    realiza validações básicas.
    
    Args:
        filepath: O caminho completo para o arquivo CSV no sistema de arquivos 
                  (pode ser o arquivo original ou o extraído do ZIP).
        
    Returns:
        Um DataFrame do Pandas se for válido, ou None em caso de erro.
    """
    try:
        # A chave é usar o 'filepath' recebido, que aponta para o arquivo no disco.
        df = pd.read_csv(filepath) 
        
        if df.empty:
            st.error("O arquivo CSV está vazio. Não foi possível carregar os dados.")
            return None
            
        st.success(f"DataFrame lido com sucesso: {len(df)} linhas e {len(df.columns)} colunas.")
        return df
        
    except FileNotFoundError:
        st.error(f"Erro: O arquivo CSV não foi encontrado no caminho: {filepath}. Verifique se o ZIP continha um CSV.")
        return None
    except Exception as e:
        st.error(f"Erro ao ler ou processar o arquivo CSV. Verifique o formato. Detalhes: {e}")
        return None