import os
import gspread
import pandas as pd
from sqlalchemy import create_engine
from gspread_dataframe import set_with_dataframe
import base64
import json

print("Worker iniciado. Sincronizando dados com a Planilha Google...")

# --- ETAPA 1: Conexão com o Banco de Dados (Usando a URL interna do Fly.io) ---
DATABASE_URL = os.getenv('DATABASE_URL')
if not DATABASE_URL:
    print("ERRO: Variável de ambiente DATABASE_URL não encontrada.")
    exit()

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

try:
    engine = create_engine(DATABASE_URL)
    df_interacoes = pd.read_sql("SELECT * FROM historico_interacoes", engine)
    print(f"{len(df_interacoes)} interações encontradas no banco de dados.")
except Exception as e:
    print(f"ERRO: Não foi possível conectar ao banco de dados. Erro: {e}")
    exit()

# --- ETAPA 2: Leitura das Credenciais do Google a partir dos Secrets do Fly.io ---
GOOGLE_CREDS_BASE64 = os.getenv('GOOGLE_CREDENTIALS_JSON_BASE64')
if not GOOGLE_CREDS_BASE64:
    print("ERRO: Secret GOOGLE_CREDENTIALS_JSON_BASE64 não encontrado.")
    exit()

try:
    # Decodifica a string base64 para obter o JSON original
    creds_json_str = base64.b64decode(GOOGLE_CREDS_BASE64).decode('utf-8')
    creds_json = json.loads(creds_json_str)

    # Autentica no Google Sheets com as credenciais decodificadas
    gc = gspread.service_account_from_dict(creds_json)
    
    # **SUBSTITUA 'Dados do Bot' PELO NOME EXATO DA SUA PLANILHA**
    spreadsheet = gc.open("Dados do Bot")
    worksheet = spreadsheet.sheet1

    print("Limpando a planilha e enviando novos dados...")
    worksheet.clear()
    set_with_dataframe(worksheet, df_interacoes)
    print("Worker finalizou. Sincronização concluída com sucesso!")

except Exception as e:
    print(f"ERRO: Falha ao sincronizar com a Planilha Google. Erro: {e}")