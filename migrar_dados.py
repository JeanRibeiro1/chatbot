import pandas as pd
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv
from bot import preprocessar_texto, PerguntaResposta, SessionLocal

# Carregar variáveis de ambiente
load_dotenv()

# Verificar se o arquivo CSV existe
if not os.path.exists('perguntas_respostas.csv'):
    raise FileNotFoundError("O arquivo perguntas_respostas.csv não foi encontrado no diretório atual.")

def migrar_dados():
    try:
        # Ler o arquivo CSV
        print("Lendo arquivo CSV...")
        df = pd.read_csv('perguntas_respostas.csv', encoding='utf-8', quotechar='"')
        
        # Criar sessão do banco de dados
        print("Conectando ao banco de dados...")
        session = SessionLocal()
        
        # Processar e inserir cada linha
        print("Iniciando migração dos dados...")
        total_linhas = len(df)
        for index, row in df.iterrows():
            texto_processado = preprocessar_texto(row['pergunta'])
            nova_entrada = PerguntaResposta(
                pergunta=row['pergunta'],
                resposta=row['resposta'],
                texto_processado=texto_processado
            )
            session.add(nova_entrada)
            if (index + 1) % 10 == 0:
                print(f"Processando {index + 1} de {total_linhas} registros...")
        
        # Commit das alterações
        print("Salvando alterações no banco de dados...")
        session.commit()
        session.close()
        
        print("Migração concluída com sucesso!")
        
    except Exception as e:
        print(f"Erro durante a migração: {str(e)}")
        raise

if __name__ == "__main__":
    migrar_dados() 