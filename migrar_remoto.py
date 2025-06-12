import pandas as pd
from sqlalchemy import create_engine, Column, Integer, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
import re

# Carregar variáveis de ambiente
load_dotenv()

# Configuração do banco de dados
DATABASE_URL = os.getenv('DATABASE_URL')
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Modelo de dados
class PerguntaResposta(Base):
    __tablename__ = "perguntas_respostas"

    id = Column(Integer, primary_key=True, index=True)
    pergunta = Column(Text)
    resposta = Column(Text)
    texto_processado = Column(Text)

# Função para pré-processar texto
def preprocessar_texto(texto):
    # Converter para minúsculas
    texto = texto.lower()
    
    # Remover caracteres especiais e números
    texto = re.sub(r'[^a-záàâãéèêíìîóòôõúùûç\s]', '', texto)
    
    # Tokenização
    tokens = word_tokenize(texto)
    
    # Remover stopwords
    stop_words = set(stopwords.words('portuguese'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Stemming
    stemmer = RSLPStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    
    return ' '.join(tokens)

def migrar_dados():
    try:
        # Criar tabelas
        Base.metadata.create_all(bind=engine)
        
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