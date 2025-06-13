import os
import re
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, request
from sqlalchemy import create_engine, Column, Integer, Text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from asgiref.wsgi import WsgiToAsgi
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
import unicodedata 
# --- CONFIGURAÇÃO INICIAL ---
load_dotenv()

DATABASE_URL = os.getenv('DATABASE_URL')
if DATABASE_URL is None:
    raise ValueError("DATABASE_URL não encontrada nas variáveis de ambiente.")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

TOKEN = os.getenv('TELEGRAM_TOKEN')

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- MODELO DE DADOS ---
class PerguntaResposta(Base):
    __tablename__ = "perguntas_respostas"
    id = Column(Integer, primary_key=True, index=True)
    pergunta = Column(Text)
    resposta = Column(Text)
    texto_processado = Column(Text)

Base.metadata.create_all(bind=engine)

# --- DICIONÁRIOS E FUNÇÕES DE PROCESSAMENTO DE TEXTO ---
ABREVIACOES = {
    'adm': 'administração', 'ar': 'administração regional', 'gdf': 'governo do distrito federal',
    'ceb': 'companhia energética de brasília', 'slu': 'serviço de limpeza urbana',
    'caesb': 'companhia de saneamento ambiental do distrito federal', 'art': 'anotação de responsabilidade técnica',
    'sqs': 'superquadra sul', 'alv': 'alvará', 'doc': 'documento', 'docs': 'documentos',
}


def preprocessar_texto(texto):
    """
    Função completa para limpar e normalizar o texto:
    1. Converte para minúsculas.
    2. Remove acentos e diacríticos (ex: "construção" -> "construcao").
    3. Expande abreviações conhecidas.
    4. Remove caracteres especiais e números.
    5. Tokeniza o texto em palavras.
    6. Remove palavras de parada (stopwords).
    7. Aplica stemming para reduzir palavras à sua raiz.
    """
    # 1. Minúsculas
    texto = texto.lower()
    
    # 2. Remover acentos
    nfkd_form = unicodedata.normalize('NFD', texto)
    texto_sem_acento = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
    
    # 3. Expandir abreviações
    palavras = texto_sem_acento.split()
    palavras_expandidas = [ABREVIACOES.get(palavra, palavra) for palavra in palavras]
    texto = ' '.join(palavras_expandidas)
    
    # 4. Remover caracteres especiais e números
    texto = re.sub(r'[^a-z\s]', '', texto)
    
    # 5. Tokenização
    tokens = word_tokenize(texto)
    
    # 6. Remover stopwords
    stop_words = set(stopwords.words('portuguese'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # 7. Stemming
    stemmer = RSLPStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    
    return ' '.join(tokens)

def carregar_dataset():
    try:
        session = SessionLocal()
        df = pd.read_sql("SELECT * FROM perguntas_respostas", session.bind)
        session.close()
        return df
    except Exception as e:
        print(f"Erro ao carregar dataset: {str(e)}")
        return None

def preparar_modelo(df):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['texto_processado'])
    return vectorizer, X, df

def encontrar_resposta(pergunta, vectorizer, X, df):
    pergunta_processada = preprocessar_texto(pergunta)
    pergunta_vetor = vectorizer.transform([pergunta_processada])
    similaridades = cosine_similarity(pergunta_vetor, X)
    indice_mais_similar = similaridades.argmax()
    
    if similaridades[0, indice_mais_similar] > 0.1:
        return df.iloc[indice_mais_similar]['resposta']
    else:
        return "Desculpe, não encontrei uma resposta adequada para sua pergunta. Tente reformular ou perguntar de outra forma."

# --- FUNÇÃO DE LAZY LOADING ---
def load_model_into_context(context: ContextTypes.DEFAULT_TYPE):
    """Carrega o dataset e prepara o modelo de IA, armazenando no contexto do bot."""
    print("Modelo não encontrado no contexto. Carregando e preparando agora...")
    df = carregar_dataset()
    if df is None or df.empty:
        raise RuntimeError("Erro fatal: Não foi possível carregar o dataset do banco de dados.")
    
    vectorizer, X, df_prepared = preparar_modelo(df)
    
    context.bot_data['vectorizer'] = vectorizer
    context.bot_data['X'] = X
    context.bot_data['df'] = df_prepared
    print("Modelo carregado e preparado com sucesso.")

# --- FUNÇÕES DE HANDLER DO TELEGRAM ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('Olá! Eu sou um chatbot da Administração Regional de São Sebastião. Como posso ajudar?')

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Processa a mensagem do usuário, garantindo que o modelo de IA esteja carregado."""
    try:
        # LAZY LOADING: Se o modelo não estiver no contexto, carregue-o.
        if 'vectorizer' not in context.bot_data:
            load_model_into_context(context)

        text = update.message.text
        resposta = encontrar_resposta(
            text, 
            context.bot_data['vectorizer'], 
            context.bot_data['X'], 
            context.bot_data['df']
        )
        await update.message.reply_text(resposta)
    except Exception as e:
        error_message = f'Ocorreu um erro ao processar sua mensagem: {str(e)}'
        print(error_message)
        await update.message.reply_text(error_message)

# --- INICIALIZAÇÃO FINAL PARA O SERVIDOR ---
print("Configurando a aplicação Telegram...")
application = Application.builder().token(TOKEN).build()
application.bot_data['is_initialized'] = False

# Adiciona os handlers (rotinas que processam os comandos e mensagens)
application.add_handler(CommandHandler('start', start))
application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

# Cria a instância do servidor Flask
server = Flask(__name__)

# Define a rota do webhook que recebe as mensagens do Telegram
@server.route('/', methods=['POST'])
async def webhook():
    # Inicializa a aplicação na primeira vez que for usada
    if not application.bot_data.get('is_initialized', False):
        await application.initialize()
        application.bot_data['is_initialized'] = True

    update = Update.de_json(request.get_json(force=True), application.bot)
    await application.process_update(update)
    return 'ok'

# "Traduz" o nosso app Flask (WSGI) para um formato que o Uvicorn (ASGI) entende.
server = WsgiToAsgi(server)
print("Aplicação pronta para ser servida pelo Gunicorn/Uvicorn.")