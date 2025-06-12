# bot.py - Versão Final Corrigida

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
import nltk
import os
import pandas as pd
from dotenv import load_dotenv
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from flask import Flask, request

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
    # Adicione outras abreviações conforme necessário
}

CORRECOES = {
    'alvara': 'alvará', 'administracao': 'administração', 'brasilia': 'brasília',
    'denuncia': 'denúncia', 'iluminacao': 'iluminação', 'publica': 'pública',
    'manutencao': 'manutenção', 'horario': 'horário',
    # Adicione outras correções comuns
}

def corrigir_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'[^a-záàâãéèêíìîóòôõúùûç\s]', '', texto)
    palavras = texto.split()
    palavras_corrigidas = [ABREVIACOES.get(palavra, CORRECOES.get(palavra, palavra)) for palavra in palavras]
    return ' '.join(palavras_corrigidas)

def preprocessar_texto(texto):
    texto = corrigir_texto(texto)
    stemmer = RSLPStemmer()
    stop_words = set(stopwords.words('portuguese'))
    tokens = word_tokenize(texto)
    tokens_processados = [stemmer.stem(token) for token in tokens if token.isalpha() and token not in stop_words]
    return ' '.join(tokens_processados)

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
    # Agora estamos usando a coluna 'texto_processado' que já veio pronta do banco de dados.
    # Isso garante consistência e é mais eficiente.
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

# --- FUNÇÕES DE HANDLER DO TELEGRAM ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('Olá! Eu sou um chatbot da Administração Regional. Como posso ajudar?')

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        text = update.message.text
        resposta = encontrar_resposta(text, context.bot_data['vectorizer'], context.bot_data['X'], context.bot_data['df'])
        await update.message.reply_text(resposta)
    except Exception as e:
        await update.message.reply_text(f'Ocorreu um erro ao processar sua mensagem: {str(e)}')


# --- INICIALIZAÇÃO FINAL PARA O SERVIDOR ---

# 1. Carrega os dados e prepara o modelo UMA VEZ quando o app inicia
print("Inicializando o bot: carregando e preparando o modelo...")
df_global = carregar_dataset()
if df_global is None or df_global.empty:
    raise RuntimeError("Erro fatal: Não foi possível carregar o dataset do banco de dados ou ele está vazio.")
vectorizer_global, X_global, df_prepared_global = preparar_modelo(df_global)
print("Modelo preparado com sucesso.")

# 2. Cria a instância da aplicação do Telegram e armazena os dados do modelo
application = Application.builder().token(TOKEN).build()
application.bot_data['vectorizer'] = vectorizer_global
application.bot_data['X'] = X_global
application.bot_data['df'] = df_prepared_global

# 3. Adiciona os handlers (rotinas que processam os comandos e mensagens)
application.add_handler(CommandHandler('start', start))
application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

# 4. Cria a instância do servidor Flask (esta é a variável 'server' que o Gunicorn procura)
server = Flask(__name__)

# 5. Define a rota do webhook que recebe as mensagens do Telegram
@server.route('/', methods=['POST'])
async def webhook():
    update = Update.de_json(request.get_json(force=True), application.bot)
    await application.process_update(update)
    return 'ok'