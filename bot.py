# bot.py
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

# Carregar variáveis de ambiente
load_dotenv()

# Baixar recursos necessários do NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('rslp')

# TOKEN do Bot
TOKEN = os.getenv('TELEGRAM_TOKEN')

# Dicionário de abreviações comuns
ABREVIACOES = {
    'adm': 'administração',
    'ar': 'administração regional',
    'gdf': 'governo do distrito federal',
    'ceb': 'companhia energética de brasília',
    'slu': 'serviço de limpeza urbana',
    'caesb': 'companhia de saneamento ambiental do distrito federal',
    'art': 'anotação de responsabilidade técnica',
    'sqs': 'superquadra sul',
    'alv': 'alvará',
    'doc': 'documento',
    'docs': 'documentos',
    'pç': 'praça',
    'pça': 'praça',
    'r': 'rua',
    'av': 'avenida',
    'q': 'quadra',
    'lote': 'loteamento',
    'lt': 'loteamento',
    'bl': 'bloco',
    'apto': 'apartamento',
    'apt': 'apartamento',
    'casa': 'residência',
    'res': 'residência',
    'com': 'comércio',
    'coml': 'comercial',
    'loja': 'estabelecimento comercial',
    'estab': 'estabelecimento',
    'tel': 'telefone',
    'fone': 'telefone',
    'cel': 'celular',
    'email': 'e-mail',
    'mail': 'e-mail',
    'site': 'página web',
    'web': 'página web',
    'pag': 'página',
    'pg': 'página',
    'end': 'endereço',
    'ender': 'endereço',
    'loc': 'localização',
    'local': 'localização',
    'hor': 'horário',
    'hr': 'horário',
    'hora': 'horário',
    'func': 'funcionamento',
    'atend': 'atendimento',
    'serv': 'serviço',
    'srv': 'serviço',
    'proc': 'processo',
    'proc': 'procedimento',
    'doc': 'documentação',
    'req': 'requisito',
    'requis': 'requisito',
    'sol': 'solicitação',
    'solic': 'solicitação',
    'ped': 'pedido',
    'pedido': 'solicitação',
    'den': 'denúncia',
    'denunc': 'denúncia',
    'recl': 'reclamação',
    'reclam': 'reclamação',
    'inf': 'informação',
    'info': 'informação',
    'infos': 'informações',
    'dúv': 'dúvida',
    'duv': 'dúvida',
    'duvida': 'dúvida',
    'ajuda': 'assistência',
    'socorro': 'assistência',
    'urg': 'urgente',
    'urgente': 'urgente',
    'pri': 'prioridade',
    'prio': 'prioridade',
    'priori': 'prioridade',
    'import': 'importante',
    'imp': 'importante',
    'necess': 'necessário',
    'nec': 'necessário',
    'obrig': 'obrigatório',
    'obg': 'obrigatório',
    'obriga': 'obrigatório',
    'prec': 'preciso',
    'precis': 'preciso',
    'preciso': 'preciso',
    'quero': 'desejo',
    'desej': 'desejo',
    'desejo': 'desejo'
}

# Dicionário de correções comuns
CORRECOES = {
    'alvara': 'alvará',
    'alvaras': 'alvarás',
    'administracao': 'administração',
    'brasilia': 'brasília',
    'minh': 'minha',
    'denuncia': 'denúncia',
    'vazamento': 'vazamento',
    'agua': 'água',
    'terrenos': 'terreno',
    'baldios': 'baldio',
    'poda': 'poda',
    'irregular': 'irregular',
    'iluminação': 'iluminação',
    'publica': 'pública',
    'manutenção': 'manutenção',
    'sede': 'sede',
    'plano': 'plano',
    'piloto': 'piloto',
    'fazer': 'solicitar',
    'entrar': 'contato',
    'contato': 'contato',
    'solicitação': 'solicitar',
    'denunciar': 'denúncia',
    'denuncia': 'denúncia',
    'horario': 'horário'
}

# Função para corrigir texto
def corrigir_texto(texto):
    # Converter para minúsculas
    texto = texto.lower()
    
    # Remover caracteres especiais e números
    texto = re.sub(r'[^a-záàâãéèêíìîóòôõúùûç\s]', '', texto)
    
    # Substituir abreviações
    palavras = texto.split()
    palavras_corrigidas = []
    
    for palavra in palavras:
        # Verificar se é uma abreviação
        if palavra in ABREVIACOES:
            palavras_corrigidas.append(ABREVIACOES[palavra])
        # Verificar se precisa de correção
        elif palavra in CORRECOES:
            palavras_corrigidas.append(CORRECOES[palavra])
        else:
            palavras_corrigidas.append(palavra)
    
    return ' '.join(palavras_corrigidas)

# Função global para pré-processamento de texto
def preprocessar_texto(texto):
    # Corrigir texto
    texto = corrigir_texto(texto)
    
    stemmer = RSLPStemmer()
    stop_words = set(stopwords.words('portuguese'))
    tokens = word_tokenize(texto)
    
    # Processar tokens
    tokens_processados = []
    for token in tokens:
        # Remover stopwords e tokens não alfabéticos
        if token.isalpha() and token not in stop_words:
            # Aplicar stemming
            token_stem = stemmer.stem(token)
            tokens_processados.append(token_stem)
    
    return ' '.join(tokens_processados)

# Carregar e preparar o dataset
def carregar_dataset():
    try:
        df = pd.read_csv('perguntas_respostas.csv', encoding='utf-8', quotechar='"')
        return df
    except Exception as e:
        print(f"Erro ao carregar dataset: {str(e)}")
        return None

# Preparar o modelo de similaridade
def preparar_modelo(df):
    df['texto_processado'] = df['pergunta'].apply(preprocessar_texto)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['texto_processado'])
    return vectorizer, X, df

# Palavras-chave para reconhecimento de perguntas administrativas
PALAVRAS_CHAVE = [
    # Palavras interrogativas
    'como', 'onde', 'quando', 'quem', 'qual', 'quais', 'o que', 'por que', 'porque', '?',
    # Verbos de ação
    'solicitar', 'contato', 'entrar', 'contatar', 'fazer', 'denunciar', 'denúncia',
    'manutenção', 'manter', 'consertar', 'reparar', 'reclamar', 'reclamação',
    # Termos específicos
    'alvará', 'alvara', 'limpeza', 'terreno', 'baldio', 'horário', 'abre', 'funciona',
    'poda', 'árvore', 'árvores', 'iluminação', 'pública', 'publica', 'luz', 'poste',
    'sede', 'local', 'endereço', 'endereco', 'plano', 'piloto', 'regional',
    # Verbos auxiliares
    'posso', 'devo', 'preciso', 'quero', 'desejo', 'gostaria'
]

# Função para encontrar resposta mais similar
def encontrar_resposta(pergunta, vectorizer, X, df):
    pergunta_processada = preprocessar_texto(pergunta)
    pergunta_vetor = vectorizer.transform([pergunta_processada])
    
    similaridades = cosine_similarity(pergunta_vetor, X)
    indice_mais_similar = similaridades.argmax()
    
    # Reduzindo o limiar de similaridade para 0.1
    if similaridades[0, indice_mais_similar] > 0.1:
        return df.iloc[indice_mais_similar]['resposta']
    else:
        # Tentar encontrar por palavras-chave específicas
        palavras_chave = {
            'alvará': 'Para solicitar um alvará de construção, você precisa apresentar os seguintes documentos na administração regional: projeto arquitetônico, ART do responsável técnico, comprovante de propriedade do terreno e documentos pessoais.',
            'contato': 'Você pode entrar em contato com a administração regional através do telefone (61) 9999-9999, e-mail contato@adminregional.df.gov.br ou pessoalmente na sede administrativa.',
            'limpeza': 'Para solicitar a limpeza de terrenos baldios, você pode fazer a solicitação através do aplicativo 156, site do GDF ou diretamente na administração regional.',
            'horário': 'A administração regional funciona de segunda a sexta, das 8h às 18h, e aos sábados das 8h às 12h.',
            'denúncia': 'Para fazer uma denúncia de poda irregular ou qualquer outra irregularidade, você pode usar o aplicativo 156, site do GDF ou comparecer pessoalmente na administração regional.',
            'iluminação': 'Para solicitar manutenção de iluminação pública, você pode fazer a solicitação através do aplicativo 156, site da CEB ou diretamente na administração regional.',
            'sede': 'A sede da administração regional do Plano Piloto fica na SQS 104, Bloco A, Edifício Sede.',
            'poda': 'Para denunciar poda irregular de árvores, você pode fazer a denúncia através do aplicativo 156, site do GDF ou diretamente na administração regional.',
            'administração': 'A administração regional é responsável por diversos serviços como: emissão de alvarás, fiscalização de obras, manutenção de áreas públicas, limpeza urbana, entre outros serviços essenciais para a comunidade.'
        }
        
        # Verifica cada palavra da pergunta processada
        for palavra in pergunta_processada.split():
            if palavra in palavras_chave:
                return palavras_chave[palavra]
        
        return "Desculpe, não encontrei uma resposta adequada para sua pergunta. Tente reformular ou perguntar de outra forma."

# Função para iniciar o bot
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [
            InlineKeyboardButton("Perguntas Administração", callback_data='admin')
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        'Olá! Eu sou um chatbot da Administração Regional. Como posso ajudar?',
        reply_markup=reply_markup
    )

# Função para processar mensagens
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        text = update.message.text.lower()
        
        # Verifica se é uma pergunta sobre administração
        if any(palavra in text for palavra in PALAVRAS_CHAVE) or 'administração' in text or 'regional' in text:
            # Se estiver no modo de perguntas administrativas ou se a mensagem claramente for uma pergunta administrativa
            if context.user_data.get('aguardando_pergunta_admin', False) or any(palavra in text for palavra in ['administração', 'regional', 'alvará', 'contato', 'denúncia']):
                resposta = encontrar_resposta(text, context.bot_data['vectorizer'], 
                                            context.bot_data['X'], context.bot_data['df'])
                await update.message.reply_text(resposta)
                context.user_data['aguardando_pergunta_admin'] = False
            else:
                # Se não estiver no modo de perguntas administrativas, mas a mensagem parecer uma pergunta administrativa
                context.user_data['aguardando_pergunta_admin'] = True
                resposta = encontrar_resposta(text, context.bot_data['vectorizer'], 
                                            context.bot_data['X'], context.bot_data['df'])
                await update.message.reply_text(resposta)
                context.user_data['aguardando_pergunta_admin'] = False
        else:
            # Se não for uma pergunta administrativa
            if context.user_data.get('aguardando_pergunta_admin', False):
                await update.message.reply_text("Por favor, faça uma pergunta sobre a administração regional.")
                context.user_data['aguardando_pergunta_admin'] = False
            else:
                await update.message.reply_text("Por favor, faça uma pergunta sobre a administração regional.")
    except Exception as e:
        await update.message.reply_text(f'Ocorreu um erro ao processar sua mensagem: {str(e)}')

# Função para iniciar perguntas sobre administração
async def iniciar_perguntas_admin(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    context.user_data['aguardando_pergunta_admin'] = True
    await query.edit_message_text(
        text="Por favor, digite sua pergunta sobre a administração regional de Brasília."
    )

# Função para processar botões
async def button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    if query.data == 'admin':
        await iniciar_perguntas_admin(update, context)

# Função principal
def main():
    try:
        # Carregar e preparar o dataset
        df = carregar_dataset()
        if df is None:
            print("Erro ao carregar o dataset. Verifique se o arquivo perguntas_respostas.csv existe.")
            return
            
        vectorizer, X, df = preparar_modelo(df)
        
        # Criar a aplicação
        application = Application.builder().token(TOKEN).build()
        
        # Armazenar dados do modelo no bot
        application.bot_data['vectorizer'] = vectorizer
        application.bot_data['X'] = X
        application.bot_data['df'] = df

        # Adicionar handlers
        application.add_handler(CommandHandler('start', start))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
        application.add_handler(CallbackQueryHandler(button))

        # Iniciar o bot
        print('Bot iniciado...')
        application.run_polling()
    except Exception as e:
        print(f'Erro ao iniciar o bot: {str(e)}')

if __name__ == '__main__':
    main()
