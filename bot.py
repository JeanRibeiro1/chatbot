# bot.py
# This bot uses Twilio for WhatsApp communication.
# Ensure the following environment variables are set:
# TWILIO_ACCOUNT_SID: Your Twilio Account SID
# TWILIO_AUTH_TOKEN: Your Twilio Auth Token
# TWILIO_WHATSAPP_NUMBER: Your Twilio WhatsApp-enabled number (e.g., whatsapp:+14155238886)

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
from flask import Flask, request # Corrected Flask import
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse # Corrected TwiML import

# Carregar variáveis de ambiente
load_dotenv()

# Baixar recursos necessários do NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('rslp')

# TOKEN do Bot (Telegram) - Will be replaced by Twilio credentials
# TOKEN = os.getenv('TELEGRAM_TOKEN') # Ensure this is or remains commented
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_WHATSAPP_NUMBER = os.getenv('TWILIO_WHATSAPP_NUMBER')

# Global variables for NLP model and data
vectorizer = None
X = None
df_global = None # Renamed to avoid conflict with df in preparar_modelo
app = Flask(__name__)
client = None # Will be initialized in main

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
    global df_global # Use the global df_global
    try:
        df_global = pd.read_csv('perguntas_respostas.csv', encoding='utf-8', quotechar='"')
        return df_global
    except Exception as e:
        print(f"Erro ao carregar dataset: {str(e)}")
        return None

# Preparar o modelo de similaridade
def preparar_modelo(current_df): # Renamed df to current_df to avoid confusion with global df_global
    current_df['texto_processado'] = current_df['pergunta'].apply(preprocessar_texto)
    global vectorizer, X # Use global vectorizer and X
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(current_df['texto_processado'])
    return vectorizer, X, current_df # Return them, though they are global now

# Palavras-chave para reconhecimento de perguntas administrativas (REMOVED as per stateless WhatsApp interaction)
# PALAVRAS_CHAVE = [
#     # Palavras interrogativas
#     'como', 'onde', 'quando', 'quem', 'qual', 'quais', 'o que', 'por que', 'porque', '?',
#     # Verbos de ação
#     'solicitar', 'contato', 'entrar', 'contatar', 'fazer', 'denunciar', 'denúncia',
#     'manutenção', 'manter', 'consertar', 'reparar', 'reclamar', 'reclamação',
#     # Termos específicos
#     'alvará', 'alvara', 'limpeza', 'terreno', 'baldio', 'horário', 'abre', 'funciona',
#     'poda', 'árvore', 'árvores', 'iluminação', 'pública', 'publica', 'luz', 'poste',
#     'sede', 'local', 'endereço', 'endereco', 'plano', 'piloto', 'regional',
#     # Verbos auxiliares
#     'posso', 'devo', 'preciso', 'quero', 'desejo', 'gostaria'
# ]

# Função para encontrar resposta mais similar
def encontrar_resposta(pergunta, current_vectorizer, current_X, current_df): # Parameters passed explicitly
    pergunta_processada = preprocessar_texto(pergunta)
    pergunta_vetor = current_vectorizer.transform([pergunta_processada])
    
    similaridades = cosine_similarity(pergunta_vetor, current_X)
    indice_mais_similar = similaridades.argmax()
    
    # Reduzindo o limiar de similaridade para 0.1
    if similaridades[0, indice_mais_similar] > 0.1:
        return current_df.iloc[indice_mais_similar]['resposta']
    else:
        # Tentar encontrar por palavras-chave específicas (Simplified: this part might need review for effectiveness without context)
        palavras_chave_respostas = { # Renamed to avoid conflict
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
            if palavra in palavras_chave_respostas:
                return palavras_chave_respostas[palavra]
        
        return "Desculpe, não encontrei uma resposta adequada para sua pergunta. Tente reformular ou perguntar de outra forma."

# Removed Telegram specific functions: start, handle_message, iniciar_perguntas_admin, button

@app.route("/whatsapp", methods=['POST'])
def whatsapp_reply():
    global vectorizer, X, df_global # Access global NLP model data
    # global client # Access global Twilio client

    incoming_msg = request.values.get('Body', '').lower()
    from_number = request.values.get('From', '') # User's WhatsApp number with "whatsapp:" prefix

    if not incoming_msg:
        return '', 200 # Or some error message

    # Logic to find answer using `encontrar_resposta`
    # Ensure vectorizer, X, and df_global are loaded before the first request
    if vectorizer is None or X is None or df_global is None:
        print("Error: NLP model not loaded.")
        # Potentially send an error message back to user or log
        return "Erro interno do servidor: modelo NLP não carregado.", 500

    resposta_bot = encontrar_resposta(incoming_msg, vectorizer, X, df_global)

    # print(f"Incoming message from {from_number}: {incoming_msg}")
    # print(f"Bot response: {resposta_bot}")

    # This is how to send a message
    if client and from_number and TWILIO_WHATSAPP_NUMBER: # Use the global variable
        try:
            client.messages.create(
                from_=f'whatsapp:{TWILIO_WHATSAPP_NUMBER}', # Bot's Twilio WhatsApp number
                body=resposta_bot,
                to=from_number # User's WhatsApp number
            )
            print(f"Message sent to {from_number}")
        except Exception as e:
            print(f"Error sending Twilio message: {e}")
    else:
        if not client:
            print("Twilio client not initialized.")
        if not from_number:
            print("Sender number missing.")
        if not TWILIO_WHATSAPP_NUMBER:
            print("Twilio WhatsApp number not configured in environment variables.")
        print("Cannot send reply due to missing Twilio client or configuration.")
    
    # For now, we'll just return the bot's response in the HTTP response for testing purposes.
    # Twilio expects an empty response or TwiML. Replying directly in HTTP response is not how it works.
    # This return is a placeholder for testing the `encontrar_resposta` logic via HTTP.
    # In a real Twilio setup, you'd return str(MessagingResponse()) or an empty string.
    # For now, returning an empty string and 200 OK as a typical webhook acknowledgement.
    # The actual reply is sent via client.messages.create()
    return '', 200

# Função principal
def main():
    global vectorizer, X, df_global, client # Declare them as global to modify

    try:
        # Carregar e preparar o dataset
        loaded_df = carregar_dataset() # Loads into df_global
        if loaded_df is None:
            print("Erro ao carregar o dataset. Verifique se o arquivo perguntas_respostas.csv existe.")
            return
            
        # preparar_modelo will use and set global vectorizer, X using df_global
        preparar_modelo(df_global)
        
        # Initialize Twilio client
        # account_sid and auth_token are now global variables loaded at the start of the script.
        if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN:
            client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
            print("Twilio client initialized.")
        else:
            print("Twilio credentials (TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN) not found in environment. Client not initialized.")

        # Run Flask app
        print('Starting Flask app for WhatsApp bot...')
        app.run(debug=True, port=5000) # Port 5000 is common for Flask dev
    except Exception as e:
        print(f'Erro ao iniciar o bot: {str(e)}')

if __name__ == '__main__':
    main()
