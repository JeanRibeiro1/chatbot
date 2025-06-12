import os
import asyncio
from telegram.ext import Application
from dotenv import load_dotenv

# Carrega as variáveis de ambiente de um arquivo .env (se existir)
load_dotenv()

# Pega as credenciais do ambiente
TOKEN = os.getenv('TELEGRAM_TOKEN')
ADMIN_CHAT_ID = os.getenv('ADMIN_CHAT_ID')

async def main():
    """
    Envia uma mensagem de notificação para o admin quando o bot inicia.
    """
    if not TOKEN or not ADMIN_CHAT_ID:
        print("AVISO: As variáveis TELEGRAM_TOKEN e ADMIN_CHAT_ID são necessárias para a notificação de startup.")
        # Não paramos o deploy por causa disso, apenas avisamos.
        return

    print(f"Tentando enviar mensagem de startup para o chat ID: {ADMIN_CHAT_ID}")

    try:
        # Cria uma aplicação simples apenas para ter acesso ao objeto 'bot'
        application = Application.builder().token(TOKEN).build()

        # Envia a mensagem
        await application.bot.send_message(
            chat_id=ADMIN_CHAT_ID,
            text="Oi! O bot foi iniciado ou atualizado com sucesso no servidor. ✅"
        )
        print("Mensagem de startup enviada com sucesso.")

    except Exception as e:
        print(f"ERRO: Ocorreu um erro ao enviar a mensagem de startup: {e}")
        print("Lembre-se: O admin precisa ter iniciado uma conversa com o bot (/start) pelo menos uma vez.")

if __name__ == '__main__':
    asyncio.run(main())