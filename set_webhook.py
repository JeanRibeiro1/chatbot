# set_webhook.py

import os
import asyncio
from telegram.ext import Application

# Carrega as variáveis de ambiente
TOKEN = os.getenv('TELEGRAM_TOKEN')
APP_NAME = os.getenv('FLY_APP_NAME')

async def main():
    if not TOKEN or not APP_NAME:
        print("Erro: As variáveis de ambiente TELEGRAM_TOKEN e FLY_APP_NAME são necessárias.")
        return

    application = Application.builder().token(TOKEN).build()
    webhook_url = f"https://{APP_NAME}.fly.dev"
    
    # Define o webhook no Telegram
    await application.bot.set_webhook(url=webhook_url)
    print(f"Webhook configurado com sucesso para a URL: {webhook_url}")

if __name__ == '__main__':
    asyncio.run(main())