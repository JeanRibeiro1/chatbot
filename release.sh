#!/bin/sh

# O comando 'set -e' garante que o script irá parar imediatamente se algum comando falhar.
set -e

# 1. Baixa os pacotes do NLTK
python download_nltk.py

# 2. Executa a migração do banco de dados
python migrar_remoto.py

# 3. Configura o webhook do Telegram
python set_webhook.py