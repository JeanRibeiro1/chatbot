#!/bin/sh

# O comando 'set -e' garante que o script irá parar imediatamente se algum comando falhar.
set -e

# Pausa por 5 segundos para dar tempo ao banco de dados de iniciar.
echo "Aguardando 5 segundos para o banco de dados iniciar..."
sleep 5

# 1. Baixa os pacotes do NLTK
python download_nltk.py

echo "Executando a sincronização com a planilha no deploy..."
python worker.py
echo "Sincronização no deploy concluída."

# 2. Executa a migração do banco de dados
#python migrar_remoto.py

# 3. Configura o webhook do Telegram
python set_webhook.py

#python notify_startup.py