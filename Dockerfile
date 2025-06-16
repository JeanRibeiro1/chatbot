# Dockerfile - Versão Final Corrigida

FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# --- ORDEM CORRIGIDA ---

# 1. Copia PRIMEIRO o arquivo de dependências
COPY requirements.txt .

# Adicione estas linhas para instalar o Supercronic
ENV SUPERCRONIC_VERSION=v0.2.29
# Versão corrigida: Instala o curl, depois baixa o supercronic, e por fim limpa o cache
RUN apt-get update && apt-get install -y curl && \
    curl -fsSL "https://github.com/aptible/supercronic/releases/download/${SUPERCRONIC_VERSION}/supercronic-linux-amd64" -o /usr/local/bin/supercronic && \
    chmod +x /usr/local/bin/supercronic && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 2. Em seguida, INSTALA as dependências (incluindo o nltk)
RUN pip install --no-cache-dir -r requirements.txt

# 3. AGORA que o nltk já foi instalado, podemos usá-lo para baixar os pacotes de dados
RUN python -m nltk.downloader -d /usr/share/nltk_data punkt stopwords rslp

# --- FIM DA CORREÇÃO ---

# 4. Copia o resto dos arquivos do projeto (bot.py, .csv, etc.)
COPY . .

# 5. Expõe a porta que o Gunicorn (nosso servidor web) vai usar
EXPOSE 8080