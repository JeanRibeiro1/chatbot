# Dockerfile - Versão Final Corrigida

FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# --- ORDEM CORRIGIDA ---

# 1. Copia PRIMEIRO o arquivo de dependências
COPY requirements.txt .

# 2. Em seguida, INSTALA as dependências (incluindo o nltk)
RUN pip install --no-cache-dir -r requirements.txt

# 3. AGORA que o nltk já foi instalado, podemos usá-lo para baixar os pacotes de dados
RUN python -m nltk.downloader -d /usr/share/nltk_data punkt stopwords rslp

# --- FIM DA CORREÇÃO ---

# 4. Copia o resto dos arquivos do projeto (bot.py, .csv, etc.)
COPY . .

# 5. Expõe a porta que o Gunicorn (nosso servidor web) vai usar
EXPOSE 8080