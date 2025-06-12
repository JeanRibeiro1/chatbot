import nltk

print("Baixando recursos do NLTK...")
try:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('rslp')
    print("Recursos do NLTK baixados com sucesso.")
except Exception as e:
    print(f"Erro ao baixar recursos do NLTK: {e}")
    # Levanta uma exceção para interromper o processo de deploy se o download falhar
    raise