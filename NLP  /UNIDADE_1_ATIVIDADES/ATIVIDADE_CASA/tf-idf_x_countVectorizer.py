from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# dataset
from src.utils import load_movie_review_clean_dataset
corpus = load_movie_review_clean_dataset()

# Separando o corpus em features (textos) e labels (sentimentos)
texts = corpus['review']
labels = corpus['sentiment']

def preprocess_text(text):
    text = text.lower() # Convertendo para minúsculas
    text = re.sub(r'\d+', '', text) # Removendo números
    text = re.sub(r'[^\w\s]', '', text) # Removendo caracteres especiais
    stop_words = set(stopwords.words('english')) # Removendo stopwords
    lemmatizer = WordNetLemmatizer() # lematizando
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    return text

# Aplicando a função de pré-processamento
texts = texts.apply(preprocess_text)

# Dividindo o dataset em treino e teste
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Pipeline para CountVectorizer
count_pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])

# Pipeline para TF-IDF
tfidf_pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('classifier', MultinomialNB())
])

# Treinando e avaliando o modelo com CountVectorizer
count_pipeline.fit(X_train, y_train)
y_pred_count = count_pipeline.predict(X_test)

# Treinando e avaliando o modelo com TF-IDF
tfidf_pipeline.fit(X_train, y_train)
y_pred_tfidf = tfidf_pipeline.predict(X_test)

# métricas de avaliação para CountVectorizer
print("\nAvaliação do CountVectorizer:")
print(confusion_matrix(y_test, y_pred_count))
print(classification_report(y_test, y_pred_count))

# métricas de avaliação para TF-IDF
print("\nAvaliação do TF-IDF:")
print(confusion_matrix(y_test, y_pred_tfidf))
print(classification_report(y_test, y_pred_tfidf))

# Os dois tiveram desempenhos semelhantes, 77%.