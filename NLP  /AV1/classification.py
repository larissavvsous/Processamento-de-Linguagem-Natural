from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from preprocessing import preprocessamento, stemming
from vectorizer import count_vectorizer, tf_idf

# Carregar datasets
coment_pre_process = pd.read_csv("content_lemmatized_stemmed.csv")
coment_sem_tratamento = pd.read_csv("content_sem_trat.csv")

print('\nVerificar o balanceamento das classes:')
class_distribution = coment_pre_process['score'].value_counts()
print(class_distribution)

# Extraindo as labels
labels = coment_pre_process['score']

# a) Utilizando a vetorização por TF-IDF, compare os resultados de acerto do classificador
# com pré-processamento e sem pré-processamento. Mostre as taxas para os casos de forma organizada.

# Vetorização TF-IDF
tf_idf_vectors_preprocessed, _ = tf_idf(coment_pre_process['content_stemmed'])
tf_idf_vectors_raw, _ = tf_idf(coment_sem_tratamento['content'])

# Separação dos dados
X_train_preprocessed, X_test_preprocessed, y_train, y_test = train_test_split(tf_idf_vectors_preprocessed, labels, test_size=0.3, random_state=42)
X_train_raw, X_test_raw, _, _ = train_test_split(tf_idf_vectors_raw, labels, test_size=0.3, random_state=42)

# Definir o classificador
model = MultinomialNB()

# Definir o grid de parâmetros para otimizar
param_grid = {'alpha': [0.1, 0.5, 1.0, 2.0, 5.0]}

# GridSearchCV para pré-processamento
grid_search_preprocessed = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search_preprocessed.fit(X_train_preprocessed, y_train)

# Melhor modelo para dados pré-processados
best_model_preprocessed = grid_search_preprocessed.best_estimator_
y_pred_preprocessed = best_model_preprocessed.predict(X_test_preprocessed)

# GridSearchCV para dados sem tratamento
grid_search_raw = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search_raw.fit(X_train_raw, y_train)

# Melhor modelo para dados sem pré-processamento
best_model_raw = grid_search_raw.best_estimator_
y_pred_raw = best_model_raw.predict(X_test_raw)

print("TF-IDF com pré-processamento (melhor alpha: {:.2f}):".format(grid_search_preprocessed.best_params_['alpha']))
print(classification_report(y_test, y_pred_preprocessed, zero_division=1))
print(confusion_matrix(y_test, y_pred_preprocessed))

print("TF-IDF sem pré-processamento (melhor alpha: {:.2f}):".format(grid_search_raw.best_params_['alpha']))
print(classification_report(y_test, y_pred_raw, zero_division=1))
print(confusion_matrix(y_test, y_pred_raw))

# b) Comparação entre CountVectorizer e TF-IDF
count_vectors, _ = count_vectorizer(coment_pre_process['content_stemmed'])
X_train_count, X_test_count, _, _ = train_test_split(count_vectors, labels, test_size=0.3, random_state=42)

grid_search_count = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search_count.fit(X_train_count, y_train)

# Melhor modelo para CountVectorizer
best_model_count = grid_search_count.best_estimator_
y_pred_count = best_model_count.predict(X_test_count)

print("CountVectorizer com pré-processamento (melhor alpha: {:.2f}):".format(grid_search_count.best_params_['alpha']))
print(classification_report(y_test, y_pred_count, zero_division=1))
print(confusion_matrix(y_test, y_pred_count))

# c) Comparação entre lematização e stemming
coment_pre_process['content_stemmed'] = coment_pre_process['content'].apply(lambda x: preprocessamento(x, use_lemmatization=False))
coment_pre_process['content_lemmatized'] = coment_pre_process['content'].apply(lambda x: preprocessamento(x, use_lemmatization=True))

tf_idf_vectors_stemmed, _ = tf_idf(coment_pre_process['content_stemmed'])
tf_idf_vectors_lemmatized, _ = tf_idf(coment_pre_process['content_lemmatized'])

X_train_stemmed, X_test_stemmed, _, _ = train_test_split(tf_idf_vectors_stemmed, labels, test_size=0.3, random_state=42)
X_train_lemmatized, X_test_lemmatized, _, _ = train_test_split(tf_idf_vectors_lemmatized, labels, test_size=0.3, random_state=42)

grid_search_stemmed = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search_stemmed.fit(X_train_stemmed, y_train)
best_model_stemmed = grid_search_stemmed.best_estimator_
y_pred_stemmed = best_model_stemmed.predict(X_test_stemmed)

grid_search_lemmatized = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search_lemmatized.fit(X_train_lemmatized, y_train)
best_model_lemmatized = grid_search_lemmatized.best_estimator_
y_pred_lemmatized = best_model_lemmatized.predict(X_test_lemmatized)

print("TF-IDF com stemming (melhor alpha: {:.2f}):".format(grid_search_stemmed.best_params_['alpha']))
print(classification_report(y_test, y_pred_stemmed, zero_division=1))
print(confusion_matrix(y_test, y_pred_stemmed))

print("TF-IDF com lemmatização (melhor alpha: {:.2f}):".format(grid_search_lemmatized.best_params_['alpha']))
print(classification_report(y_test, y_pred_lemmatized, zero_division=1))
print(confusion_matrix(y_test, y_pred_lemmatized))

# fazer o push dessa ativ