from collections import Counter
import math

def saco_palavras(corpus):
    word_dict = {}
    for text in corpus:
        for word in text:
            if word not in word_dict:
                word_dict[word] = len(word_dict)
    return word_dict

def count_vectorizer(corpus):
    word_dict = saco_palavras(corpus)  # Cria dicionário de palavras

    count_vectors = []
    for text in corpus:
        vector = [0] * len(word_dict)
        word_count = Counter(text)
        for word, count in word_count.items():
            if word in word_dict:
                vector[word_dict[word]] = count
        count_vectors.append(vector)

    return count_vectors, word_dict

def tf_idf(corpus):
    word_dict = saco_palavras(corpus)  # Cria dicionário de palavras

    # Contagem de documentos por palavra
    doc_count = {}
    for text in corpus:
        unique_words = set(text)
        for word in unique_words:
            if word not in doc_count:
                doc_count[word] = 0
            doc_count[word] += 1

    # Cálculo do IDF
    idf = {}
    num_docs = len(corpus)
    for word, count in doc_count.items():
        idf[word] = math.log(num_docs / (count + 1)) + 1  # Smoothing adicionado

    # Cálculo do TF-IDF
    tf_idf_vectors = []
    for text in corpus:
        vector = [0] * len(word_dict)
        word_count = Counter(text)
        for word, count in word_count.items():
            if word in word_dict:
                tf = count / len(text)
                vector[word_dict[word]] = tf * idf[word]
        tf_idf_vectors.append(vector)

    return tf_idf_vectors, word_dict


# teste de funcionamento
print("\nTeste de funções(da segunda questão):\n")
corpus = [
    ['this', 'is', 'a', 'test'],
    ['this', 'test', 'is', 'only', 'a', 'test'],
    ['testing', 'is', 'fun']
]

count_vectors, word_dict = count_vectorizer(corpus)
print("Count Vectors:", count_vectors)

tf_idf_vectors, word_dict = tf_idf(corpus)
print("TF-IDF Vectors:", tf_idf_vectors)
