from src.utils import load_movie_review_dataset

# Import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

corpus = load_movie_review_dataset()
corpus = corpus['tagline'].dropna()

# Create CountVectorizer object
vectorizer = CountVectorizer()

# Generate matrix of word vectors
bow_matrix = vectorizer.fit_transform(corpus)

# Print the shape of bow_matrix
print(bow_matrix.shape)