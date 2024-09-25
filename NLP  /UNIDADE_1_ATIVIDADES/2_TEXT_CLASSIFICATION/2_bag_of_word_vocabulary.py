import pandas as pd
from src.utils import load_movie_review_dataset

# Import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

corpus = load_movie_review_dataset()
corpus = corpus['tagline'].dropna()

# Create CountVectorizer object
vectorizer = CountVectorizer()

# Generate matrix of word vectors
bow_matrix = vectorizer.fit_transform(corpus)

# Convert bow_matrix into a DataFrame
bow_df = pd.DataFrame(bow_matrix.toarray())

# Map the column names to vocabulary
bow_df.columns = vectorizer.get_feature_names_out()

# Print bow_df
print(bow_df)