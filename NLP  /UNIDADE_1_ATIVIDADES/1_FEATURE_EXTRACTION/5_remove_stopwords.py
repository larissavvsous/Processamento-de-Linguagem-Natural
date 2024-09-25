import spacy
from src.utils import load_gettyburg

# Load the Gettysburg Address text
gettysburg = load_gettyburg()

# Load model and create Doc object
nlp = spacy.load('en_core_web_sm')

# Load stopwords
stopwords = spacy.lang.en.stop_words.STOP_WORDS

# Create a Doc object
doc = nlp(gettysburg)

# Generate lemmatized tokens
lemmas = [token.lemma_ for token in doc]

# Remove stopwords and non-alphabetic tokens
a_lemmas = [lemma for lemma in lemmas if lemma.isalpha() and lemma.lower() not in stopwords]

# Print string after text cleaning
print(' '.join(a_lemmas))
