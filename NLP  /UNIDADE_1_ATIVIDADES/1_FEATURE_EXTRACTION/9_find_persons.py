import spacy
from src.utils import load_text_tc

tc = load_text_tc()

# Load model and create Doc object
nlp = spacy.load('en_core_web_sm')


def find_persons(text):
    # Create Doc object
    doc = nlp(text)

    # Identify the persons
    persons = [ent.text for ent in doc.ents if ent.label_ == 'PERSON']

    # Return persons
    return persons


print(find_persons(tc))
