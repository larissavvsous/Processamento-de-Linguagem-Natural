import pandas as pd

# Frases
sentences = [
    'Believe in your dreams',
    'Make your dreams happen',
    'Fight for your dreams'
]

frases = pd.DataFrame(sentences, columns=['Frase'])
print(frases)

print('\n')

# Tabela de frequência de termos (TF)
tab_freq_terms = {
    'Termo': ['Believe', 'in', 'your', 'dreams', 'Make', 'happen', 'Fight', 'for'],
    'Frase 0': [1/4, 1/4, 1/4, 1/4, 0, 0, 0, 0],
    'Frase 1': [0, 0, 1/4, 1/4, 1/4, 1/4, 0, 0],
    'Frase 2': [0, 0, 1/4, 1/4, 0, 0, 1/4, 1/4]
}

tab = pd.DataFrame(tab_freq_terms)
print(tab)
