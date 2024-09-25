# Atividade
# Uma função que retorne a quantidade de sentenças em um texto.

import re

def quantidade_sentencas(texto):
    # Considerando sentenças que terminam com ., ! ou ?
    sentencas = re.split(r'[.!?]+', texto)
    # Remover sentenças vazias que podem ser geradas pela divisão
    sentencas = [s for s in sentencas if s.strip()]
    return len(sentencas)

texto1 = "Esse primeiro texto é um teste para a primeira função. Que retorna a quantidade de sentenças."
print("\n",texto1)
print("Quantidade de sentenças no texto:")
print(quantidade_sentencas(texto1), "\n")


# Uma função que retorne a quantidade de palavras que começam com letra maiúscula em um texto.

def quantidade_palavras_maiusculas(texto):
    palavras = texto.split()
    count = sum(1 for palavra in palavras if palavra[0].isupper())
    return count

texto2 = "Esse segundo texto é um Teste para a segunda função. Que retorna a quantidade De palavras com letra Maiúscula."
print(texto2)
print("Quantidade de palavras maiúsculas no texto:")
print(quantidade_palavras_maiusculas(texto2), "\n")

#
# Uma função que retorne a quantidade de caracteres numéricos em um texto.

def quantidade_caracteres_numericos(texto):
    return sum(char.isdigit() for char in texto)

texto3 = "Esse terceir0 texto é um test3 para a terce1ra função, que ret0rna a quantidade de caractere5 numéricos."
print(texto3)
print("Quantidade de caracteres numéricos no texto:")
print(quantidade_caracteres_numericos(texto3), "\n")


# Uma função que retorne a quantidade de palavras que estão  em caixa alta.

def quant_palavras_caixa_alta(texto):
    palavras = texto.split()
    maiusculas = [palavra for palavra in palavras if palavra.isupper()]
    return len(maiusculas)

texto4 = "Esse QUARTO texto é um TESTE para a quarta função, que retorna a QUANTIDADE de palavras em caixa ALTA."
print(texto4)
print("Quantidade de palavras em caixa alta no texto:")
print(quant_palavras_caixa_alta(texto4), "\n")

# TESTAR TODAS AS FUNÇÕES COM UMA ÚNICA FRASE

print("Testar nas 4 funções a mesma frase:")
frase_geral = "Essa frase é GERAL. Para t0das as funções. Que RETORNA o resultad0 de cada funçã0."
print("Frase: ", frase_geral)
print("Quantd. de sentenças:")
print(quantidade_sentencas(frase_geral), "\n")
print("Quantd. de palavras maiúsculas:")
print(quantidade_palavras_maiusculas(frase_geral), "\n")
print("Quantd. de palavras em caixa alta:")
print(quant_palavras_caixa_alta(frase_geral), "\n")
print("Quantd. de caracteres numéricos:")
print(quantidade_caracteres_numericos(frase_geral), "\n")



# FAZER UM DATAFRAME COM 4 COLUNAS

import pandas as pd
dados = {
    'textos': ["Esse primeiro texto é um teste para a primeira função. Que retorna a quantidade de sentenças.",
               "Esse segundo texto é um Teste para a segunda função. Que retorna a quantidade De palavras com letra Maiúscula.",
               "Esse terceir0 texto é um test3 para a terce1ra função, que ret0rna a quantidade de caractere5 numéricos.",
               "Esse QUARTO texto é um TESTE para a quarta função, que retorna a QUANTIDADE de palavras em caixa ALTA."]
}

df = pd.DataFrame(dados)
print(df)

df['sentencas'] = df['textos'].apply(quantidade_sentencas)
df['palavras_letra_maiusc'] = df['textos'].apply(quantidade_palavras_maiusculas)
df['caract_numericos'] = df['textos'].apply(quantidade_caracteres_numericos)
df['palavras_caixa_alta'] = df['textos'].apply(quant_palavras_caixa_alta)

print(df)

