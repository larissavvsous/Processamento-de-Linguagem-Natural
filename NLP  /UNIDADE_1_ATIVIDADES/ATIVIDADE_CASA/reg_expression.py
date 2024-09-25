#atividade data  04/04

#10 funções para análise de expressões regulares

import re

# contagens de correspondências
print("\n1. Contagem de correspondências:")
def contar_python(texto_py):
    ocorrencias = re.findall(r'\bPython\b', texto_py, re.IGNORECASE)
    return len(ocorrencias)

texto_py = "Python é uma linguagem de programação. Muitas pessoas preferem Python porque ele é mais fácil de aprender."
contagem = contar_python(texto_py)
print(f'A palavra "Python" aparece {contagem} vezes no texto.')
print("\n")

# validação de email
print("2. Validação de email:")
def validar_email(email):
    padrao = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(padrao, email))

emails = [
    "maria@gmail.com",
    "larissasousa@gmail.com",  # Corrigido
    "sara.olga@outlook.com",
    "vitoria@gmail.com",  # Corrigido
    "cicero@yahoo.com"
]

for email in emails:
    if validar_email(email):
        print(f'"{email}" é um endereço de email válido.')
    else:
        print(f'"{email}" não é um endereço de email válido.')

print("\n")

# extração de números de telefone
print("3. Extração de números de telefone:")
def extrair_telefones(texto):
    padrao = r'\b(\d{2,3}[\s-]?)?(\d{4,5})[\s-]?(\d{4})\b'
    telefones_formatados = []

    for t in texto:
        telefones = re.findall(padrao, t)
        telefones_formatados.extend([''.join(telefone) for telefone in telefones])

    return telefones_formatados

lista_telefones = [
    "(21) 1234-5678",
    "9876-5432",
    "abc",
    "55 98765-4321",
    "(31)1234-5678",
    "def",
    "021 8765 4321",
    "87654321",
    "987654321"
]

telefones = extrair_telefones(lista_telefones)
print("4. Números de telefone encontrados:", telefones)
print("\n")

# substituição de palavras
print("Substituição de palavras:")
def substituir_gato_por_cachorro(texto_gato):
    padrao = r'\bgato\b'
    texto_substituido = [re.sub(padrao, 'cachorro', t) for t in texto_gato]
    return texto_substituido

texto_gato = [
    "O gato é um animal doméstico muito fofo.",
    "Existem várias raças de gato, como o gato persa e o gato siamês.",
    "Meu gato adora brincar com bolinhas.",
    "Meu gato tem pelo cinza."
]

novo_texto = substituir_gato_por_cachorro(texto_gato)
for t in novo_texto:
    print(t)

print("\n")

# extração de URLs
print("5. Extração de URLs:")
def extrair_urls(texto):
    padrao = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
    # Junta todas as strings da lista em uma única string
    texto_completo = ' '.join(texto)
    urls = re.findall(padrao, texto_completo)
    return urls

texto_url = [
    "https://www.exemplodeum.com/larissa",
    "http://outroexemplo.com.br",
    "larissavvsousa@alu.ufc.br",
    "água"
]

urls_encontradas = extrair_urls(texto_url)
print("URLs encontradas:", urls_encontradas)
print("\n")

# verificar senha segura
print("6. Verificação de senha segura:")
def validar_senha(senha):
    padrao = r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$'
    return bool(re.match(padrao, senha))

senhas = [
    "Senha123!",
    "outrasenha",
    "SenhaSegura@10",
    "12345678",
    "Senh@123"
]

for senha in senhas:
    if validar_senha(senha):
        print(f'A senha "{senha}" é segura.')
    else:
        print(f'A senha "{senha}" não é segura.')

print("\n")

# extração de palavras
print("7. Extração de palavras:")
def extrair_palavras(texto_palavras):
    padrao = r'\b\w+\b'
    palavras = re.findall(padrao, texto_palavras)
    return palavras

texto_palavras = "PyCharm é um Ambiente de Desenvolvimento Integrado usado para programar em Python."

palavras_encontradas = extrair_palavras(texto_palavras)
print("Palavras encontradas:", palavras_encontradas)
print("\n")

# validação de data
print("8. Validação de data:")
def validar_data(data):
    padrao = r'^(0[1-9]|[12][0-9]|3[01])/(0[1-9]|1[0-2])/\d{4}$'
    return bool(re.match(padrao, data))

datas = [
    "31/12/23",
    "06/11/2024",
    "15/1/2022",
    "25/13/2023",
    "30-12-2023",
    "08/05/2019"
]

for data in datas:
    if validar_data(data):
        print(f'A data "{data}" está no formato válido.')
    else:
        print(f'A data "{data}" não está no formato válido.')

print("\n")

# extração de nomes próprios
print("9. Extração de nomes próprios:")
def extrair_nomes_proprios(texto_nomes_prop):
    padrao = r'\b[A-ZÁÉÍÓÚÇÃÕ][a-záéíóúçãõ]*\b'
    nomes = re.findall(padrao, texto_nomes_prop)
    return nomes

texto_nomes_prop = "João joga Bola, Maria trabalha no shopping, Pedro é Motorista da UFC de Itapajé e Ana é blogueira."

nomes_encontrados = extrair_nomes_proprios(texto_nomes_prop)
print("Nomes próprios encontrados:", nomes_encontrados)
print("\n")

# contagem de vogais
print("10. Contagem de vogais:")
def contar_vogais(texto_vog):
    padrao = r'[aeiouAEIOU]'
    vogais = re.findall(padrao, texto_vog)
    return len(vogais)

texto_vog = "O céu é azul e o sol é amarelo."

num_vogais = contar_vogais(texto_vog)
print("Número de vogais:", num_vogais)