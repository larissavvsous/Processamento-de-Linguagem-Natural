# AVALIAÇÃO 1
> Orientações para execução da prova.

Esse documento exibe as descrições das questões e a relação dos datasets que devem ser utiizados 
pelos alunos e alunas.

O modelo de documento seguinte mostra como você deve registrar por escrito o desenvolvimento. 
https://docs.google.com/document/d/1hIwPx9W-k3LnXRJrkWYTsbrtx4NfP88_/edit?usp=sharing&ouid=118351454454462119994&rtpof=true&sd=true

##  Aluno - Dataset

BRUNA BARRETO MESQUITA: https://www.kaggle.com/datasets/fredericods/ptbr-sentiment-analysis-datasets

JOAO DAVI OLIVEIRA BARBOSA: https://www.kaggle.com/datasets/hrmello/brazilian-portuguese-hatespeech-dataset

LAURA DE LIMA MENDES: https://www.kaggle.com/datasets/fredericods/ptbr-sentiment-analysis-datasets?select=buscape.csv

LARISSA VITÓRIA: https://www.kaggle.com/datasets/shivkumarganesh/plenty-of-fish-google-play-store-reviews

MARIA BIANCA SOUSA COSTA: https://www.kaggle.com/datasets/moesiof/portuguese-narrative-essays

MARIA VANESSA SOUSA MESQUITA : https://github.com/kamplus/FakeNewsSetGen/tree/master/Dataset

MATEUS SILVA MATOS: https://www.kaggle.com/datasets/brunoluvizotto/brazilian-headlines-sentiments

PEDRO COELHO SAMPAIO FILHO: https://huggingface.co/datasets/nilc-nlp/assin

RUAN RODRIGUES SOUSA: https://huggingface.co/datasets/johnidouglas/twitter-sentiment-pt-BR-md-2-l

THAYS FERREIRA UCHOA ALBUQUERQUE: https://huggingface.co/datasets/AiresPucrs/sentiment-analysis-pt


### Questão 1

[preprocessing.py](preprocessing.py)

Nesta primeira questão você deve implementar funções de manipulação do dataset realizar os pré-processmentos necessários, como stemming, lemmatização, remoção de carateris maiusculos, verificar stopwords, verificação de menções
, de acordo as características do seu dataset. Em resume prepare o mesmo para aplicação de extração de atributos. A estrutura do código deve permitir que possam ser importadas as funções em outras questões.


### Questão 2

[vectorizer.py](vectorizer.py)

Voce deve implementar funções para extração de atributos com CountVectorizer e TF-IDF. Faça as duas funções implementadas manualmente, sem funções prontas para criar dicionário de palavras e os cálculos.
A estrutura do código deve permitir que possam ser importadas as funções em outras questões.


### Questão 3

[classification.py](classification.py)


Neste exercicio você deve utilizar um único classifiador para aplicar no seu dataset, de acordo com a label escolhida.
Voce deve comparar os resultados quantitiativos nos seguintes casos:

a) Utiizando a vetorização por TF-IDF, compare os resultados de acerto do classificador com pré-processamento e sem pré-processamento. Mostre as taxas para os casos de forma organizada.

b) Compare as vetorizações CountVectorizer x TF-IDF, usando com pré-processamento que você escolheu no item a)

C) Faça um variação de dois pré-procesamentos que compare lemmatização e steming, considerando a melhor forma de vetorização vista no item b)

 

### Observações para o Relatório 

Discutir **organizadamente** os resultados obtidos de cada questão.
Ao concluir o relatório, compartilhar com **alysonbnr@ufc.br** até 15/08


### Observações para o Apresentação

Criar apresentação para realizar até 15/08