import pandas as pd
from src.utils import load_df1_one_hot

df1 = load_df1_one_hot()

# Print os atributos do dataframe
print(df1.head())

# execute o one hot encoding em df1
df1_encoded = pd.get_dummies(df1, columns=['feature 1', 'feature 2', 'feature 3', 'feature 4','feature 5','label'])

# print os novos atributos
print(df1_encoded.columns)

# print as primeiras 5 linhas de df1
print(df1_encoded.head(5))


