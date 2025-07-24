import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("./imdb.csv")
df.shape
df.columns
df.info()
df.head()
df.nunique()
df.isnull().sum()

dfLimp = df
dfLimp = dfLimp.drop_duplicates(subset=['Series_Title'],
                            keep='last').reset_index(drop=True)

dfLimp['Runtime'] = dfLimp['Runtime'].str.replace(" min", "",
                                                  regex=False).astype(int)

try:
  dfLimp['Released_Year'] = dfLimp['Released_Year'].astype(int)
except:
  maskInvalida = ~dfLimp['Released_Year'].astype(str).str.match(r'^\d+$')
  median_year = dfLimp.loc[~maskInvalida, 'Released_Year'].astype(int).median()
  dfLimp.loc[maskInvalida, 'Released_Year'] = median_year
  dfLimp['Released_Year'] = dfLimp['Released_Year'].astype(int)


mapa_certificado = {
    'G': 'Livre',
    'U': 'Livre',
    'A': 'Livre',
    'Approved': 'Livre',
    'Passed': 'Livre',

    'PG': '12',
    'GP': '12',
    'UA': '12',
    'U/A': '12',
    'TV-PG': '12',

    'PG-13': '14',
    'TV-14': '14',

    'R': '16',
    '16': '16',

    'TV-MA': '18',

    'Unrated': 'Não classificado',
    'Not Rated': 'Não classificado',
    None: 'Não classificado',
    np.nan: 'Não classificado'
}


dfLimp['Certificate_BR'] = dfLimp['Certificate'].map(mapa_certificado)
dfLimp['Certificate_BR'] = dfLimp['Certificate_BR'].fillna('Não classificado')
dfLimp['Meta_score'] = dfLimp['Meta_score'].fillna(dfLimp['Meta_score'].mean())
dfLimp['Gross'] = dfLimp['Gross'].str.replace(',', '')
dfLimp['Gross'] = pd.to_numeric(dfLimp['Gross'], errors='coerce')
dfLimp['Gross'] = dfLimp['Gross'].fillna(dfLimp['Gross'].mean())
dfLimp['Gross_Milhoes'] = dfLimp['Gross'] / 1_000_000
dfLimp = dfLimp.drop(['Poster_Link','Certificate','Star1', 'Star2','Star3', 'Star4', 'Gross'],axis=1)
dfLimp.shape
dfLimp.columns
dfLimp.describe()
numericas = dfLimp[['Released_Year', 'Runtime', 'IMDB_Rating', 'Meta_score', 'No_of_Votes', 'Gross_Milhoes']]
corr = numericas.corr()
plt.figure(figsize=(15,4))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de Correlação')
plt.show()


sns.scatterplot(data=dfLimp, x='Runtime', y='Meta_score')
plt.title('Runtime vs Meta_score')
plt.show()

plt.figure(figsize=(10, 5))
ax = sns.countplot(data=dfLimp, x='Certificate_BR', order=dfLimp['Certificate_BR'].value_counts().index)
plt.title('Número de Filmes por Classificação Indicativa (BR)')
plt.xlabel('Classificação')
plt.ylabel('Número de Filmes')
plt.tight_layout()


for ponto in ax.patches:
    count = int(ponto.get_height())
    ax.annotate(f'{count}',
                (ponto.get_x() + ponto.get_width() / 2., ponto.get_height()),
                ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.show()

plt.figure(figsize=(10, 4))
top_directors = dfLimp['Director'].value_counts().nlargest(10)
sns.barplot(x=top_directors.values, y=top_directors.index, color='#A4A9AD')
plt.title('Top 10 Diretores com Mais Filmes')
plt.xlabel('Número de Filmes')
plt.ylabel('Diretor')
plt.tight_layout()
plt.show()

dfLimp['Genre'] = dfLimp['Genre'].str.replace(' ', '')  
dfLimp['Genre'] = dfLimp['Genre'].str.split(',')


mlb = MultiLabelBinarizer()
generos = pd.DataFrame(mlb.fit_transform(dfLimp['Genre']),
                              columns=mlb.classes_,
                              index=dfLimp.index)

dfLimp = pd.concat([dfLimp.drop(columns=['Genre']), generos], axis=1)


dfLimp = dfLimp[dfLimp['Certificate_BR'] != 'Não classificado']


encoder = OneHotEncoder(sparse_output=False, dtype=int)

encoded = encoder.fit_transform(dfLimp[['Certificate_BR']])

colunas_cert = encoder.get_feature_names_out(['Certificate_BR'])

df_certificado = pd.DataFrame(encoded, columns=colunas_cert, index=dfLimp.index)

dfLimp = pd.concat([dfLimp.drop(columns=['Certificate_BR']), df_certificado], axis=1)

dfLimp.columns
dfLimp.info()

dfLimp = dfLimp.drop(['Series_Title', 'Overview','Director'],axis=1)

X = dfLimp.drop(columns=['IMDB_Rating'])
y = dfLimp['IMDB_Rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Instancia os modelos
modelos = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

for nome, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'\nModelo: {nome}')
    print(f'MSE: {mse:.2f}')
    print(f'R²: {r2:.2f}')


print("\nValidação Cruzada (R²):")
for nome, modelo in modelos.items():
    scores = cross_val_score(modelo, X, y, cv=5, scoring='r2')
    print(f"{nome}: Média = {scores.mean():.3f}, Desvio = {scores.std():.3f}")