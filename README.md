
# Projeto do curso  FIC - Introdução ao Python com Ciência de Dados

## 🎯 Objetivo

Este projeto tem como objetivo aplicar todas as etapas de um processo de Ciência de Dados, desde a coleta até a avaliação de modelos preditivos. Colocando em pratico o que foi discutido durante as aulas do curso. A proposta é analisar um conjunto de dados de filmes da IMDb e construir modelos capazes de prever a nota média (`IMDB_Rating`) dos filmes com base em suas características.

---

## 📂 Dataset

* **Fonte**: Kaggle - [IMDb Top 1000 Movies Dataset](https://www.kaggle.com/datasets/PromptCloudHQ/imdb-data)
* **Justificativa**: Este dataset foi escolhido por conter informações relevantes e ricas sobre filmes, como gênero, duração, votos, arrecadação, entre outros. Ele permite explorar diversas variáveis e aplicar técnicas de modelagem supervisionada para regressão.

---

## 🔍 Análise Exploratória

* Análise estatística descritiva das variáveis numéricas.
* Visualizações utilizando `seaborn` e `matplotlib`:

  * Matriz de correlação.
  * Distribuição de filmes por classificação indicativa (BR).
  * Relação entre tempo de duração e metascore.
  * Top 10 diretores com mais filmes no dataset.
* Tratamento de outliers implícito nas etapas de limpeza.
* Identificação e tratamento de valores ausentes em colunas como `Meta_score` e `Gross`.

---

## 🧹 Pré-processamento

* **Valores faltantes**:

  * `Meta_score` e `Gross`: preenchidos com a média.
  * `Released_Year`: valores inválidos foram substituídos pela mediana.
* **Transformações**:

  * Conversão de colunas para tipos adequados (`Runtime`, `Gross`).
  * Criação da coluna `Certificate_BR` a partir de `Certificate`, adaptando para classificações brasileiras.
* **Codificação**:

  * Gêneros (`Genre`): codificados com `MultiLabelBinarizer`.
  * Classificação indicativa (`Certificate_BR`): codificada com `OneHotEncoder`.
* **Redução de colunas**: Remoção de colunas irrelevantes como `Poster_Link`, `Overview`, `Stars` e `Series_Title`.

---

## 🤖 Modelagem

* **Divisão dos dados**: Holdout com `train_test_split` (80% treino, 20% teste).
* **Modelos aplicados**:

  1. **Regressão Linear**
  2. **Random Forest Regressor**
  3. **Gradient Boosting Regressor**
* **Justificativa**:

  * **Regressão Linear**: modelo base simples e interpretável.
  * **Random Forest**: modelo de ensemble robusto contra overfitting e com bom desempenho para dados tabulares.
  * **Gradient Boosting**: modelo avançado de boosting que melhora o desempenho em tarefas de regressão complexas.

---

## 📈 Avaliação

* **Métricas utilizadas**:

  * **MSE** (Erro Quadrático Médio)
  * **R²** (Coeficiente de Determinação)
* **Validação Cruzada**:

  * Validação cruzada com 5 folds (`cross_val_score`) para verificar a estabilidade dos modelos.
* **Comparação de modelos**:

  * O desempenho dos três algoritmos foi comparado com base nos resultados de MSE, R² e média da validação cruzada.

---

## 🛠️ Tecnologias Utilizadas

* Python 3
* Pandas, NumPy
* Seaborn, Matplotlib
* Scikit-learn

---

## ✍️ Autor

**Alexsandro da Silva**

---