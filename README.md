# Iris Cluster Analyzer

**Avaliação 2 — Aprendizado Não Supervisionado (K-Means)**

**Universidade:** Universidade Federal do Tocantins (UFT)  
**Curso:** Ciência da Computação  
**Disciplina:** Inteligência Artificial  
**Discente:** Ester Arraiz de Matos

---

## 1. Sobre o projeto

O **Iris Cluster Analyzer** é uma aplicação interativa desenvolvida em Python, utilizando o framework Streamlit, com o objetivo de demonstrar a aplicação do algoritmo de agrupamento **K-Means** e a avaliação da qualidade dos agrupamentos por meio do **coeficiente de silhueta (Silhouette Score)**.

O projeto utiliza o conjunto de dados Iris, disponibilizado pelo UCI Machine Learning Repository e acessível pela biblioteca Scikit-learn.

A aplicação permite explorar os dados, identificar a quantidade mais adequada de clusters, visualizar os agrupamentos e analisar como diferentes valores de K influenciam os resultados.

## 2. Aprendizado não supervisionado

O aprendizado não supervisionado é uma abordagem de aprendizado de máquina na qual os algoritmos identificam padrões e estruturas nos dados sem utilizar respostas ou classificações previamente fornecidas durante o treinamento.

Diferentemente do aprendizado supervisionado, em que o modelo aprende a partir de exemplos rotulados, o aprendizado não supervisionado busca descobrir relações e semelhanças entre os registros.

Neste projeto, o K-Means analisa as características das flores sem utilizar suas espécies como referência para formar os grupos.

### Algoritmo K-Means

O K-Means é um algoritmo de agrupamento que organiza os dados em K grupos, denominados *clusters*, com base na similaridade entre suas características.

Seu funcionamento consiste em:

1. Definir a quantidade K de clusters.
2. Inicializar K centroides.
3. Associar cada amostra ao centroide mais próximo.
4. Recalcular os centroides com base nas amostras atribuídas.
5. Repetir o processo até atingir um critério de convergência.

O objetivo é formar grupos cujos elementos apresentem maior similaridade entre si.

## 3. Base de dados Iris

O dataset Iris contém 150 amostras de flores pertencentes a três espécies:

- Iris Setosa
- Iris Versicolor
- Iris Virginica

Cada amostra possui quatro características numéricas, medidas em centímetros:

- Comprimento da sépala
- Largura da sépala
- Comprimento da pétala
- Largura da pétala

As espécies são utilizadas apenas como referência para interpretação dos resultados, não participando do treinamento do K-Means.

**Fonte:** [Iris — UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/53/iris)

## 4. Tecnologias utilizadas

| Tecnologia | Finalidade |
|---|---|
| Python | Linguagem de programação |
| Streamlit | Interface web interativa |
| Pandas | Manipulação e análise dos dados |
| Scikit-learn | K-Means, padronização, PCA e silhueta |
| Plotly | Visualização interativa dos resultados |

## 5. Metodologia

O desenvolvimento foi organizado nas seguintes etapas:

**Etapa 1 — Carregamento dos dados**

Utilização da função `load_iris()` da biblioteca Scikit-learn para obter as 150 amostras e suas quatro características numéricas.

**Etapa 2 — Padronização**

Aplicação do `StandardScaler` para transformar as características em variáveis com média zero e desvio-padrão unitário, evitando que diferenças de escala influenciem indevidamente o agrupamento.

**Etapa 3 — Aplicação do K-Means**

Execução do algoritmo para diferentes quantidades de clusters, variando de K = 2 até K = 10.

Foram utilizados os parâmetros:

- `n_clusters`: quantidade de grupos.
- `random_state=42`: reprodutibilidade dos resultados.
- `n_init=10`: número de inicializações do algoritmo.

**Etapa 4 — Avaliação pelo coeficiente de silhueta**

Para cada valor de K, é calculado o coeficiente de silhueta utilizando as quatro características padronizadas.

A métrica varia de -1 a 1. Valores mais elevados indicam, em geral, grupos mais coesos internamente e mais separados entre si.

**Etapa 5 — Seleção do melhor K**

O sistema compara os coeficientes obtidos e identifica automaticamente o valor de K que apresenta a maior silhueta no intervalo analisado.

**Etapa 6 — Visualização dos agrupamentos**

Utilização da Análise de Componentes Principais (PCA) para representar os dados em duas dimensões, permitindo visualizar os clusters em gráficos de dispersão.

O PCA é utilizado somente para visualização, não para o treinamento do K-Means.

## 6. Funcionalidades da aplicação

A interface está organizada em quatro seções:

- **Dados:** apresenta o conjunto Iris, suas características e estatísticas descritivas.
- **Escolha de K:** apresenta a comparação dos coeficientes de silhueta e identifica o melhor K.
- **Agrupamento:** permite selecionar diferentes valores de K e visualizar os clusters gerados.
- **Notas metodológicas:** apresenta explicações sobre as técnicas utilizadas e as limitações da análise.

A aplicação também permite exportar os resultados dos agrupamentos em formato CSV.

## 7. Como executar

### Pré-requisitos

- Python 3.10 ou superior.
- Pip instalado.
- Terminal ou ambiente de desenvolvimento, como VS Code.

### Instalação

Clone o repositório ou baixe os arquivos do projeto.

Abra o terminal na pasta do projeto e crie um ambiente virtual:

```bash
python -m venv .venv
```

Ative o ambiente virtual.

**Windows — PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

**Linux/macOS:**

```bash
source .venv/bin/activate
```

Instale as dependências:

```bash
python -m pip install -r requirements.txt
```

Inicie a aplicação:

```bash
python -m streamlit run app.py
```

A interface estará disponível, normalmente, em:

`http://localhost:8501`

## 8. Resultados e interpretação

A aplicação permite comparar diferentes quantidades de clusters e identificar aquela que apresenta o maior coeficiente de silhueta entre os valores avaliados.

É importante destacar que o dataset Iris contém três espécies conhecidas, mas o valor de K com maior silhueta não precisa necessariamente ser igual a três.

Isso ocorre porque o algoritmo considera a similaridade entre as características numéricas das flores, enquanto o coeficiente de silhueta avalia a coesão e a separação dos grupos formados.

Portanto, o melhor K identificado representa a configuração com maior pontuação segundo a métrica adotada, dentro do intervalo testado.

## 9. Considerações finais

O projeto demonstra a aplicação prática de técnicas de aprendizado não supervisionado, utilizando o algoritmo K-Means para identificar padrões em um conjunto de dados real.

A utilização do coeficiente de silhueta permite avaliar quantitativamente diferentes configurações de agrupamento, enquanto a interface Streamlit facilita a exploração e a interpretação visual dos resultados.

Dessa forma, a aplicação contribui para a compreensão dos conceitos de agrupamento, padronização de dados, seleção de parâmetros e avaliação de modelos de aprendizado de máquina.

## 10. Referências

UCI MACHINE LEARNING REPOSITORY. **Iris Dataset**. Disponível em: https://archive.ics.uci.edu/dataset/53/iris.

SCIKIT-LEARN. **KMeans**. Disponível em: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html.

SCIKIT-LEARN. **Silhouette Score**. Disponível em: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html.

STREAMLIT. **Documentation**. Disponível em: https://docs.streamlit.io/.