import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title='Análise de agrupamentos - Iris', layout='wide')

PALETA = ['#1f3a5f', '#b5651d', '#4a7c59', '#7a3b3b', '#5b7f95',
          '#c2a14d', '#6b5b95', '#3f7f7f', '#9c6b4e', '#8a8d91']

st.markdown(
    '''
    <style>
    .block-container {padding-top: 2.2rem; max-width: 1200px;}
    #MainMenu, footer {visibility: hidden;}
    h1 {font-size: 1.7rem; font-weight: 600; letter-spacing: -0.01em;}
    h2, h3 {font-weight: 600; letter-spacing: -0.005em;}
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #dfe3e8;
        border-radius: 4px;
        padding: 0.7rem 0.9rem;
    }
    div[data-testid="stMetricLabel"] p {font-size: 0.8rem; color: #5b6573;}
    div[data-testid="stMetricValue"] {font-size: 1.4rem;}
    button[data-baseweb="tab"] {font-size: 0.95rem;}
    </style>
    ''',
    unsafe_allow_html=True,
)


@st.cache_data
def carregar_dados():
    iris = load_iris(as_frame=True)
    x = iris.data.copy()
    x.columns = ['Comprimento da sépala', 'Largura da sépala',
                 'Comprimento da pétala', 'Largura da pétala']
    especies = pd.Series(iris.target).map(dict(enumerate(iris.target_names)))
    return x, especies


@st.cache_data
def avaliar_k(x_padronizado, k_min=2, k_max=10):
    linhas = []
    for k in range(k_min, k_max + 1):
        rotulos = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(x_padronizado)
        linhas.append({'K': k, 'Silhueta': silhouette_score(x_padronizado, rotulos)})
    return pd.DataFrame(linhas)


def ajustar_figura(fig, altura=420):
    fig.update_layout(
        template='simple_white',
        height=altura,
        margin=dict(l=10, r=10, t=50, b=10),
        title=dict(font=dict(size=15)),
        legend=dict(title=None),
        colorway=PALETA,
    )
    return fig


x, especies = carregar_dados()
x_padronizado = StandardScaler().fit_transform(x)
resultado_df = avaliar_k(x_padronizado)
melhor = resultado_df.loc[resultado_df['Silhueta'].idxmax()]
melhor_k = int(melhor['K'])

st.title('Análise de agrupamentos na base Iris')
st.caption('K-Means não supervisionado, com avaliação pelo coeficiente de silhueta.')

with st.sidebar:
    st.header('Informações')
    st.markdown(f'Melhor K pela silhueta: **{melhor_k}**')
    st.divider()
    st.caption(
        'As variáveis são padronizadas antes do ajuste. '
        'A espécie real não é usada no treinamento.'
    )

aba_dados, aba_k, aba_clusters, aba_notas = st.tabs(
    ['Dados', 'Escolha de K', 'Agrupamento', 'Notas metodológicas'])

with aba_dados:
    st.subheader('Base de dados')
    st.write('São 150 amostras de flores com quatro medidas em centímetros. '
             'A espécie é exibida apenas como referência.')
    c1, c2, c3 = st.columns(3)
    c1.metric('Amostras', len(x))
    c2.metric('Variáveis', x.shape[1])
    c3.metric('Espécies', especies.nunique())

    tabela = x.copy()
    tabela['Espécie (referência)'] = especies.values
    st.dataframe(tabela, use_container_width=True, height=320)

    st.subheader('Estatísticas descritivas')
    st.dataframe(x.describe().round(2), use_container_width=True)

with aba_k:
    st.subheader('Número de clusters e coeficiente de silhueta')
    st.write('O K-Means foi executado para K de 2 a 10. '
             'Valores maiores de silhueta indicam grupos mais coesos e mais separados entre si.')
    c1, c2 = st.columns(2)
    c1.metric('K com maior silhueta', melhor_k)
    c2.metric('Silhueta máxima', f"{melhor['Silhueta']:.3f}")

    fig = px.line(resultado_df, x='K', y='Silhueta', markers=True,
                  title='Silhueta por número de clusters')
    fig.update_traces(line_color=PALETA[0], marker=dict(size=7))
    fig.add_vline(x=melhor_k, line_dash='dot', line_color='#8a8d91')
    fig.update_xaxes(dtick=1)
    st.plotly_chart(ajustar_figura(fig), use_container_width=True)

    st.dataframe(resultado_df.style.format({'Silhueta': '{:.4f}'}),
                 use_container_width=True, hide_index=True)

with aba_clusters:
    st.subheader('Visualização dos agrupamentos')

    st.write(
        'Altere o número de clusters para observar '
        'como o K-Means organiza as flores.'
    )

    k_escolhido = st.slider(
        'Número de clusters (K)',
        min_value=2,
        max_value=10,
        value=melhor_k,
        step=1,
        key='slider_clusters'
    )

    st.info(
        f'K selecionado: {k_escolhido} | '
        f'Melhor K pela silhueta: {melhor_k}'
    )

    modelo = KMeans(
        n_clusters=k_escolhido,
        random_state=42,
        n_init=10
    )
    grupos = modelo.fit_predict(x_padronizado)
    silhueta = silhouette_score(x_padronizado, grupos)

    c1, c2 = st.columns(2)
    c1.metric('K selecionado', k_escolhido)
    c2.metric('Silhueta', f'{silhueta:.3f}')

    coordenadas = PCA(n_components=2).fit_transform(x_padronizado)
    rotulo_grupo = [f'Grupo {g + 1}' for g in grupos]
    visual = pd.DataFrame({
        'Componente 1': coordenadas[:, 0],
        'Componente 2': coordenadas[:, 1],
        'Cluster': rotulo_grupo,
        'Espécie (referência)': especies.values,
    })
    fig = px.scatter(visual, x='Componente 1', y='Componente 2', color='Cluster',
                     hover_data=['Espécie (referência)'],
                     category_orders={'Cluster': sorted(set(rotulo_grupo))},
                     title='Projeção dos clusters em duas componentes principais')
    fig.update_traces(marker=dict(size=8, opacity=0.85, line=dict(width=0)))
    st.plotly_chart(ajustar_figura(fig, 480), use_container_width=True)
    st.caption('O K-Means usa as quatro variáveis padronizadas. '
               'O PCA serve apenas para a visualização em duas dimensões.')

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('**Tamanho de cada grupo**')
        contagem = (pd.Series(rotulo_grupo).value_counts().sort_index()
                    .rename_axis('Cluster').reset_index(name='Amostras'))
        st.dataframe(contagem, use_container_width=True, hide_index=True)
    with col_b:
        st.markdown('**Clusters x espécie de referência**')
        cruzada = pd.crosstab(pd.Series(especies.values, name='Espécie'),
                              pd.Series(rotulo_grupo, name='Cluster'))
        st.dataframe(cruzada, use_container_width=True)

    exportacao = x.copy()
    exportacao['Cluster'] = grupos + 1
    st.download_button('Baixar resultados (CSV)',
                       exportacao.to_csv(index=False).encode('utf-8-sig'),
                       file_name='iris_clusters.csv', mime='text/csv')

with aba_notas:
    st.subheader('Notas metodológicas')
    st.markdown(f'''
**Resultado da busca por K.** Entre 2 e 10, o maior coeficiente de silhueta ocorre em K = {melhor_k} ({melhor['Silhueta']:.3f}).

**Método**

- **K-Means:** agrupa as amostras pela proximidade entre suas medidas, sem usar a espécie.
- **Padronização:** coloca as quatro variáveis em escala comparável antes do ajuste.
- **Silhueta:** varia de -1 a 1. Valores mais altos indicam grupos mais coesos e separados.
- **PCA:** reduz os dados a dois eixos, somente para visualização.

**Limitações**

- A maior silhueta não implica que os clusters coincidam com as três espécies. Duas delas se sobrepõem nas medidas, e é comum que a silhueta favoreça K = 2.
- O K escolhido é o melhor entre os valores testados segundo essa métrica, não uma descrição definitiva da estrutura dos dados.
''')