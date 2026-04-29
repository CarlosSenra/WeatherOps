import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

class AnaliseGraficaEDA:
    def __init__(self, df): 
        """
        Inicializa a classe, cria uma cópia do DataFrame para não alterar o original,
        garante que a coluna de data seja datetime e ordena os dados.
        """
        self.df = df.copy()
        
        
        if 'data_hora' in self.df.columns:
            self.df['data_hora'] = pd.to_datetime(self.df['data_hora'])
            self.df = self.df.sort_values(by='data_hora')
        else:
            print("Atenção: Coluna 'data_hora' não encontrada. Alguns gráficos de linha podem falhar.")

    def plot_serie_temporal(self, coluna, titulo=None):
        """Gera um gráfico de linha interativo para uma variável ao longo do tempo."""
        if titulo is None:
            titulo = f"Série Temporal: {coluna}"
            
        fig = px.line(self.df, x='data_hora', y=coluna, title=titulo,
                      labels={'data_hora': 'Data e Hora', coluna: coluna})
        
        
        fig.update_xaxes(rangeslider_visible=True)
        fig.show()

    def plot_temperaturas_conjuntas(self):
        """Plota as temperaturas Máxima, Mínima e Média/Ar no mesmo gráfico."""
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=self.df['data_hora'], y=self.df['temp_max_c'],
                                 mode='lines', name='Temp Máx', line=dict(color='red')))
        fig.add_trace(go.Scatter(x=self.df['data_hora'], y=self.df['temp_ar_c'],
                                 mode='lines', name='Temp Ar', line=dict(color='green')))
        fig.add_trace(go.Scatter(x=self.df['data_hora'], y=self.df['temp_min_c'],
                                 mode='lines', name='Temp Mín', line=dict(color='blue')))

        fig.update_layout(title='Dinâmica Diária das Temperaturas (°C)',
                          xaxis_title='Data e Hora',
                          yaxis_title='Temperatura (°C)',
                          hovermode="x unified") 
        fig.show()

    def plot_distribuicao(self, coluna):
        """Gera um histograma com boxplot marginal para ver a distribuição dos dados."""
        fig = px.histogram(self.df, x=coluna, marginal="box", 
                           title=f"Distribuição de {coluna}",
                           labels={coluna: coluna})
        fig.show()

    def plot_matriz_correlacao(self):
        """Gera um heatmap de correlação (Pearson) entre as variáveis numéricas."""
        # Seleciona apenas as colunas numéricas, removendo a data
        df_numerico = self.df.select_dtypes(include=['float64', 'int64'])
        correlacao = df_numerico.corr()

        fig = px.imshow(correlacao, text_auto='.2f', aspect="auto",
                        color_continuous_scale='RdBu_r', 
                        title="Matriz de Correlação das Variáveis Meteorológicas")
        fig.show()
        
    def resumo_dados_ausentes(self):
        """Retorna um DataFrame mostrando o percentual de dados nulos por coluna."""
        nulos = self.df.isnull().sum()
        percentual = (nulos / len(self.df)) * 100
        resumo = pd.DataFrame({'Nulos': nulos, 'Percentual (%)': percentual}).sort_values(by='Nulos', ascending=False)
        return resumo[resumo['Nulos'] > 0]
    

class AnaliseGraficaAnualEDA: 
    def __init__(self, df):
        """
        Inicializa a classe e prepara os dados extraindo Ano, Mês e Dia.
        """
        self.df = df.copy()
        
        if 'data_hora' in self.df.columns:
            self.df['data_hora'] = pd.to_datetime(self.df['data_hora'])
            self.df['ano'] = self.df['data_hora'].dt.year
            self.df['mes'] = self.df['data_hora'].dt.month
            self.df['nome_mes'] = self.df['data_hora'].dt.strftime('%b')
            self.df['dia_do_ano'] = self.df['data_hora'].dt.dayofyear
            self.df['hora'] = self.df['data_hora'].dt.hour
            self.df['data_sem_hora'] = self.df['data_hora'].dt.date
        else:
            raise ValueError("A coluna 'data_hora' é obrigatória para esta análise.")

    def comparar_distribuicao_anual(self, coluna):
        fig = px.box(self.df, x='ano', y=coluna, color='ano',
                     title=f'Comparação Anual - Distribuição de {coluna}',
                     labels={'ano': 'Ano', coluna: coluna})
        fig.update_xaxes(type='category')
        fig.show()

    def evolucao_mensal_por_ano(self, coluna, agregacao='mean'):
        if agregacao == 'mean':
            df_agregado = self.df.groupby(['ano', 'mes'])[coluna].mean().reset_index()
            titulo = f'Média Mensal de {coluna} por Ano'
        elif agregacao == 'sum':
            df_agregado = self.df.groupby(['ano', 'mes'])[coluna].sum().reset_index()
            titulo = f'Total Acumulado Mensal de {coluna} por Ano'
        else:
            raise ValueError("Agregação deve ser 'mean' ou 'sum'.")

        fig = px.line(df_agregado, x='mes', y=coluna, color='ano', markers=True,
                      title=titulo, labels={'mes': 'Mês', coluna: coluna})
        fig.update_xaxes(tickmode='linear', tick0=1, dtick=1)
        fig.update_traces(line=dict(width=3))
        fig.show()

    def acumulado_anual_precipitacao(self):
        if 'precipitacao_total_mm' not in self.df.columns:
            print("A coluna 'precipitacao_total_mm' não existe no DataFrame.")
            return

        df_chuva = self.df.groupby('ano')['precipitacao_total_mm'].sum().reset_index()
        fig = px.bar(df_chuva, x='ano', y='precipitacao_total_mm', text_auto='.1f',
                     title='Precipitação Total Acumulada por Ano',
                     labels={'ano': 'Ano', 'precipitacao_total_mm': 'Chuva Total (mm)'},
                     color='precipitacao_total_mm', color_continuous_scale='Blues')
        fig.update_xaxes(type='category')
        fig.show()

    def perfil_horario_por_ano(self, coluna, agregacao='mean'):
        """
        Gráfico de linhas mostrando o perfil médio (ou somado) hora a hora,
        com uma linha por ano. Ideal para comparar padrões diários entre anos
        — ex.: temperatura média às 14h foi maior em 2023 do que em 2022?

        Parâmetros
        ----------
        coluna     : variável a analisar
        agregacao  : 'mean' para médias (temperatura, umidade…)
                     'sum'  para totais (precipitação, radiação…)
        """
        func = 'mean' if agregacao == 'mean' else 'sum'
        df_hora = (
            self.df.groupby(['ano', 'hora'])[coluna]
            .agg(func)
            .reset_index()
        )
        df_hora['ano'] = df_hora['ano'].astype(str)   # cor discreta na legenda

        titulo = (
            f'Perfil Horário — {"Média" if func == "mean" else "Total"} de {coluna} por Ano'
        )
        fig = px.line(
            df_hora, x='hora', y=coluna, color='ano',
            markers=True, title=titulo,
            labels={'hora': 'Hora do dia (0–23)', coluna: coluna, 'ano': 'Ano'},
        )
        fig.update_xaxes(tickmode='linear', tick0=0, dtick=1, range=[-0.5, 23.5])
        fig.update_traces(line=dict(width=2.5))
        fig.update_layout(hovermode='x unified')
        fig.show()

    def distribuicao_horaria_por_ano(self, coluna):
        """
        Boxplot por hora do dia, com facetas (subplots) separadas por ano.
        Permite ver, em cada ano, como a variabilidade dentro do dia se distribui
        — ex.: a dispersão de temperatura às 15h é maior no verão de 2024?

        A figura fica mais alta conforme o número de anos aumenta.
        """
        anos = sorted(self.df['ano'].unique())
        n_anos = len(anos)

        fig = px.box(
            self.df, x='hora', y=coluna,
            facet_row='ano',
            color='ano',
            title=f'Distribuição por Hora do Dia — {coluna} (por Ano)',
            labels={'hora': 'Hora', coluna: coluna, 'ano': 'Ano'},
            height=300 * n_anos,
        )
        fig.update_xaxes(tickmode='linear', tick0=0, dtick=1)
        fig.update_layout(showlegend=False)
        fig.show()

    def evolucao_diaria_por_ano(self, coluna, agregacao='mean'):
        """
        Gráfico de linhas com a evolução dia a dia ao longo do ano (eixo X =
        dia do ano, 1–366), com uma linha por ano.
        Permite alinhar os anos num mesmo eixo temporal e comparar diretamente
        — ex.: o dia 200 (19/jul) foi mais quente em 2022 ou 2023?

        Parâmetros
        ----------
        coluna     : variável a analisar
        agregacao  : 'mean' | 'sum'
        """
        func = 'mean' if agregacao == 'mean' else 'sum'
        df_dia = (
            self.df.groupby(['ano', 'dia_do_ano'])[coluna]
            .agg(func)
            .reset_index()
        )
        df_dia['ano'] = df_dia['ano'].astype(str)

        titulo = (
            f'Evolução Diária — {"Média" if func == "mean" else "Total"} '
            f'de {coluna} por Ano'
        )
        fig = px.line(
            df_dia, x='dia_do_ano', y=coluna, color='ano',
            title=titulo,
            labels={'dia_do_ano': 'Dia do Ano (1–366)', coluna: coluna, 'ano': 'Ano'},
        )
        fig.update_traces(line=dict(width=1.5), opacity=0.85)
        fig.update_layout(hovermode='x unified')
        fig.show()

    def heatmap_hora_mes_por_ano(self, coluna, agregacao='mean'):
        """
        Heatmap 12×24 (mês × hora) com uma aba (subplot) por ano.
        Combina as duas granularidades (hora e mês) numa única visualização,
        deixando evidente sazonalidade intra-dia e intra-anual simultaneamente.
        Excelente para variáveis como temperatura, umidade e radiação solar.

        Parâmetros
        ----------
        coluna     : variável a analisar
        agregacao  : 'mean' | 'sum'
        """
        from plotly.subplots import make_subplots

        func = 'mean' if agregacao == 'mean' else 'sum'
        label_z = f'{"Média" if func == "mean" else "Total"} de {coluna}'

        df_hm = (
            self.df.groupby(['ano', 'mes', 'hora'])[coluna]
            .agg(func)
            .reset_index()
        )

        anos = sorted(df_hm['ano'].unique())
        n_anos = len(anos)

        # Calcula a escala de cor global para que todos os anos sejam comparáveis
        z_min = df_hm[coluna].min()
        z_max = df_hm[coluna].max()

        fig = make_subplots(
            rows=1, cols=n_anos,
            subplot_titles=[str(a) for a in anos],
            shared_yaxes=True,
        )

        meses_abrev = ['Jan','Fev','Mar','Abr','Mai','Jun',
                       'Jul','Ago','Set','Out','Nov','Dez']

        for i, ano in enumerate(anos, start=1):
            pivot = (
                df_hm[df_hm['ano'] == ano]
                .pivot(index='mes', columns='hora', values=coluna)
                .reindex(index=range(1, 13), columns=range(0, 24))
            )

            heatmap = go.Heatmap(
                z=pivot.values,
                x=[f'{h:02d}h' for h in range(24)],
                y=meses_abrev,
                colorscale='RdBu_r',
                zmin=z_min, zmax=z_max,
                colorbar=dict(title=label_z) if i == n_anos else dict(showticklabels=False),
                showscale=(i == n_anos),      # mostra barra de cor só no último subplot
                hovertemplate='Mês: %{y}<br>Hora: %{x}<br>Valor: %{z:.2f}<extra></extra>',
            )
            fig.add_trace(heatmap, row=1, col=i)

        fig.update_layout(
            title_text=f'Heatmap Hora × Mês — {label_z}',
            height=420,
            width=350 * n_anos,
        )
        fig.show()

    def acumulado_anual_precipitacao(self):
        """
        Gera um gráfico de barras com o volume total de chuva (precipitação) por ano.
        """
        if 'precipitacao_total_mm' not in self.df.columns:
            print("A coluna 'precipitacao_total_mm' não existe no DataFrame.")
            return

        df_chuva = self.df.groupby('ano')['precipitacao_total_mm'].sum().reset_index()

        fig = px.bar(df_chuva, x='ano', y='precipitacao_total_mm', text_auto='.1f',
                     title='Precipitação Total Acumulada por Ano',
                     labels={'ano': 'Ano', 'precipitacao_total_mm': 'Chuva Total (mm)'},
                     color='precipitacao_total_mm', color_continuous_scale='Blues')
        
        fig.update_xaxes(type='category')
        fig.show()

    def contar_dias_extremos(self, temperatura_limite, tipo='max'):
        """
        Conta quantos dias por ano a temperatura ultrapassou um limite (calor) 
        ou ficou abaixo de um limite (frio).
        
        tipo: 'max' (para dias mais quentes que o limite) ou 'min' (para dias mais frios)
        """
        # CORREÇÃO AQUI: Note os colchetes duplos [['temp_max_c', 'temp_min_c']]
        df_diario = self.df.groupby(['ano', 'data_sem_hora'])[['temp_max_c', 'temp_min_c']].agg(
            temp_max_c=('temp_max_c', 'max'),
            temp_min_c=('temp_min_c', 'min')
        ).reset_index()

        if tipo == 'max':
            dias_extremos = df_diario[df_diario['temp_max_c'] >= temperatura_limite]
            titulo = f'Número de Dias no Ano com Temperatura ≥ {temperatura_limite}°C'
        else:
            dias_extremos = df_diario[df_diario['temp_min_c'] <= temperatura_limite]
            titulo = f'Número de Dias no Ano com Temperatura ≤ {temperatura_limite}°C'

        contagem = dias_extremos.groupby('ano').size().reset_index(name='qtd_dias')
        
        # Garante que anos com 0 dias extremos apareçam no gráfico
        todos_os_anos = pd.DataFrame({'ano': self.df['ano'].unique()})
        contagem = todos_os_anos.merge(contagem, on='ano', how='left').fillna(0)

        fig = px.bar(contagem, x='ano', y='qtd_dias', text_auto=True,
                     title=titulo, labels={'ano': 'Ano', 'qtd_dias': 'Quantidade de Dias'})
        
        fig.update_xaxes(type='category')
        fig.update_traces(marker_color='red' if tipo == 'max' else 'blue')
        fig.show()

    