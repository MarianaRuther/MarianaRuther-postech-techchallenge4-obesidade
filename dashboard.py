"""
Dashboard Analítico - Insights sobre Obesidade
Painel com principais descobertas para equipe médica
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    page_title="Dashboard Analítico - Obesidade",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cores personalizadas vibrantes
COLORS = {
    'Obesity_Type_III': '#d32f2f',  # Vermelho escuro
    'Obesity_Type_II': '#f57c00',   # Laranja
    'Obesity_Type_I': '#fbc02d',    # Amarelo
    'Overweight_Level_II': '#ffa726', # Laranja claro
    'Overweight_Level_I': '#ffcc02',  # Amarelo claro
    'Normal_Weight': '#66bb6a',      # Verde
    'Insufficient_Weight': '#42a5f5' # Azul
}

@st.cache_data
def load_data():
    """Carrega e processa os dados"""
    # entrega macro: carrego dados uma vez e reutilizo - cache_data otimiza performance
    df = pd.read_csv('Obesity.csv')
    
    # aplico feature engineering básico para análises do dashboard
    df['BMI'] = df['Weight'] / (df['Height'] ** 2)
    
    def categorize_bmi(bmi):
        if bmi < 18.5:
            return 'Peso Insuficiente'
        elif bmi < 25:
            return 'Peso Normal'
        elif bmi < 30:
            return 'Sobrepeso'
        else:
            return 'Obesidade'
    
    df['BMI_Category'] = df['BMI'].apply(categorize_bmi)
    
    # Mapear níveis de obesidade para português
    obesity_map = {
        'Insufficient_Weight': 'Peso Insuficiente',
        'Normal_Weight': 'Peso Normal',
        'Overweight_Level_I': 'Sobrepeso Nível I',
        'Overweight_Level_II': 'Sobrepeso Nível II',
        'Obesity_Type_I': 'Obesidade Tipo I',
        'Obesity_Type_II': 'Obesidade Tipo II',
        'Obesity_Type_III': 'Obesidade Tipo III'
    }
    
    df['Obesity_PT'] = df['Obesity'].map(obesity_map)
    
    # Mapear gênero para português
    df['Gender_PT'] = df['Gender'].map({'Male': 'Masculino', 'Female': 'Feminino'})
    
    return df

def generate_ai_insights(df):
    """Gera insights usando análise estatística avançada (simulando IA)"""
    # entrega macro: insights automáticos usando análise estatística - simula ia para equipe médica
    insights = []
    
    # analiso correlações entre fatores de risco e obesidade
    # começo com histórico familiar que é um fator importante
    family_yes = df[df['family_history'] == 'yes']
    family_no = df[df['family_history'] == 'no']
    
    if len(family_yes) > 0 and len(family_no) > 0:
        family_yes_obesity = (family_yes['Obesity'].str.contains('Obesity', na=False).sum() / len(family_yes)) * 100
        family_no_obesity = (family_no['Obesity'].str.contains('Obesity', na=False).sum() / len(family_no)) * 100
        risk_increase = family_yes_obesity - family_no_obesity
        
        insights.append({
            'title': '🔴 Fator de Risco Crítico: Histórico Familiar',
            'description': f'Pacientes com histórico familiar têm {risk_increase:.1f} pontos percentuais a mais de risco de obesidade.',
            'data': {
                'Com histórico': family_yes_obesity,
                'Sem histórico': family_no_obesity
            },
            'severity': 'Alto' if risk_increase > 20 else 'Moderado'
        })
    
    # analiso impacto da atividade física - fator protetor importante
    low_activity = df[df['FAF'] <= 1]
    high_activity = df[df['FAF'] >= 2]
    
    if len(low_activity) > 0 and len(high_activity) > 0:
        # calculo taxa de obesidade para cada grupo
        low_obesity = (low_activity['Obesity'].str.contains('Obesity', na=False).sum() / len(low_activity)) * 100
        high_obesity = (high_activity['Obesity'].str.contains('Obesity', na=False).sum() / len(high_activity)) * 100
        protection = high_obesity - low_obesity
        
        insights.append({
            'title': '💪 Fator Protetor: Atividade Física',
            'description': f'Atividade física regular reduz o risco de obesidade em {abs(protection):.1f} pontos percentuais.',
            'data': {
                'Baixa atividade': low_obesity,
                'Alta atividade': high_obesity
            },
            'severity': 'Proteção Forte' if abs(protection) > 15 else 'Proteção Moderada'
        })
    
    # Alimentos calóricos
    favc_yes = df[df['FAVC'] == 'yes']
    favc_no = df[df['FAVC'] == 'no']
    
    if len(favc_yes) > 0 and len(favc_no) > 0:
        favc_yes_obesity = (favc_yes['Obesity'].str.contains('Obesity', na=False).sum() / len(favc_yes)) * 100
        favc_no_obesity = (favc_no['Obesity'].str.contains('Obesity', na=False).sum() / len(favc_no)) * 100
        risk = favc_yes_obesity - favc_no_obesity
        
        insights.append({
            'title': '🍔 Fator de Risco: Alimentos Calóricos',
            'description': f'Consumo frequente de alimentos calóricos aumenta o risco em {risk:.1f} pontos percentuais.',
            'data': {
                'Consome frequentemente': favc_yes_obesity,
                'Não consome': favc_no_obesity
            },
            'severity': 'Alto' if risk > 15 else 'Moderado'
        })
    
    # Consumo de vegetais
    high_veg = df[df['FCVC'] >= 2.5]
    low_veg = df[df['FCVC'] < 2.5]
    
    if len(high_veg) > 0 and len(low_veg) > 0:
        high_veg_obesity = (high_veg['Obesity'].str.contains('Obesity', na=False).sum() / len(high_veg)) * 100
        low_veg_obesity = (low_veg['Obesity'].str.contains('Obesity', na=False).sum() / len(low_veg)) * 100
        protection = high_veg_obesity - low_veg_obesity
        
        insights.append({
            'title': '🥗 Fator Protetor: Consumo de Vegetais',
            'description': f'Maior consumo de vegetais reduz o risco em {abs(protection):.1f} pontos percentuais.',
            'data': {
                'Alto consumo': high_veg_obesity,
                'Baixo consumo': low_veg_obesity
            },
            'severity': 'Proteção Moderada'
        })
    
    return insights

def main():
    """Função principal do dashboard"""
    
    # Header com estilo
    st.markdown("""
    <style>
    .dashboard-header {
        background: linear-gradient(90deg, #1f77b4 0%, #42a5f5 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .dashboard-header h1 {
        color: white;
        margin: 0;
    }
    .insight-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="dashboard-header"><h1>📊 Dashboard Analítico - Estudo sobre Obesidade</h1><p style="font-size:1.2rem; margin:0;">Principais insights para a equipe médica</p></div>', unsafe_allow_html=True)
    
    # Carregar dados
    df = load_data()
    
    # métricas principais do dashboard - entrega macro: visão geral para equipe médica
    st.header("📈 Métricas Gerais da População")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total de Pacientes", 
            f"{len(df):,}",
            delta=None,
            delta_color="normal"
        )
    
    with col2:
        avg_age = df['Age'].mean()
        st.metric("Idade Média", f"{avg_age:.1f} anos")
    
    with col3:
        avg_bmi = df['BMI'].mean()
        st.metric("IMC Médio", f"{avg_bmi:.1f}")
    
    with col4:
        # calculo taxa geral de obesidade na população
        obesity_rate = (df['Obesity'].str.contains('Obesity', na=False).sum() / len(df)) * 100
        st.metric("Taxa de Obesidade", f"{obesity_rate:.1f}%")
    
    st.markdown("---")
    
    # visualizações coloridas da distribuição - entrega macro: gráficos interativos e coloridos
    st.header("🎯 Distribuição dos Níveis de Obesidade")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # gráfico de pizza com cores por severidade
        obesity_counts = df['Obesity_PT'].value_counts()
        
        # mapeio cores vibrantes para cada nível - vermelho para mais grave, verde para normal
        color_map = {
            'Obesidade Tipo III': '#d32f2f',
            'Obesidade Tipo II': '#f57c00',
            'Obesidade Tipo I': '#fbc02d',
            'Sobrepeso Nível II': '#ffa726',
            'Sobrepeso Nível I': '#ffcc02',
            'Peso Normal': '#66bb6a',
            'Peso Insuficiente': '#42a5f5'
        }
        
        colors_list = [color_map.get(nivel, '#999999') for nivel in obesity_counts.index]
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=obesity_counts.index,
            values=obesity_counts.values,
            hole=0.4,
            marker=dict(colors=colors_list, line=dict(color='#FFFFFF', width=2)),
            textinfo='label+percent',
            textposition='outside'
        )])
        
        fig_pie.update_layout(
            title="Distribuição por Nível de Obesidade",
            height=400,
            showlegend=True,
            legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.1)
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Gráfico de barras horizontal colorido
        fig_bar = go.Figure()
        
        fig_bar.add_trace(go.Bar(
            y=obesity_counts.index,
            x=obesity_counts.values,
            orientation='h',
            marker=dict(
                color=colors_list,
                line=dict(color='rgba(0,0,0,0.3)', width=1)
            ),
            text=[f"{v} pacientes" for v in obesity_counts.values],
            textposition='outside'
        ))
        
        fig_bar.update_layout(
            title="Contagem por Nível de Obesidade",
            xaxis_title="Quantidade de Pacientes",
            yaxis_title="",
            height=400,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig_bar, use_container_width=True)
    
    st.markdown("---")
    
    # análise detalhada por gênero - entrega importante para identificar diferenças
    st.header("👥 Análise Detalhada por Gênero")
    
    # calculo estatísticas agregadas por gênero
    gender_stats = df.groupby('Gender_PT').agg({
        'BMI': 'mean',
        'Age': 'mean',
        'Weight': 'mean',
        'Height': 'mean'
    }).round(2)
    
    # calculo taxas de obesidade separadas por gênero para comparação
    male_obesity_rate = (df[df['Gender_PT'] == 'Masculino']['Obesity'].str.contains('Obesity', na=False).sum() / 
                        len(df[df['Gender_PT'] == 'Masculino'])) * 100
    female_obesity_rate = (df[df['Gender_PT'] == 'Feminino']['Obesity'].str.contains('Obesity', na=False).sum() / 
                          len(df[df['Gender_PT'] == 'Feminino'])) * 100
    
    # Cards de métricas
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("👨 Masculino - IMC", f"{gender_stats.loc['Masculino', 'BMI']:.1f}")
    with col2:
        st.metric("👨 Masculino - Idade", f"{gender_stats.loc['Masculino', 'Age']:.0f} anos")
    with col3:
        st.metric("👨 Masculino - Obesidade", f"{male_obesity_rate:.1f}%")
    with col4:
        st.metric("👩 Feminino - IMC", f"{gender_stats.loc['Feminino', 'BMI']:.1f}")
    with col5:
        st.metric("👩 Feminino - Obesidade", f"{female_obesity_rate:.1f}%")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Gráficos principais
    col1, col2 = st.columns(2)
    
    with col1:
        # Gráfico de barras agrupadas (não empilhadas) para melhor comparação
        gender_obesity = pd.crosstab(df['Gender_PT'], df['Obesity_PT'], normalize='index') * 100
        
        fig_gender = go.Figure()
        
        for nivel in gender_obesity.columns:
            fig_gender.add_trace(go.Bar(
                name=nivel,
                x=gender_obesity.index,
                y=gender_obesity[nivel],
                marker_color=color_map.get(nivel, '#999999'),
                text=[f"{v:.1f}%" for v in gender_obesity[nivel]],
                textposition='outside'
            ))
        
        fig_gender.update_layout(
            title=dict(
                text="📊 Distribuição de Obesidade por Gênero (%)",
                x=0.5,
                xanchor='center',
                font=dict(size=16)
            ),
            xaxis_title="Gênero",
            yaxis_title="Percentual (%)",
            barmode='group',  # Mudado de 'stack' para 'group' para melhor comparação
            height=500,
            margin=dict(t=80, b=50, l=50, r=50),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.25,
                xanchor="center",
                x=0.5
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig_gender, use_container_width=True)
    
    with col2:
        # Gráfico de comparação de métricas físicas
        metrics_comparison = pd.DataFrame({
            'Masculino': [
                gender_stats.loc['Masculino', 'BMI'],
                gender_stats.loc['Masculino', 'Weight'],
                gender_stats.loc['Masculino', 'Height'] * 100,  # Converter para cm
                gender_stats.loc['Masculino', 'Age']
            ],
            'Feminino': [
                gender_stats.loc['Feminino', 'BMI'],
                gender_stats.loc['Feminino', 'Weight'],
                gender_stats.loc['Feminino', 'Height'] * 100,  # Converter para cm
                gender_stats.loc['Feminino', 'Age']
            ]
        }, index=['IMC', 'Peso (kg)', 'Altura (cm)', 'Idade (anos)'])
        
        fig_metrics = go.Figure()
        
        fig_metrics.add_trace(go.Bar(
            name='Masculino',
            x=metrics_comparison.index,
            y=metrics_comparison['Masculino'],
            marker_color='#3498db',
            text=[f"{v:.1f}" for v in metrics_comparison['Masculino']],
            textposition='outside'
        ))
        
        fig_metrics.add_trace(go.Bar(
            name='Feminino',
            x=metrics_comparison.index,
            y=metrics_comparison['Feminino'],
            marker_color='#e91e63',
            text=[f"{v:.1f}" for v in metrics_comparison['Feminino']],
            textposition='outside'
        ))
        
        # Calcular valor máximo para ajustar o range
        max_metric_value = max(
            metrics_comparison['Masculino'].max(),
            metrics_comparison['Feminino'].max()
        )
        
        fig_metrics.update_layout(
            title=dict(
                text="📏 Comparação de Métricas Físicas",
                x=0.5,
                xanchor='center',
                font=dict(size=16)
            ),
            xaxis_title="Métrica",
            yaxis_title="Valor",
            barmode='group',
            height=500,
            margin=dict(t=80, b=80, l=50, r=50),  # Margem inferior maior para textos
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(range=[0, max_metric_value * 1.25])  # Espaço extra para textos externos
        )
        
        st.plotly_chart(fig_metrics, use_container_width=True)
    
    # Gráfico de pizza comparativo
    st.subheader("🍰 Distribuição Detalhada por Nível de Obesidade")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Pizza para Masculino
        male_obesity = df[df['Gender_PT'] == 'Masculino']['Obesity_PT'].value_counts()
        colors_male = [color_map.get(nivel, '#999999') for nivel in male_obesity.index]
        
        fig_male = go.Figure(data=[go.Pie(
            labels=male_obesity.index,
            values=male_obesity.values,
            hole=0.4,
            marker=dict(colors=colors_male, line=dict(color='#FFFFFF', width=2)),
            textinfo='label+percent',
            textposition='outside',
            title='Masculino'
        )])
        
        fig_male.update_layout(
            title="👨 Distribuição - Masculino",
            height=400,
            showlegend=True,
            legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.1)
        )
        
        st.plotly_chart(fig_male, use_container_width=True)
    
    with col2:
        # Pizza para Feminino
        female_obesity = df[df['Gender_PT'] == 'Feminino']['Obesity_PT'].value_counts()
        colors_female = [color_map.get(nivel, '#999999') for nivel in female_obesity.index]
        
        fig_female = go.Figure(data=[go.Pie(
            labels=female_obesity.index,
            values=female_obesity.values,
            hole=0.4,
            marker=dict(colors=colors_female, line=dict(color='#FFFFFF', width=2)),
            textinfo='label+percent',
            textposition='outside',
            title='Feminino'
        )])
        
        fig_female.update_layout(
            title="👩 Distribuição - Feminino",
            height=400,
            showlegend=True,
            legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.1)
        )
        
        st.plotly_chart(fig_female, use_container_width=True)
    
    # Estatísticas detalhadas com explicações
    st.subheader("📋 Análise Estatística Detalhada por Gênero")
    
    st.info("""
    **💡 O que significam essas estatísticas?**
    - **Média:** Valor médio do grupo (ex: IMC médio de 28.5 significa que a maioria dos pacientes tem IMC próximo a esse valor)
    - **Desvio Padrão:** Mede a variação dos dados (quanto menor, mais homogêneo o grupo)
    - **Mínimo/Máximo:** Valores extremos observados no grupo
    """)
    
    # Calcular estatísticas
    gender_stats_detailed = df.groupby('Gender_PT').agg({
        'BMI': ['mean', 'std', 'min', 'max'],
        'Age': ['mean', 'std', 'min', 'max'],
        'Weight': ['mean', 'std', 'min', 'max'],
        'Height': ['mean', 'std', 'min', 'max']
    }).round(2)
    
    # Criar visualizações por métrica
    metrics_to_show = ['BMI', 'Age', 'Weight', 'Height']
    metric_names = {
        'BMI': 'IMC (Índice de Massa Corporal)',
        'Age': 'Idade',
        'Weight': 'Peso (kg)',
        'Height': 'Altura (m)'
    }
    
    for metric in metrics_to_show:
        st.markdown(f"### 📊 {metric_names[metric]}")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            male_mean = gender_stats_detailed.loc['Masculino', (metric, 'mean')]
            female_mean = gender_stats_detailed.loc['Feminino', (metric, 'mean')]
            diff = abs(male_mean - female_mean)
            diff_pct = (diff / female_mean * 100) if female_mean > 0 else 0
            
            st.metric(
                "👨 Masculino - Média",
                f"{male_mean:.2f}",
                delta=f"{diff:.2f} de diferença" if metric != 'BMI' else None
            )
        
        with col2:
            st.metric(
                "👩 Feminino - Média",
                f"{female_mean:.2f}"
            )
        
        with col3:
            male_std = gender_stats_detailed.loc['Masculino', (metric, 'std')]
            st.metric(
                "👨 Masculino - Variação",
                f"±{male_std:.2f}",
                help="Desvio padrão: quanto menor, mais homogêneo o grupo"
            )
        
        with col4:
            female_std = gender_stats_detailed.loc['Feminino', (metric, 'std')]
            st.metric(
                "👩 Feminino - Variação",
                f"±{female_std:.2f}",
                help="Desvio padrão: quanto menor, mais homogêneo o grupo"
            )
        
        # Gráfico de comparação
        fig_metric = go.Figure()
        
        fig_metric.add_trace(go.Bar(
            name='Masculino',
            x=['Média', 'Mínimo', 'Máximo'],
            y=[
                gender_stats_detailed.loc['Masculino', (metric, 'mean')],
                gender_stats_detailed.loc['Masculino', (metric, 'min')],
                gender_stats_detailed.loc['Masculino', (metric, 'max')]
            ],
            marker_color='#3498db',
            text=[
                f"{gender_stats_detailed.loc['Masculino', (metric, 'mean')]:.2f}",
                f"{gender_stats_detailed.loc['Masculino', (metric, 'min')]:.2f}",
                f"{gender_stats_detailed.loc['Masculino', (metric, 'max')]:.2f}"
            ],
            textposition='outside'
        ))
        
        fig_metric.add_trace(go.Bar(
            name='Feminino',
            x=['Média', 'Mínimo', 'Máximo'],
            y=[
                gender_stats_detailed.loc['Feminino', (metric, 'mean')],
                gender_stats_detailed.loc['Feminino', (metric, 'min')],
                gender_stats_detailed.loc['Feminino', (metric, 'max')]
            ],
            marker_color='#e91e63',
            text=[
                f"{gender_stats_detailed.loc['Feminino', (metric, 'mean')]:.2f}",
                f"{gender_stats_detailed.loc['Feminino', (metric, 'min')]:.2f}",
                f"{gender_stats_detailed.loc['Feminino', (metric, 'max')]:.2f}"
            ],
            textposition='outside'
        ))
        
        # Calcular valor máximo para ajustar o range do eixo Y
        max_value = max(
            gender_stats_detailed.loc['Masculino', (metric, 'max')],
            gender_stats_detailed.loc['Feminino', (metric, 'max')]
        )
        
        fig_metric.update_layout(
            title=dict(
                text=f"Comparação: {metric_names[metric]}",
                x=0.5,
                xanchor='center',
                font=dict(size=16)
            ),
            barmode='group',
            height=400,
            margin=dict(t=80, b=80, l=50, r=50),  # Margens maiores para textos externos
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(
                range=[0, max_value * 1.25],  # 25% de espaço extra para textos externos
                title=metric_names[metric]
            )
        )
        
        st.plotly_chart(fig_metric, use_container_width=True)
        
        # Insight automático
        if metric == 'BMI':
            if male_mean > female_mean:
                st.success(f"💡 **Insight:** Homens apresentam IMC médio {diff:.2f} pontos maior que mulheres ({male_mean:.2f} vs {female_mean:.2f}).")
            else:
                st.info(f"💡 **Insight:** Mulheres apresentam IMC médio {diff:.2f} pontos maior que homens ({female_mean:.2f} vs {male_mean:.2f}).")
        elif metric == 'Weight':
            if male_mean > female_mean:
                st.success(f"💡 **Insight:** Homens pesam em média {diff:.1f}kg a mais que mulheres ({male_mean:.1f}kg vs {female_mean:.1f}kg).")
            else:
                st.info(f"💡 **Insight:** Mulheres pesam em média {diff:.1f}kg a mais que homens ({female_mean:.1f}kg vs {male_mean:.1f}kg).")
        elif metric == 'Height':
            if male_mean > female_mean:
                st.success(f"💡 **Insight:** Homens são em média {diff*100:.1f}cm mais altos que mulheres ({male_mean*100:.1f}cm vs {female_mean*100:.1f}cm).")
            else:
                st.info(f"💡 **Insight:** Mulheres são em média {abs(diff)*100:.1f}cm mais altas que homens ({female_mean*100:.1f}cm vs {male_mean*100:.1f}cm).")
        
        st.markdown("<br>", unsafe_allow_html=True)
    
    # Resumo final
    count_by_gender = df['Gender_PT'].value_counts()
    st.markdown("---")
    st.markdown("### 📌 Resumo da Amostra")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #3498db 0%, #2980b9 100%); padding: 1.5rem; border-radius: 10px; color: white;">
            <h3 style="color: white; margin: 0 0 0.5rem 0;">👨 Masculino</h3>
            <p style="color: white; font-size: 1.2rem; margin: 0;"><strong>{count_by_gender.get('Masculino', 0)} pacientes</strong></p>
            <p style="color: white; margin: 0.5rem 0 0 0;">Representa {count_by_gender.get('Masculino', 0)/len(df)*100:.1f}% da amostra total</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #e91e63 0%, #c2185b 100%); padding: 1.5rem; border-radius: 10px; color: white;">
            <h3 style="color: white; margin: 0 0 0.5rem 0;">👩 Feminino</h3>
            <p style="color: white; font-size: 1.2rem; margin: 0;"><strong>{count_by_gender.get('Feminino', 0)} pacientes</strong></p>
            <p style="color: white; margin: 0.5rem 0 0 0;">Representa {count_by_gender.get('Feminino', 0)/len(df)*100:.1f}% da amostra total</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Análise de Fatores de Risco
    st.header("⚠️ Análise de Fatores de Risco")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Histórico familiar
        family_obesity = pd.crosstab(df['family_history'], df['Obesity_PT'], normalize='index') * 100
        family_obesity.index = family_obesity.index.map({'yes': 'Sim', 'no': 'Não'})
        
        fig_family = go.Figure()
        
        for nivel in family_obesity.columns:
            fig_family.add_trace(go.Bar(
                name=nivel,
                x=family_obesity.index,
                y=family_obesity[nivel],
                marker_color=color_map.get(nivel, '#999999')
            ))
        
        fig_family.update_layout(
            title="Impacto do Histórico Familiar (%)",
            xaxis_title="Histórico Familiar",
            yaxis_title="Percentual (%)",
            barmode='stack',
            height=400,
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.05)
        )
        
        st.plotly_chart(fig_family, use_container_width=True)
    
    with col2:
        # Alimentos calóricos
        favc_obesity = pd.crosstab(df['FAVC'], df['Obesity_PT'], normalize='index') * 100
        favc_obesity.index = favc_obesity.index.map({'yes': 'Sim', 'no': 'Não'})
        
        fig_favc = go.Figure()
        
        for nivel in favc_obesity.columns:
            fig_favc.add_trace(go.Bar(
                name=nivel,
                x=favc_obesity.index,
                y=favc_obesity[nivel],
                marker_color=color_map.get(nivel, '#999999')
            ))
        
        fig_favc.update_layout(
            title="Impacto de Alimentos Calóricos (%)",
            xaxis_title="Consome Alimentos Calóricos?",
            yaxis_title="Percentual (%)",
            barmode='stack',
            height=400,
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.05)
        )
        
        st.plotly_chart(fig_favc, use_container_width=True)
    
    # Atividade física e vegetais
    st.subheader("💪 Impacto da Atividade Física e Alimentação")
    
    st.info("💡 **Como ler estes gráficos:** Cada barra mostra a taxa de obesidade para diferentes níveis de atividade física ou consumo de vegetais. Quanto maior a barra vermelha/laranja, maior o risco de obesidade.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Criar categorias claras de atividade física
        df['FAF_Category'] = pd.cut(
            df['FAF'].round(), 
            bins=[-0.5, 0.5, 1.5, 2.5, 3.5], 
            labels=['Nenhuma', '1-2x/semana', '2-3x/semana', '4-5x/semana']
        )
        
        # Calcular taxa de obesidade por categoria
        faf_analysis = df.groupby('FAF_Category').agg({
            'Obesity': lambda x: (x.str.contains('Obesity', na=False).sum() / len(x) * 100)
        }).round(1)
        faf_analysis.columns = ['Taxa de Obesidade (%)']
        
        # Gráfico simples e claro
        fig_faf = go.Figure()
        
        fig_faf.add_trace(go.Bar(
            x=faf_analysis.index,
            y=faf_analysis['Taxa de Obesidade (%)'],
            marker=dict(
                color=faf_analysis['Taxa de Obesidade (%)'],
                colorscale='RdYlGn_r',
                showscale=True,
                colorbar=dict(title="Taxa de<br>Obesidade (%)")
            ),
            text=[f"{v:.1f}%" for v in faf_analysis['Taxa de Obesidade (%)']],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Taxa de Obesidade: %{y:.1f}%<extra></extra>'
        ))
        
        max_faf_value = max(faf_analysis['Taxa de Obesidade (%)'])
        
        fig_faf.update_layout(
            title=dict(
                text="📊 Taxa de Obesidade por Nível de Atividade Física",
                x=0.5,
                xanchor='center',
                font=dict(size=16)
            ),
            xaxis_title="Frequência de Atividade Física",
            yaxis_title="Taxa de Obesidade (%)",
            height=450,
            margin=dict(t=80, b=80, l=50, r=50),  # Margem inferior maior para textos
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(range=[0, max_faf_value * 1.3])  # Mais espaço para textos externos
        )
        
        st.plotly_chart(fig_faf, use_container_width=True)
        
        # Insight
        max_faf = faf_analysis['Taxa de Obesidade (%)'].idxmax()
        min_faf = faf_analysis['Taxa de Obesidade (%)'].idxmin()
        diff_faf = faf_analysis.loc[max_faf, 'Taxa de Obesidade (%)'] - faf_analysis.loc[min_faf, 'Taxa de Obesidade (%)']
        
        st.success(f"💡 **Insight:** Pessoas com '{max_faf}' atividade física têm {diff_faf:.1f} pontos percentuais a mais de risco de obesidade comparado a '{min_faf}'.")
    
    with col2:
        # Criar categorias claras de consumo de vegetais
        df['FCVC_Category'] = pd.cut(
            df['FCVC'].round(), 
            bins=[0.5, 1.5, 2.5, 3.5], 
            labels=['Baixo (1)', 'Moderado (2)', 'Alto (3)']
        )
        
        # Calcular taxa de obesidade por categoria
        fcvc_analysis = df.groupby('FCVC_Category').agg({
            'Obesity': lambda x: (x.str.contains('Obesity', na=False).sum() / len(x) * 100)
        }).round(1)
        fcvc_analysis.columns = ['Taxa de Obesidade (%)']
        
        # Gráfico simples e claro
        fig_fcvc = go.Figure()
        
        fig_fcvc.add_trace(go.Bar(
            x=fcvc_analysis.index,
            y=fcvc_analysis['Taxa de Obesidade (%)'],
            marker=dict(
                color=fcvc_analysis['Taxa de Obesidade (%)'],
                colorscale='RdYlGn_r',
                showscale=True,
                colorbar=dict(title="Taxa de<br>Obesidade (%)")
            ),
            text=[f"{v:.1f}%" for v in fcvc_analysis['Taxa de Obesidade (%)']],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Taxa de Obesidade: %{y:.1f}%<extra></extra>'
        ))
        
        max_fcvc_value = max(fcvc_analysis['Taxa de Obesidade (%)'])
        
        fig_fcvc.update_layout(
            title=dict(
                text="🥗 Taxa de Obesidade por Consumo de Vegetais",
                x=0.5,
                xanchor='center',
                font=dict(size=16)
            ),
            xaxis_title="Nível de Consumo de Vegetais",
            yaxis_title="Taxa de Obesidade (%)",
            height=450,
            margin=dict(t=80, b=80, l=50, r=50),  # Margem inferior maior para textos
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(range=[0, max_fcvc_value * 1.3])  # Mais espaço para textos externos
        )
        
        st.plotly_chart(fig_fcvc, use_container_width=True)
        
        # Insight
        max_fcvc = fcvc_analysis['Taxa de Obesidade (%)'].idxmax()
        min_fcvc = fcvc_analysis['Taxa de Obesidade (%)'].idxmin()
        diff_fcvc = fcvc_analysis.loc[max_fcvc, 'Taxa de Obesidade (%)'] - fcvc_analysis.loc[min_fcvc, 'Taxa de Obesidade (%)']
        
        st.success(f"💡 **Insight:** Consumo '{max_fcvc}' de vegetais está associado a {diff_fcvc:.1f} pontos percentuais a mais de risco de obesidade comparado a '{min_fcvc}'.")
    
    st.markdown("---")
    
    # Análise de IMC
    st.header("📏 Análise de IMC")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Box plot colorido
        fig_box = go.Figure()
        
        for nivel in df['Obesity_PT'].unique():
            data = df[df['Obesity_PT'] == nivel]['BMI']
            fig_box.add_trace(go.Box(
                y=data,
                name=nivel,
                marker_color=color_map.get(nivel, '#999999'),
                boxmean='sd'
            ))
        
        fig_box.update_layout(
            title="Distribuição de IMC por Nível de Obesidade",
            yaxis_title="IMC",
            xaxis_title="Nível de Obesidade",
            height=500,
            showlegend=False
        )
        
        st.plotly_chart(fig_box, use_container_width=True)
    
    with col2:
        # Scatter plot interativo
        fig_scatter = px.scatter(
            df,
            x='Age',
            y='BMI',
            color='Obesity_PT',
            size='Weight',
            hover_data=['Gender_PT', 'Weight', 'Height'],
            title="Relação entre Idade, IMC e Peso",
            labels={'Age': 'Idade (anos)', 'BMI': 'IMC', 'Weight': 'Peso (kg)', 'Gender_PT': 'Gênero'},
            color_discrete_map=color_map,
            size_max=20
        )
        
        fig_scatter.update_layout(height=500)
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    st.markdown("---")
    
    # seção de insights automáticos - entrega macro: análise tipo ia para equipe médica
    st.header("🤖 Insights Gerados por Análise Estatística Avançada")
    
    ai_insights = generate_ai_insights(df)
    
    for insight in ai_insights:
        with st.expander(insight['title'], expanded=True):
            st.markdown(f"**{insight['description']}**")
            
            # gráfico de comparação para cada insight
            fig_insight = go.Figure()
            
            categories = list(insight['data'].keys())
            values = list(insight['data'].values())
            
            fig_insight.add_trace(go.Bar(
                x=categories,
                y=values,
                marker_color=['#e74c3c', '#3498db'],
                text=[f"{v:.1f}%" for v in values],
                textposition='outside'
            ))
            
            max_insight_value = max(values) if values else 100
            
            fig_insight.update_layout(
                title=dict(
                    text=f"Comparação: {insight['title']}",
                    x=0.5,
                    xanchor='center',
                    font=dict(size=14)
                ),
                yaxis_title="Taxa de Obesidade (%)",
                height=350,
                margin=dict(t=60, b=80, l=50, r=50),  # Margens para textos externos
                yaxis=dict(range=[0, max_insight_value * 1.3]),  # Espaço extra para textos
                showlegend=False
            )
            
            st.plotly_chart(fig_insight, use_container_width=True)
            
            st.markdown(f"**Severidade:** {insight['severity']}")
    
    st.markdown("---")
    
    # recomendações estratégicas - entrega macro: insights acionáveis para equipe médica
    st.header("🎯 Recomendações Estratégicas para Equipe Médica")
    
    recommendations = [
        {
            'icon': '🔍',
            'title': 'Triagem Preventiva',
            'content': 'Implementar avaliação de risco de obesidade em consultas de rotina, especialmente para pacientes com histórico familiar. Criar protocolo de triagem rápida usando este sistema preditivo.'
        },
        {
            'icon': '📚',
            'title': 'Programas de Prevenção',
            'content': 'Desenvolver programas focados em educação nutricional e promoção de atividade física. Priorizar pacientes identificados com múltiplos fatores de risco.'
        },
        {
            'icon': '📊',
            'title': 'Monitoramento Contínuo',
            'content': 'Acompanhar pacientes com múltiplos fatores de risco (histórico familiar + baixa atividade física + consumo de alimentos calóricos). Estabelecer check-ups trimestrais.'
        },
        {
            'icon': '⚡',
            'title': 'Intervenção Precoce',
            'content': 'Identificar e intervir em pacientes com sobrepeso antes que progridam para obesidade. A intervenção precoce tem maior taxa de sucesso e menor custo.'
        },
        {
            'icon': '👥',
            'title': 'Abordagem Multidisciplinar',
            'content': 'Envolver nutricionistas, educadores físicos e psicólogos no tratamento de pacientes com obesidade. Criar equipe multidisciplinar dedicada.'
        }
    ]
    
    for rec in recommendations:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 10px; margin: 1rem 0; color: white;">
            <h3 style="color: white; margin: 0 0 0.5rem 0;">{rec['icon']} {rec['title']}</h3>
            <p style="color: white; margin: 0;">{rec['content']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: #f0f2f6; border-radius: 10px;">
        <h3>📌 Desenvolvido para auxiliar a equipe médica na prevenção e tratamento da obesidade</h3>
        <p><strong>Desenvolvido por Mariana Ruther de Araújo (10DTAT) - Tech Challenge 4 FIAP</strong></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
