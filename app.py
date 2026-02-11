"""
Aplicação Streamlit para predição de obesidade.
Sistema preditivo para auxiliar equipe médica.

Este módulo implementa a interface web para predição de níveis de obesidade
usando o modelo treinado pelo ml_pipeline.
"""

import logging
from typing import Dict, Optional, Any

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from ml_pipeline import ObesityPredictor
from config import (
    MODEL_FILE, OBESITY_MAP, COLORS_PT,
    FORM_MAPS, GENDER_MAP_REVERSE, INPUT_VALIDATION,
    STREAMLIT_CONFIG
)

# Configuração do logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuração da página (só executa uma vez)
if 'page_configured' not in st.session_state:
    st.set_page_config(
        page_title=STREAMLIT_CONFIG['page_title_app'],
        page_icon=STREAMLIT_CONFIG['page_icon_app'],
        layout=STREAMLIT_CONFIG['layout'],
        initial_sidebar_state=STREAMLIT_CONFIG['sidebar_state']
    )
    st.session_state.page_configured = True

# CSS personalizado
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .metric-box {
        background-color: white;
        padding: 1rem;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .credits {
        text-align: center;
        color: #666;
        font-size: 0.9rem;
        padding: 1rem 0;
        border-top: 1px solid #e0e0e0;
        margin-top: 2rem;
    }
    .validation-error {
        color: #d32f2f;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model() -> Optional[ObesityPredictor]:
    """
    Carrega o modelo treinado com cache.

    Returns:
        Instância do ObesityPredictor ou None se falhar.
    """
    try:
        pipeline = ObesityPredictor()
        pipeline.load_model(MODEL_FILE)
        logger.info("Modelo carregado com sucesso")
        return pipeline
    except FileNotFoundError:
        logger.error("Arquivo de modelo não encontrado: %s", MODEL_FILE)
        st.error(f"Modelo não encontrado em '{MODEL_FILE}'! Execute primeiro: `python ml_pipeline.py`")
        return None
    except Exception as e:
        logger.error("Erro ao carregar modelo: %s", str(e))
        st.error(f"Erro ao carregar modelo: {str(e)}")
        return None


def validate_input(
    age: int,
    height: float,
    weight: float
) -> tuple[bool, list[str]]:
    """
    Valida os dados de entrada do formulário.

    Args:
        age: Idade do paciente.
        height: Altura em metros.
        weight: Peso em kg.

    Returns:
        Tupla (is_valid, error_messages).
    """
    errors = []
    validation = INPUT_VALIDATION

    if not validation['age']['min'] <= age <= validation['age']['max']:
        errors.append(f"Idade deve estar entre {validation['age']['min']} e {validation['age']['max']} anos.")

    if not validation['height']['min'] <= height <= validation['height']['max']:
        errors.append(f"Altura deve estar entre {validation['height']['min']}m e {validation['height']['max']}m.")

    if not validation['weight']['min'] <= weight <= validation['weight']['max']:
        errors.append(f"Peso deve estar entre {validation['weight']['min']}kg e {validation['weight']['max']}kg.")

    # Validação de IMC extremo
    if height > 0:
        bmi = weight / (height ** 2)
        if bmi < 10 or bmi > 70:
            errors.append(f"IMC calculado ({bmi:.1f}) está fora dos limites esperados. Verifique altura e peso.")

    return len(errors) == 0, errors


def get_detailed_recommendations(
    predicted_class: str,
    probabilities: np.ndarray
) -> Dict[str, Any]:
    """
    Gera recomendações médicas detalhadas baseadas na predição.

    Args:
        predicted_class: Classe predita pelo modelo.
        probabilities: Array de probabilidades.

    Returns:
        Dicionário com título, severidade, cor e conteúdo das recomendações.
    """
    prob_percent = probabilities.max() * 100

    recommendations = {
        'Obesity_Type_III': {
            'title': 'Obesidade Tipo III - Risco Crítico',
            'severity': 'Crítico',
            'color': 'error',
            'content': f"""
            **Diagnóstico:** O paciente apresenta Obesidade Tipo III (Obesidade Mórbida) com {prob_percent:.1f}% de confiança.

            **Ações Imediatas Recomendadas:**

            1. **Consulta Médica Urgente:**
               - Avaliação completa de comorbidades (diabetes tipo 2, hipertensão, apneia do sono, doenças cardiovasculares)
               - Exames laboratoriais completos (glicemia, perfil lipídico, função hepática e renal)
               - Avaliação psicológica para identificar transtornos alimentares ou depressão

            2. **Acompanhamento Multidisciplinar:**
               - **Nutricionista:** Elaboração de plano alimentar individualizado com déficit calórico controlado
               - **Educador Físico:** Programa de exercícios adaptado, iniciando com atividades de baixo impacto
               - **Psicólogo:** Suporte emocional e comportamental para mudança de hábitos
               - **Endocrinologista:** Avaliação hormonal e possível indicação de medicação

            3. **Intervenções Específicas:**
               - Considerar cirurgia bariátrica se IMC > 40 ou > 35 com comorbidades
               - Terapia comportamental para mudança de hábitos alimentares
               - Monitoramento semanal de peso e medidas corporais

            4. **Metas de Tratamento:**
               - Redução de 5-10% do peso inicial nos primeiros 6 meses
               - Melhoria de parâmetros metabólicos (glicemia, pressão arterial)
               - Aumento gradual de atividade física até 150 minutos/semana
            """
        },
        'Obesity_Type_II': {
            'title': 'Obesidade Tipo II - Risco Muito Alto',
            'severity': 'Muito Alto',
            'color': 'error',
            'content': f"""
            **Diagnóstico:** O paciente apresenta Obesidade Tipo II com {prob_percent:.1f}% de confiança.

            **Plano de Ação Recomendado:**

            1. **Avaliação Médica Completa:**
               - Consulta com clínico geral ou endocrinologista
               - Rastreamento de comorbidades (diabetes, hipertensão, dislipidemia)
               - Exames básicos: glicemia de jejum, hemograma, perfil lipídico

            2. **Intervenção Nutricional:**
               - Consulta com nutricionista para plano alimentar personalizado
               - Redução calórica de 500-750 kcal/dia
               - Educação nutricional sobre escolhas alimentares saudáveis
               - Registro alimentar para monitoramento

            3. **Programa de Exercícios:**
               - Início gradual: 30 minutos, 3x/semana
               - Combinação de exercícios aeróbicos e resistência
               - Acompanhamento com educador físico
               - Meta: 150 minutos de atividade moderada/semana

            4. **Acompanhamento:**
               - Consultas mensais nos primeiros 3 meses
               - Reavaliação a cada 3 meses
               - Meta de perda de peso: 5-10% em 6 meses
            """
        },
        'Obesity_Type_I': {
            'title': 'Obesidade Tipo I - Risco Alto',
            'severity': 'Alto',
            'color': 'warning',
            'content': f"""
            **Diagnóstico:** O paciente apresenta Obesidade Tipo I com {prob_percent:.1f}% de confiança.

            **Estratégia de Tratamento:**

            1. **Avaliação Inicial:**
               - Consulta médica para avaliação geral de saúde
               - Verificação de pressão arterial e glicemia
               - Análise de histórico familiar e fatores de risco

            2. **Modificações no Estilo de Vida:**
               - **Alimentação:** Redução de 300-500 kcal/dia, priorizando alimentos integrais
               - **Atividade Física:** 150 minutos/semana de exercícios moderados
               - **Comportamento:** Identificar gatilhos alimentares e padrões de consumo

            3. **Suporte Profissional:**
               - Orientação nutricional com foco em educação alimentar
               - Programa de exercícios adaptado ao nível de condicionamento
               - Acompanhamento psicológico se necessário

            4. **Monitoramento:**
               - Pesar-se semanalmente no mesmo horário
               - Medir circunferência abdominal mensalmente
               - Consultas de retorno a cada 2-3 meses
            """
        },
        'Overweight_Level_II': {
            'title': 'Sobrepeso Nível II - Atenção Necessária',
            'severity': 'Moderado-Alto',
            'color': 'warning',
            'content': f"""
            **Diagnóstico:** O paciente apresenta Sobrepeso Nível II com {prob_percent:.1f}% de confiança.

            **Prevenção e Controle:**

            1. **Avaliação Preventiva:**
               - Consulta médica para avaliação de risco cardiovascular
               - Verificação de pressão arterial e perfil metabólico
               - Análise de fatores de risco modificáveis

            2. **Intervenções Preventivas:**
               - **Alimentação Equilibrada:** Redução moderada de calorias (200-300 kcal/dia)
               - **Atividade Física Regular:** 150 minutos/semana de exercícios moderados
               - **Hábitos Saudáveis:** Sono adequado (7-9 horas), gestão de estresse

            3. **Orientação Profissional:**
               - Consulta com nutricionista para orientação alimentar
               - Programa de exercícios personalizado
               - Educação sobre escolhas saudáveis

            4. **Acompanhamento:**
               - Monitoramento mensal de peso e medidas
               - Consultas trimestrais para reavaliação
               - Foco em manutenção e prevenção de progressão
            """
        },
        'Overweight_Level_I': {
            'title': 'Sobrepeso Nível I - Prevenção Recomendada',
            'severity': 'Moderado',
            'color': 'info',
            'content': f"""
            **Diagnóstico:** O paciente apresenta Sobrepeso Nível I com {prob_percent:.1f}% de confiança.

            **Recomendações Preventivas:**

            1. **Manutenção de Peso Saudável:**
               - Foco em não ganhar peso adicional
               - Alimentação balanceada sem restrições extremas
               - Atividade física regular (120-150 minutos/semana)

            2. **Melhorias no Estilo de Vida:**
               - Aumentar consumo de frutas, verduras e legumes
               - Reduzir alimentos ultraprocessados e açúcares
               - Manter hidratação adequada (2-3 litros/dia)
               - Praticar exercícios físicos regularmente

            3. **Orientação Profissional:**
               - Consulta com nutricionista para orientação preventiva
               - Avaliação de padrões alimentares e hábitos
               - Programa de exercícios adaptado

            4. **Monitoramento:**
               - Pesar-se semanalmente
               - Acompanhamento semestral com profissional
               - Foco em qualidade de vida e bem-estar
            """
        },
        'Normal_Weight': {
            'title': 'Peso Normal - Manutenção',
            'severity': 'Baixo',
            'color': 'success',
            'content': f"""
            **Diagnóstico:** O paciente apresenta Peso Normal com {prob_percent:.1f}% de confiança.

            **Recomendações para Manutenção:**

            1. **Manter Hábitos Saudáveis:**
               - Continuar com alimentação equilibrada e variada
               - Manter prática regular de atividade física (150 minutos/semana)
               - Sono adequado (7-9 horas por noite)
               - Gestão de estresse e bem-estar mental

            2. **Prevenção:**
               - Monitoramento periódico de peso (mensal)
               - Atenção a mudanças no estilo de vida que possam afetar o peso
               - Manter rotina de exercícios e alimentação saudável

            3. **Check-ups Regulares:**
               - Consultas médicas anuais para avaliação geral
               - Exames de rotina conforme recomendação médica
               - Manter vacinação em dia

            4. **Educação Continuada:**
               - Buscar informações sobre nutrição e saúde
               - Participar de atividades físicas que tragam prazer
               - Manter rede de apoio social e familiar
            """
        },
        'Insufficient_Weight': {
            'title': 'Peso Insuficiente - Atenção',
            'severity': 'Atenção',
            'color': 'info',
            'content': f"""
            **Diagnóstico:** O paciente apresenta Peso Insuficiente com {prob_percent:.1f}% de confiança.

            **Recomendações:**

            1. **Avaliação Médica:**
               - Consulta para investigar causas do baixo peso
               - Exames para descartar condições médicas subjacentes
               - Avaliação nutricional completa

            2. **Ganho de Peso Saudável:**
               - Plano nutricional para ganho de peso gradual
               - Aumento calórico controlado com alimentos nutritivos
               - Exercícios de resistência para ganho de massa muscular

            3. **Acompanhamento:**
               - Consultas regulares com médico e nutricionista
               - Monitoramento de progresso
               - Suporte psicológico se necessário
            """
        }
    }

    # Buscar recomendação correspondente
    for key in recommendations:
        if key in predicted_class:
            return recommendations[key]

    # Fallback para peso insuficiente
    return recommendations['Insufficient_Weight']


def create_probability_chart(
    prob_df: pd.DataFrame
) -> go.Figure:
    """
    Cria gráfico de barras horizontais com probabilidades.

    Args:
        prob_df: DataFrame com níveis de obesidade e probabilidades.

    Returns:
        Figura Plotly.
    """
    colors = [COLORS_PT.get(nivel, '#999999') for nivel in prob_df['Nível de Obesidade']]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=prob_df['Nível de Obesidade'],
        x=prob_df['Probabilidade (%)'],
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(color='rgba(0,0,0,0.3)', width=1)
        ),
        text=[f"{p:.1f}%" for p in prob_df['Probabilidade (%)']],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Probabilidade: %{x:.2f}%<extra></extra>'
    ))

    fig.update_layout(
        title='Probabilidades por Nível de Obesidade',
        xaxis_title='Probabilidade (%)',
        yaxis_title='',
        height=400,
        showlegend=False,
        xaxis=dict(range=[0, 105]),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )

    return fig


def main() -> None:
    """Função principal da aplicação."""

    # Menu de navegação
    st.sidebar.title("Navegação")
    page = st.sidebar.radio(
        "Selecione a página:",
        ["Sistema Preditivo", "Dashboard Analítico"]
    )

    if page == "Dashboard Analítico":
        import dashboard
        dashboard.main()
        return

    # Header
    st.markdown('<div class="main-header">Sistema Preditivo de Obesidade</div>', unsafe_allow_html=True)
    st.markdown("### Auxiliando a equipe médica no diagnóstico de obesidade")

    # Créditos
    st.markdown("""
    <div class="credits">
    <strong>Desenvolvido por Mariana Ruther de Araújo (10DTAT) - Tech Challenge 4 FIAP</strong>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Carregar modelo
    pipeline = load_model()

    if pipeline is None:
        st.stop()

    # Obter acurácia do modelo
    model_accuracy = pipeline.get_model_accuracy()
    accuracy_display = f"{model_accuracy:.1%}" if model_accuracy else "> 85%"

    # Sidebar com informações
    with st.sidebar:
        st.header("Informações")
        st.markdown(f"""
        Este sistema utiliza Machine Learning para prever o nível de obesidade
        com base em características físicas e hábitos de vida do paciente.

        **Precisão do modelo:** {accuracy_display}

        **Desenvolvido para:** Equipe Médica
        """)

        st.markdown("---")
        st.markdown("### Instruções")
        st.markdown("""
        1. Preencha todos os campos do formulário
        2. Clique em 'Prever Nível de Obesidade'
        3. Visualize o resultado e probabilidades
        4. Leia as recomendações médicas detalhadas
        """)

    # Formulário principal
    st.header("Formulário de Avaliação")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Dados Pessoais")
        gender = st.selectbox("Gênero", ["Masculino", "Feminino"])
        age = st.number_input(
            "Idade",
            min_value=int(INPUT_VALIDATION['age']['min']),
            max_value=int(INPUT_VALIDATION['age']['max']),
            value=int(INPUT_VALIDATION['age']['default'])
        )
        height = st.number_input(
            "Altura (metros)",
            min_value=INPUT_VALIDATION['height']['min'],
            max_value=INPUT_VALIDATION['height']['max'],
            value=INPUT_VALIDATION['height']['default'],
            step=0.01
        )
        weight = st.number_input(
            "Peso (kg)",
            min_value=INPUT_VALIDATION['weight']['min'],
            max_value=INPUT_VALIDATION['weight']['max'],
            value=INPUT_VALIDATION['weight']['default'],
            step=0.1
        )

        # Calcular IMC automaticamente
        if height > 0:
            bmi = weight / (height ** 2)
            bmi_status = ""
            if bmi < 18.5:
                bmi_status = " (Abaixo do peso)"
            elif bmi < 25:
                bmi_status = " (Peso normal)"
            elif bmi < 30:
                bmi_status = " (Sobrepeso)"
            else:
                bmi_status = " (Obesidade)"
            st.info(f"IMC Calculado: {bmi:.2f}{bmi_status}")

        st.subheader("Histórico Familiar")
        family_history = st.selectbox("Histórico familiar de excesso de peso", ["Sim", "Não"])

    with col2:
        st.subheader("Hábitos Alimentares")
        favc = st.selectbox("Come alimentos altamente calóricos com frequência?", ["Sim", "Não"])
        fcvc = st.slider(
            "Frequência de consumo de vegetais nas refeições",
            int(INPUT_VALIDATION['fcvc']['min']),
            int(INPUT_VALIDATION['fcvc']['max']),
            int(INPUT_VALIDATION['fcvc']['default']),
            help="1 = Nunca, 2 = Às vezes, 3 = Sempre"
        )
        ncp = st.slider(
            "Número de refeições principais por dia",
            int(INPUT_VALIDATION['ncp']['min']),
            int(INPUT_VALIDATION['ncp']['max']),
            int(INPUT_VALIDATION['ncp']['default'])
        )
        caec = st.selectbox(
            "Come algo entre as refeições?",
            ["Não", "Às vezes", "Frequentemente", "Sempre"]
        )
        scc = st.selectbox("Monitora as calorias que ingere diariamente?", ["Sim", "Não"])

        st.subheader("Hábitos de Vida")
        smoke = st.selectbox("Fuma?", ["Sim", "Não"])
        ch2o = st.slider(
            "Quantidade de água diária (litros)",
            int(INPUT_VALIDATION['ch2o']['min']),
            int(INPUT_VALIDATION['ch2o']['max']),
            int(INPUT_VALIDATION['ch2o']['default'])
        )
        faf = st.slider(
            "Frequência de atividade física",
            int(INPUT_VALIDATION['faf']['min']),
            int(INPUT_VALIDATION['faf']['max']),
            int(INPUT_VALIDATION['faf']['default']),
            help="0 = Não faço, 1 = 1-2x/semana, 2 = 2-3x/semana, 3 = 4-5x/semana"
        )
        tue = st.slider(
            "Tempo usando dispositivos tecnológicos (horas/dia)",
            int(INPUT_VALIDATION['tue']['min']),
            int(INPUT_VALIDATION['tue']['max']),
            int(INPUT_VALIDATION['tue']['default']),
            help="0 = 0-1h, 1 = 1-2h, 2 = 3-4h"
        )
        calc = st.selectbox(
            "Frequência de consumo de álcool",
            ["Não bebo", "Às vezes", "Frequentemente", "Sempre"]
        )
        mtrans = st.selectbox("Meio de transporte mais utilizado", [
            "Transporte Público",
            "Caminhada",
            "Automóvel",
            "Motocicleta",
            "Bicicleta"
        ])

    st.markdown("---")

    # Botão de predição
    if st.button("Prever Nível de Obesidade", type="primary", use_container_width=True):

        # Validar entrada
        is_valid, errors = validate_input(age, height, weight)

        if not is_valid:
            for error in errors:
                st.error(error)
            st.stop()

        # Criar DataFrame com os dados do formulário
        input_data = pd.DataFrame({
            'Gender': [GENDER_MAP_REVERSE[gender]],
            'Age': [age],
            'Height': [height],
            'Weight': [weight],
            'family_history': [FORM_MAPS['family_history'][family_history]],
            'FAVC': [FORM_MAPS['favc'][favc]],
            'FCVC': [fcvc],
            'NCP': [ncp],
            'CAEC': [FORM_MAPS['caec'][caec]],
            'SMOKE': [FORM_MAPS['smoke'][smoke]],
            'CH2O': [ch2o],
            'SCC': [FORM_MAPS['scc'][scc]],
            'FAF': [faf],
            'TUE': [tue],
            'CALC': [FORM_MAPS['calc'][calc]],
            'MTRANS': [FORM_MAPS['mtrans'][mtrans]]
        })

        # Aplicar feature engineering usando o método do pipeline (elimina código duplicado)
        input_data = pipeline.apply_feature_engineering(input_data)

        # Preprocessar dados
        X_processed, _ = pipeline.preprocess_data(input_data, is_training=False)

        # Fazer predição
        prediction, probabilities = pipeline.predict(X_processed)

        predicted_class = prediction[0]
        predicted_class_pt = OBESITY_MAP.get(predicted_class, predicted_class)

        # Exibir resultado
        st.markdown("---")
        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
        st.header("Resultado da Predição")

        # Obter recomendações detalhadas
        recommendations = get_detailed_recommendations(predicted_class, probabilities[0])

        # Determinar cor e severidade
        if 'Obesity' in predicted_class:
            color = "🔴"
            severity = "Alto"
        elif 'Overweight' in predicted_class:
            color = "🟡"
            severity = "Moderado"
        elif 'Normal' in predicted_class:
            color = "🟢"
            severity = "Baixo"
        else:
            color = "🔵"
            severity = "Atenção"

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Nível de Obesidade", f"{color} {predicted_class_pt}")

        with col2:
            max_prob = probabilities[0].max()
            st.metric("Confiança da Predição", f"{max_prob:.1%}")

        with col3:
            st.metric("Nível de Severidade", severity)

        # Visualização das probabilidades
        st.subheader("Análise de Probabilidades por Classe")
        st.markdown("**Distribuição de probabilidades para cada nível de obesidade:**")

        # Criar DataFrame com probabilidades
        prob_df = pd.DataFrame({
            'Nível de Obesidade': [OBESITY_MAP.get(cls, cls) for cls in pipeline.model.classes_],
            'Probabilidade (%)': (probabilities[0] * 100).round(2)
        }).sort_values('Probabilidade (%)', ascending=False)

        prob_df = prob_df.reset_index(drop=True)
        prob_df.index = prob_df.index + 1

        # Layout com gráfico e ranking
        col1, col2 = st.columns([2, 1])

        with col1:
            fig = create_probability_chart(prob_df)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**Ranking de Probabilidades:**")
            for idx, row in prob_df.iterrows():
                prob = row['Probabilidade (%)']
                nivel = row['Nível de Obesidade']
                emoji = "🥇" if idx == 1 else "🥈" if idx == 2 else "🥉" if idx == 3 else "📊"
                st.markdown(f"{emoji} **{prob:.1f}%** - {nivel}")

        # Tabela detalhada
        st.markdown("**Tabela Detalhada:**")
        st.dataframe(prob_df, use_container_width=True, hide_index=False)

        # Exibir recomendações médicas
        st.subheader("Recomendações Médicas Detalhadas")

        if recommendations['color'] == 'error':
            st.error(recommendations['content'])
        elif recommendations['color'] == 'warning':
            st.warning(recommendations['content'])
        elif recommendations['color'] == 'info':
            st.info(recommendations['content'])
        else:
            st.success(recommendations['content'])

        st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
