"""
Aplicação Streamlit para predição de obesidade
Sistema preditivo para auxiliar equipe médica
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from ml_pipeline import ObesityPredictor
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    page_title="Sistema Preditivo de Obesidade",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Carrega o modelo treinado"""
    # entrega macro: carrego o modelo uma vez e reutilizo - cache_resource otimiza performance
    try:
        pipeline = ObesityPredictor()
        pipeline.load_model('obesity_model.joblib')
        return pipeline
    except:
        st.error("⚠️ Modelo não encontrado! Por favor, execute primeiro o ml_pipeline.py")
        return None

def get_detailed_recommendations(predicted_class, probabilities, obesity_map):
    """Gera recomendações médicas detalhadas baseadas na predição"""
    # entrega macro: recomendações médicas detalhadas por nível de obesidade
    max_prob = probabilities.max()
    prob_percent = max_prob * 100
    
    if 'Obesity_Type_III' in predicted_class:
        return {
            'title': '🔴 Obesidade Tipo III - Risco Crítico',
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
        }
    
    elif 'Obesity_Type_II' in predicted_class:
        return {
            'title': '🔴 Obesidade Tipo II - Risco Muito Alto',
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
        }
    
    elif 'Obesity_Type_I' in predicted_class:
        return {
            'title': '🟠 Obesidade Tipo I - Risco Alto',
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
        }
    
    elif 'Overweight_Level_II' in predicted_class:
        return {
            'title': '🟡 Sobrepeso Nível II - Atenção Necessária',
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
        }
    
    elif 'Overweight_Level_I' in predicted_class:
        return {
            'title': '🟡 Sobrepeso Nível I - Prevenção Recomendada',
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
        }
    
    elif 'Normal_Weight' in predicted_class:
        return {
            'title': '🟢 Peso Normal - Manutenção',
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
        }
    
    else:  # Insufficient_Weight
        return {
            'title': '🔵 Peso Insuficiente - Atenção',
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

def main():
    """Função principal da aplicação"""
    
    # menu de navegação entre sistema preditivo e dashboard - entrega importante para organização
    st.sidebar.title("🧭 Navegação")
    page = st.sidebar.radio(
        "Selecione a página:",
        ["🔮 Sistema Preditivo", "📊 Dashboard Analítico"]
    )
    
    if page == "📊 Dashboard Analítico":
        # importo e executo o dashboard quando selecionado
        import dashboard
        dashboard.main()
        return
    
    # Header
    st.markdown('<div class="main-header">🏥 Sistema Preditivo de Obesidade</div>', unsafe_allow_html=True)
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
    
    # Sidebar com informações
    with st.sidebar:
        st.header("ℹ️ Informações")
        st.markdown("""
        Este sistema utiliza Machine Learning para prever o nível de obesidade 
        com base em características físicas e hábitos de vida do paciente.
        
        **Precisão do modelo:** > 99%
        
        **Desenvolvido para:** Equipe Médica
        """)
        
        st.markdown("---")
        st.markdown("### 📋 Instruções")
        st.markdown("""
        1. Preencha todos os campos do formulário
        2. Clique em 'Prever Nível de Obesidade'
        3. Visualize o resultado e probabilidades
        4. Leia as recomendações médicas detalhadas
        """)
    
    # Formulário principal
    st.header("📝 Formulário de Avaliação")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dados Pessoais")
        gender = st.selectbox("Gênero", ["Masculino", "Feminino"])
        # Converter para inglês para o modelo
        gender_map = {"Masculino": "Male", "Feminino": "Female"}
        age = st.number_input("Idade", min_value=1, max_value=100, value=25)
        height = st.number_input("Altura (metros)", min_value=0.5, max_value=2.5, value=1.70, step=0.01)
        weight = st.number_input("Peso (kg)", min_value=20.0, max_value=200.0, value=70.0, step=0.1)
        
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
            st.info(f"📊 IMC Calculado: {bmi:.2f}{bmi_status}")
        
        st.subheader("Histórico Familiar")
        family_history = st.selectbox("Histórico familiar de excesso de peso", ["Sim", "Não"])
        family_map = {"Sim": "yes", "Não": "no"}
    
    with col2:
        st.subheader("Hábitos Alimentares")
        favc = st.selectbox("Come alimentos altamente calóricos com frequência?", ["Sim", "Não"])
        favc_map = {"Sim": "yes", "Não": "no"}
        fcvc = st.slider("Frequência de consumo de vegetais nas refeições", 1, 3, 2, 
                        help="1 = Nunca, 2 = Às vezes, 3 = Sempre")
        ncp = st.slider("Número de refeições principais por dia", 1, 4, 3)
        caec = st.selectbox("Come algo entre as refeições?", 
                           ["Não", "Às vezes", "Frequentemente", "Sempre"])
        caec_map = {"Não": "no", "Às vezes": "Sometimes", 
                   "Frequentemente": "Frequently", "Sempre": "Always"}
        scc = st.selectbox("Monitora as calorias que ingere diariamente?", ["Sim", "Não"])
        scc_map = {"Sim": "yes", "Não": "no"}
        
        st.subheader("Hábitos de Vida")
        smoke = st.selectbox("Fuma?", ["Sim", "Não"])
        smoke_map = {"Sim": "yes", "Não": "no"}
        ch2o = st.slider("Quantidade de água diária (litros)", 1, 3, 2)
        faf = st.slider("Frequência de atividade física", 0, 3, 1,
                        help="0 = Não faço, 1 = 1-2x/semana, 2 = 2-3x/semana, 3 = 4-5x/semana")
        tue = st.slider("Tempo usando dispositivos tecnológicos (horas/dia)", 0, 2, 1,
                       help="0 = 0-1h, 1 = 1-2h, 2 = 3-4h")
        calc = st.selectbox("Frequência de consumo de álcool", 
                           ["Não bebo", "Às vezes", "Frequentemente", "Sempre"])
        calc_map = {"Não bebo": "no", "Às vezes": "Sometimes", 
                   "Frequentemente": "Frequently", "Sempre": "Always"}
        mtrans = st.selectbox("Meio de transporte mais utilizado", [
            "Transporte Público", 
            "Caminhada", 
            "Automóvel", 
            "Motocicleta", 
            "Bicicleta"
        ])
        mtrans_map = {
            "Transporte Público": "Public_Transportation",
            "Caminhada": "Walking",
            "Automóvel": "Automobile",
            "Motocicleta": "Motorbike",
            "Bicicleta": "Bike"
        }
    
    st.markdown("---")
    
    # Botão de predição
    if st.button("🔮 Prever Nível de Obesidade", type="primary", use_container_width=True):
        
        # Criar DataFrame com os dados do formulário (convertendo para inglês)
        input_data = pd.DataFrame({
            'Gender': [gender_map[gender]],
            'Age': [age],
            'Height': [height],
            'Weight': [weight],
            'family_history': [family_map[family_history]],
            'FAVC': [favc_map[favc]],
            'FCVC': [fcvc],
            'NCP': [ncp],
            'CAEC': [caec_map[caec]],
            'SMOKE': [smoke_map[smoke]],
            'CH2O': [ch2o],
            'SCC': [scc_map[scc]],
            'FAF': [faf],
            'TUE': [tue],
            'CALC': [calc_map[calc]],
            'MTRANS': [mtrans_map[mtrans]]
        })
        
        # aplico o mesmo feature engineering do treino nos dados do formulário
        input_data['BMI'] = input_data['Weight'] / (input_data['Height'] ** 2)
        
        def categorize_bmi(bmi):
            if bmi < 18.5:
                return 'Underweight'
            elif bmi < 25:
                return 'Normal'
            elif bmi < 30:
                return 'Overweight'
            else:
                return 'Obese'
        
        input_data['BMI_Category'] = input_data['BMI'].apply(categorize_bmi)
        input_data['Age_Group'] = pd.cut(input_data['Age'], 
                                         bins=[0, 20, 30, 40, 50, 100], 
                                         labels=['Adolescente', 'Jovem', 'Adulto', 'Meia-idade', 'Idoso'])
        input_data['Healthy_Habits_Score'] = (
            (input_data['FCVC'] >= 2).astype(int) +
            (input_data['FAVC'] == 'no').astype(int) +
            (input_data['NCP'] >= 3).astype(int) +
            (input_data['SCC'] == 'yes').astype(int)
        )
        input_data['Activity_Score'] = (
            input_data['FAF'] +
            (input_data['TUE'] == 0).astype(int)
        )
        
        # preprocesso os dados usando os encoders e scaler do treino
        X_processed = pipeline.preprocess_data(input_data, is_training=False)
        
        # faço a predição - entrega macro: sistema preditivo funcionando em tempo real
        prediction, probabilities = pipeline.predict(X_processed)
        
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
        
        predicted_class = prediction[0]
        predicted_class_pt = obesity_map.get(predicted_class, predicted_class)
        
        # Exibir resultado
        st.markdown("---")
        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
        st.header("🎯 Resultado da Predição")
        
        # Obter recomendações detalhadas
        recommendations = get_detailed_recommendations(predicted_class, probabilities[0], obesity_map)
        
        # Determinar cor baseada na predição
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
        
        # visualização das probabilidades - entrega importante para clareza médica
        st.subheader("📊 Análise de Probabilidades por Classe")
        st.markdown("**Distribuição de probabilidades para cada nível de obesidade:**")
        
        # crio dataframe com todas as probabilidades ordenadas
        prob_df = pd.DataFrame({
            'Nível de Obesidade': [obesity_map.get(cls, cls) for cls in pipeline.model.classes_],
            'Probabilidade (%)': (probabilities[0] * 100).round(2)
        }).sort_values('Probabilidade (%)', ascending=False)
        
        # resetar índice para melhor visualização
        prob_df = prob_df.reset_index(drop=True)
        prob_df.index = prob_df.index + 1
        
        # layout com gráfico e ranking lado a lado
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # gráfico de barras horizontal com cores por severidade
            fig = go.Figure()
            
            # Cores baseadas na probabilidade e tipo
            colors = []
            for nivel in prob_df['Nível de Obesidade']:
                if 'Obesidade Tipo III' in nivel:
                    colors.append('#d32f2f')  # Vermelho escuro
                elif 'Obesidade Tipo II' in nivel:
                    colors.append('#f57c00')  # Laranja
                elif 'Obesidade Tipo I' in nivel:
                    colors.append('#fbc02d')  # Amarelo
                elif 'Sobrepeso' in nivel:
                    colors.append('#ffa726')  # Laranja claro
                elif 'Normal' in nivel:
                    colors.append('#66bb6a')  # Verde
                else:
                    colors.append('#42a5f5')  # Azul
            
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
        
        # exibo recomendações médicas detalhadas - entrega macro para auxiliar equipe médica
        st.subheader("💡 Recomendações Médicas Detalhadas")
        
        # uso cores diferentes conforme a severidade do caso
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
