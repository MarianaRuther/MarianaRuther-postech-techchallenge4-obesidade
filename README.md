# 🏥 Sistema Preditivo de Obesidade - Tech Challenge Fase 4

Sistema completo de Machine Learning para auxiliar a equipe médica na predição e análise de obesidade.

## 📋 Sobre o Projeto

Este projeto foi desenvolvido como parte do Tech Challenge da FIAP - Fase 4, com o objetivo de criar um sistema preditivo que auxilie médicos e médicas a prever se uma pessoa pode ter obesidade, utilizando técnicas de Machine Learning e análise de dados.

## 🎯 Objetivos

- ✅ Pipeline completo de Machine Learning com feature engineering
- ✅ Modelo com assertividade acima de 75%
- ✅ Aplicação preditiva em Streamlit
- ✅ Dashboard analítico com insights para equipe médica

## 📁 Estrutura do Projeto

```
techchallenge4/
├── Obesity.csv                    # Base de dados
├── ml_pipeline.py                 # Pipeline completo de ML
├── app.py                         # Aplicação Streamlit (Sistema Preditivo)
├── dashboard.py                   # Dashboard Analítico
├── requirements.txt               # Dependências do projeto
├── README.md                      # Este arquivo
└── obesity_model.joblib          # Modelo treinado (gerado após execução)
```

## 🚀 Como Usar

### 1. Instalação das Dependências

```bash
pip install -r requirements.txt
```

### 2. Treinar o Modelo

Execute o pipeline de Machine Learning para treinar o modelo:

```bash
python ml_pipeline.py
```

Este script irá:
- Carregar e processar os dados
- Realizar feature engineering
- Treinar o modelo
- Avaliar a performance
- Salvar o modelo em `obesity_model.joblib`

### 3. Executar a Aplicação Preditiva

Para acessar o sistema preditivo:

```bash
streamlit run app.py
```

A aplicação estará disponível em `http://localhost:8501`

### 4. Executar o Dashboard Analítico

Para visualizar o dashboard com insights:

```bash
streamlit run dashboard.py
```

Ou acesse através da aplicação principal usando o menu lateral.

## 📊 Características do Modelo

- **Algoritmo:** Random Forest / Gradient Boosting
- **Acurácia:** > 85%
- **Validação:** Cross-validation 5-fold
- **Features Engenhadas:**
  - IMC (Índice de Massa Corporal)
  - Categorização de IMC
  - Grupos etários
  - Score de hábitos saudáveis
  - Score de atividade física

## 🎨 Funcionalidades

### Sistema Preditivo (app.py)
- Formulário interativo para entrada de dados do paciente
- Predição em tempo real do nível de obesidade
- Visualização de probabilidades por classe
- Recomendações médicas baseadas na predição

### Dashboard Analítico (dashboard.py)
- Métricas gerais sobre a população estudada
- Análise de distribuição de obesidade
- Análise por gênero
- Análise de fatores de risco
- Visualizações interativas
- Insights principais
- Recomendações para equipe médica

## 📈 Variáveis do Dataset

- **Gender:** Gênero
- **Age:** Idade
- **Height:** Altura (metros)
- **Weight:** Peso (kg)
- **family_history:** Histórico familiar de excesso de peso
- **FAVC:** Consumo frequente de alimentos calóricos
- **FCVC:** Frequência de consumo de vegetais
- **NCP:** Número de refeições principais por dia
- **CAEC:** Consumo entre refeições
- **SMOKE:** Hábito de fumar
- **CH2O:** Consumo diário de água
- **SCC:** Monitoramento de calorias
- **FAF:** Frequência de atividade física
- **TUE:** Tempo usando dispositivos tecnológicos
- **CALC:** Frequência de consumo de álcool
- **MTRANS:** Meio de transporte
- **Obesity:** Nível de obesidade (target)

## 🛠️ Tecnologias Utilizadas

- **Python 3.8+**
- **Pandas:** Manipulação de dados
- **NumPy:** Computação numérica
- **Scikit-learn:** Machine Learning
- **Streamlit:** Interface web
- **Plotly:** Visualizações interativas
- **Joblib:** Serialização do modelo

## 📝 Notas Importantes

1. O modelo deve ser treinado antes de usar a aplicação Streamlit
2. Certifique-se de ter o arquivo `Obesity.csv` na raiz do projeto
3. O modelo treinado será salvo como `obesity_model.joblib`

## 👥 Desenvolvido para

Equipe médica que precisa de ferramentas auxiliares para diagnóstico e prevenção de obesidade.

## 📄 Licença

Este projeto foi desenvolvido para fins educacionais como parte do Tech Challenge FIAP.

---

**Desenvolvido com ❤️ por mim para auxiliar alguma equipe médica um dia**
