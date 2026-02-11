"""
Arquivo de configuração centralizado para o projeto de predição de obesidade.
Contém todas as constantes, mapeamentos e configurações usadas no projeto.
"""

from typing import Dict, List

# =============================================================================
# PATHS E ARQUIVOS
# =============================================================================
DATA_FILE: str = 'Obesity.csv'
MODEL_FILE: str = 'obesity_model.joblib'

# =============================================================================
# MAPEAMENTOS DE OBESIDADE
# =============================================================================
OBESITY_MAP: Dict[str, str] = {
    'Insufficient_Weight': 'Peso Insuficiente',
    'Normal_Weight': 'Peso Normal',
    'Overweight_Level_I': 'Sobrepeso Nível I',
    'Overweight_Level_II': 'Sobrepeso Nível II',
    'Obesity_Type_I': 'Obesidade Tipo I',
    'Obesity_Type_II': 'Obesidade Tipo II',
    'Obesity_Type_III': 'Obesidade Tipo III'
}

OBESITY_MAP_REVERSE: Dict[str, str] = {v: k for k, v in OBESITY_MAP.items()}

# =============================================================================
# CORES POR NÍVEL DE OBESIDADE (para visualizações)
# =============================================================================
COLORS: Dict[str, str] = {
    'Obesity_Type_III': '#d32f2f',      # Vermelho escuro - Crítico
    'Obesity_Type_II': '#f57c00',       # Laranja - Muito Alto
    'Obesity_Type_I': '#fbc02d',        # Amarelo - Alto
    'Overweight_Level_II': '#ffa726',   # Laranja claro - Moderado-Alto
    'Overweight_Level_I': '#ffcc02',    # Amarelo claro - Moderado
    'Normal_Weight': '#66bb6a',         # Verde - Normal
    'Insufficient_Weight': '#42a5f5'    # Azul - Atenção
}

# Cores em português (para gráficos que usam labels traduzidas)
COLORS_PT: Dict[str, str] = {
    'Obesidade Tipo III': '#d32f2f',
    'Obesidade Tipo II': '#f57c00',
    'Obesidade Tipo I': '#fbc02d',
    'Sobrepeso Nível II': '#ffa726',
    'Sobrepeso Nível I': '#ffcc02',
    'Peso Normal': '#66bb6a',
    'Peso Insuficiente': '#42a5f5'
}

# =============================================================================
# MAPEAMENTOS DE GÊNERO
# =============================================================================
GENDER_MAP: Dict[str, str] = {
    'Male': 'Masculino',
    'Female': 'Feminino'
}

GENDER_MAP_REVERSE: Dict[str, str] = {
    'Masculino': 'Male',
    'Feminino': 'Female'
}

# =============================================================================
# MAPEAMENTOS PARA FORMULÁRIO (Português -> Inglês)
# =============================================================================
FORM_MAPS: Dict[str, Dict[str, str]] = {
    'family_history': {'Sim': 'yes', 'Não': 'no'},
    'favc': {'Sim': 'yes', 'Não': 'no'},
    'scc': {'Sim': 'yes', 'Não': 'no'},
    'smoke': {'Sim': 'yes', 'Não': 'no'},
    'caec': {
        'Não': 'no',
        'Às vezes': 'Sometimes',
        'Frequentemente': 'Frequently',
        'Sempre': 'Always'
    },
    'calc': {
        'Não bebo': 'no',
        'Às vezes': 'Sometimes',
        'Frequentemente': 'Frequently',
        'Sempre': 'Always'
    },
    'mtrans': {
        'Transporte Público': 'Public_Transportation',
        'Caminhada': 'Walking',
        'Automóvel': 'Automobile',
        'Motocicleta': 'Motorbike',
        'Bicicleta': 'Bike'
    }
}

# =============================================================================
# CONFIGURAÇÕES DO MODELO
# =============================================================================
MODEL_CONFIG: Dict[str, any] = {
    'test_size': 0.2,
    'random_state': 42,
    'cv_folds': 5,
    'random_forest': {
        'n_estimators': 200,
        'max_depth': 20,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'n_jobs': -1
    },
    'gradient_boosting': {
        'n_estimators': 200,
        'max_depth': 10,
        'learning_rate': 0.1
    }
}

# =============================================================================
# CONFIGURAÇÕES DE FEATURE ENGINEERING
# =============================================================================
NUMERIC_COLS_TO_ROUND: List[str] = ['FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']

AGE_BINS: List[int] = [0, 20, 30, 40, 50, 100]
AGE_LABELS: List[str] = ['Adolescente', 'Jovem', 'Adulto', 'Meia-idade', 'Idoso']

BMI_THRESHOLDS: Dict[str, float] = {
    'underweight': 18.5,
    'normal': 25.0,
    'overweight': 30.0
}

# =============================================================================
# CONFIGURAÇÕES DO STREAMLIT
# =============================================================================
STREAMLIT_CONFIG: Dict[str, any] = {
    'page_title_app': 'Sistema Preditivo de Obesidade',
    'page_title_dashboard': 'Dashboard Analítico - Obesidade',
    'page_icon_app': '🏥',
    'page_icon_dashboard': '📊',
    'layout': 'wide',
    'sidebar_state': 'expanded'
}

# =============================================================================
# VALIDAÇÕES DE ENTRADA
# =============================================================================
INPUT_VALIDATION: Dict[str, Dict[str, float]] = {
    'age': {'min': 1, 'max': 120, 'default': 25},
    'height': {'min': 0.5, 'max': 2.5, 'default': 1.70},
    'weight': {'min': 20.0, 'max': 300.0, 'default': 70.0},
    'fcvc': {'min': 1, 'max': 3, 'default': 2},
    'ncp': {'min': 1, 'max': 4, 'default': 3},
    'ch2o': {'min': 1, 'max': 3, 'default': 2},
    'faf': {'min': 0, 'max': 3, 'default': 1},
    'tue': {'min': 0, 'max': 2, 'default': 1}
}

# =============================================================================
# MENSAGENS DE LOG
# =============================================================================
LOG_MESSAGES: Dict[str, str] = {
    'loading_data': 'Carregando dados...',
    'data_loaded': 'Dados carregados: {} registros, {} colunas',
    'feature_engineering': 'Iniciando Feature Engineering...',
    'preprocessing': 'Preprocessando dados...',
    'training': 'Treinando modelo...',
    'model_saved': 'Modelo salvo em: {}',
    'model_loaded': 'Modelo carregado de: {}',
    'pipeline_complete': 'Pipeline concluído com sucesso!'
}
