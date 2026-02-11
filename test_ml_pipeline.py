"""
Testes unitários para o pipeline de Machine Learning.

Execute com: pytest test_ml_pipeline.py -v
"""

import os
import tempfile
from typing import Generator

import pytest
import pandas as pd
import numpy as np

from ml_pipeline import ObesityPredictor
from config import DATA_FILE, BMI_THRESHOLDS


# Fixtures
@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Cria dados de exemplo para testes."""
    # Dados maiores para permitir split estratificado (mínimo 2 por classe)
    np.random.seed(42)
    n_samples = 30  # 10 por classe

    data = {
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'Age': np.random.randint(18, 60, n_samples),
        'Height': np.round(np.random.uniform(1.50, 1.90, n_samples), 2),
        'Weight': np.round(np.random.uniform(50, 120, n_samples), 1),
        'family_history': np.random.choice(['yes', 'no'], n_samples),
        'FAVC': np.random.choice(['yes', 'no'], n_samples),
        'FCVC': np.random.randint(1, 4, n_samples),
        'NCP': np.random.randint(1, 5, n_samples),
        'CAEC': np.random.choice(['Sometimes', 'no', 'Frequently', 'Always'], n_samples),
        'SMOKE': np.random.choice(['yes', 'no'], n_samples),
        'CH2O': np.random.randint(1, 4, n_samples),
        'SCC': np.random.choice(['yes', 'no'], n_samples),
        'FAF': np.random.randint(0, 4, n_samples),
        'TUE': np.random.randint(0, 3, n_samples),
        'CALC': np.random.choice(['Sometimes', 'no', 'Frequently', 'Always'], n_samples),
        'MTRANS': np.random.choice(['Public_Transportation', 'Walking', 'Automobile', 'Bike', 'Motorbike'], n_samples),
        # 3 classes com 10 amostras cada para permitir stratify
        'Obesity': ['Normal_Weight'] * 10 + ['Obesity_Type_I'] * 10 + ['Obesity_Type_II'] * 10
    }

    return pd.DataFrame(data)


@pytest.fixture
def predictor() -> ObesityPredictor:
    """Cria instância do ObesityPredictor."""
    return ObesityPredictor()


@pytest.fixture
def temp_model_path() -> Generator[str, None, None]:
    """Cria caminho temporário para salvar modelo."""
    with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
        yield f.name
    # Cleanup
    if os.path.exists(f.name):
        os.remove(f.name)


# Testes da classe ObesityPredictor
class TestObesityPredictor:
    """Testes para a classe ObesityPredictor."""

    def test_init(self, predictor: ObesityPredictor) -> None:
        """Testa inicialização do predictor."""
        assert predictor.model is None
        assert predictor.scaler is not None
        assert predictor.label_encoders == {}
        assert predictor.imputer is not None
        assert predictor.feature_names is None
        assert predictor.model_accuracy is None

    def test_categorize_bmi_underweight(self, predictor: ObesityPredictor) -> None:
        """Testa categorização de IMC abaixo do peso."""
        assert predictor.categorize_bmi(17.0) == 'Underweight'
        assert predictor.categorize_bmi(18.4) == 'Underweight'

    def test_categorize_bmi_normal(self, predictor: ObesityPredictor) -> None:
        """Testa categorização de IMC normal."""
        assert predictor.categorize_bmi(18.5) == 'Normal'
        assert predictor.categorize_bmi(22.0) == 'Normal'
        assert predictor.categorize_bmi(24.9) == 'Normal'

    def test_categorize_bmi_overweight(self, predictor: ObesityPredictor) -> None:
        """Testa categorização de IMC sobrepeso."""
        assert predictor.categorize_bmi(25.0) == 'Overweight'
        assert predictor.categorize_bmi(27.5) == 'Overweight'
        assert predictor.categorize_bmi(29.9) == 'Overweight'

    def test_categorize_bmi_obese(self, predictor: ObesityPredictor) -> None:
        """Testa categorização de IMC obesidade."""
        assert predictor.categorize_bmi(30.0) == 'Obese'
        assert predictor.categorize_bmi(35.0) == 'Obese'
        assert predictor.categorize_bmi(40.0) == 'Obese'


class TestFeatureEngineering:
    """Testes para feature engineering."""

    def test_apply_feature_engineering_creates_bmi(
        self,
        predictor: ObesityPredictor,
        sample_data: pd.DataFrame
    ) -> None:
        """Testa se BMI é calculado corretamente."""
        result = predictor.apply_feature_engineering(sample_data)

        assert 'BMI' in result.columns
        # Verifica cálculo: BMI = Weight / Height^2
        for idx in range(len(sample_data)):
            weight = sample_data['Weight'].iloc[idx]
            height = sample_data['Height'].iloc[idx]
            expected_bmi = weight / (height ** 2)
            assert abs(result['BMI'].iloc[idx] - expected_bmi) < 0.01

    def test_apply_feature_engineering_creates_bmi_category(
        self,
        predictor: ObesityPredictor,
        sample_data: pd.DataFrame
    ) -> None:
        """Testa se BMI_Category é criado com valores válidos."""
        result = predictor.apply_feature_engineering(sample_data)

        assert 'BMI_Category' in result.columns
        valid_categories = {'Underweight', 'Normal', 'Overweight', 'Obese'}
        assert set(result['BMI_Category'].unique()).issubset(valid_categories)

    def test_apply_feature_engineering_creates_age_group(
        self,
        predictor: ObesityPredictor,
        sample_data: pd.DataFrame
    ) -> None:
        """Testa se Age_Group é criado com valores válidos."""
        result = predictor.apply_feature_engineering(sample_data)

        assert 'Age_Group' in result.columns
        valid_groups = {'Adolescente', 'Jovem', 'Adulto', 'Meia-idade', 'Idoso'}
        # Exclui NaN que podem aparecer se idade estiver fora dos bins
        actual_groups = set(result['Age_Group'].dropna().unique())
        assert actual_groups.issubset(valid_groups)

    def test_apply_feature_engineering_creates_healthy_habits_score(
        self,
        predictor: ObesityPredictor,
        sample_data: pd.DataFrame
    ) -> None:
        """Testa se Healthy_Habits_Score é criado."""
        result = predictor.apply_feature_engineering(sample_data)

        assert 'Healthy_Habits_Score' in result.columns
        assert result['Healthy_Habits_Score'].dtype in [np.int64, np.int32, int]

    def test_apply_feature_engineering_creates_activity_score(
        self,
        predictor: ObesityPredictor,
        sample_data: pd.DataFrame
    ) -> None:
        """Testa se Activity_Score é criado."""
        result = predictor.apply_feature_engineering(sample_data)

        assert 'Activity_Score' in result.columns

    def test_apply_feature_engineering_preserves_original_columns(
        self,
        predictor: ObesityPredictor,
        sample_data: pd.DataFrame
    ) -> None:
        """Testa se colunas originais são preservadas."""
        original_cols = sample_data.columns.tolist()
        result = predictor.apply_feature_engineering(sample_data)

        for col in original_cols:
            assert col in result.columns

    def test_apply_feature_engineering_does_not_modify_original(
        self,
        predictor: ObesityPredictor,
        sample_data: pd.DataFrame
    ) -> None:
        """Testa se DataFrame original não é modificado."""
        original_shape = sample_data.shape
        _ = predictor.apply_feature_engineering(sample_data)

        assert sample_data.shape == original_shape
        assert 'BMI' not in sample_data.columns


class TestPreprocessing:
    """Testes para preprocessamento de dados."""

    def test_preprocess_data_returns_tuple(
        self,
        predictor: ObesityPredictor,
        sample_data: pd.DataFrame
    ) -> None:
        """Testa se preprocess_data retorna tupla (X, y)."""
        df = predictor.apply_feature_engineering(sample_data)
        result = predictor.preprocess_data(df, is_training=True)

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_preprocess_data_training_sets_feature_names(
        self,
        predictor: ObesityPredictor,
        sample_data: pd.DataFrame
    ) -> None:
        """Testa se feature_names é definido no treino."""
        df = predictor.apply_feature_engineering(sample_data)
        predictor.preprocess_data(df, is_training=True)

        assert predictor.feature_names is not None
        assert len(predictor.feature_names) > 0

    def test_preprocess_data_training_creates_encoders(
        self,
        predictor: ObesityPredictor,
        sample_data: pd.DataFrame
    ) -> None:
        """Testa se label encoders são criados no treino."""
        df = predictor.apply_feature_engineering(sample_data)
        predictor.preprocess_data(df, is_training=True)

        assert len(predictor.label_encoders) > 0

    def test_preprocess_data_removes_target(
        self,
        predictor: ObesityPredictor,
        sample_data: pd.DataFrame
    ) -> None:
        """Testa se coluna target é removida de X."""
        df = predictor.apply_feature_engineering(sample_data)
        X, y = predictor.preprocess_data(df, is_training=True)

        assert 'Obesity' not in X.columns
        assert y is not None
        assert len(y) == len(sample_data)


class TestModelPersistence:
    """Testes para salvar e carregar modelo."""

    def test_save_model_without_training_raises_error(
        self,
        predictor: ObesityPredictor,
        temp_model_path: str
    ) -> None:
        """Testa se salvar modelo sem treinar gera erro."""
        with pytest.raises(ValueError):
            predictor.save_model(temp_model_path)

    def test_load_model_nonexistent_file_raises_error(
        self,
        predictor: ObesityPredictor
    ) -> None:
        """Testa se carregar arquivo inexistente gera erro."""
        with pytest.raises(FileNotFoundError):
            predictor.load_model('nonexistent_model.joblib')

    def test_predict_without_model_raises_error(
        self,
        predictor: ObesityPredictor,
        sample_data: pd.DataFrame
    ) -> None:
        """Testa se predição sem modelo gera erro."""
        df = predictor.apply_feature_engineering(sample_data.drop('Obesity', axis=1))

        with pytest.raises(ValueError):
            predictor.predict(df)


class TestIntegration:
    """Testes de integração do pipeline completo."""

    def test_full_pipeline_with_sample_data(
        self,
        predictor: ObesityPredictor,
        sample_data: pd.DataFrame,
        temp_model_path: str
    ) -> None:
        """Testa pipeline completo com dados de exemplo."""
        # Feature engineering
        df = predictor.apply_feature_engineering(sample_data)

        # Preprocessamento
        X, y = predictor.preprocess_data(df, is_training=True)

        # Treinamento
        accuracy = predictor.train_model(X, y)

        # Verificações
        assert accuracy > 0
        assert predictor.model is not None
        assert predictor.model_accuracy is not None

        # Salvar e carregar
        predictor.save_model(temp_model_path)

        new_predictor = ObesityPredictor()
        new_predictor.load_model(temp_model_path)

        assert new_predictor.model is not None
        assert new_predictor.feature_names == predictor.feature_names

    def test_prediction_with_new_data(
        self,
        predictor: ObesityPredictor,
        sample_data: pd.DataFrame,
        temp_model_path: str
    ) -> None:
        """Testa predição com novos dados."""
        # Treinar modelo
        df = predictor.apply_feature_engineering(sample_data)
        X, y = predictor.preprocess_data(df, is_training=True)
        predictor.train_model(X, y)

        # Novo dado para predição
        new_data = pd.DataFrame({
            'Gender': ['Male'],
            'Age': [30],
            'Height': [1.75],
            'Weight': [80],
            'family_history': ['yes'],
            'FAVC': ['yes'],
            'FCVC': [2],
            'NCP': [3],
            'CAEC': ['Sometimes'],
            'SMOKE': ['no'],
            'CH2O': [2],
            'SCC': ['no'],
            'FAF': [1],
            'TUE': [1],
            'CALC': ['Sometimes'],
            'MTRANS': ['Public_Transportation']
        })

        # Aplicar feature engineering e preprocessar
        new_data = predictor.apply_feature_engineering(new_data)
        X_new, _ = predictor.preprocess_data(new_data, is_training=False)

        # Predição
        predictions, probabilities = predictor.predict(X_new)

        assert len(predictions) == 1
        assert probabilities.shape[0] == 1
        assert probabilities.shape[1] > 0
        assert np.abs(probabilities.sum() - 1.0) < 0.01


class TestDataLoading:
    """Testes para carregamento de dados."""

    def test_load_data_file_not_found(
        self,
        predictor: ObesityPredictor
    ) -> None:
        """Testa erro ao carregar arquivo inexistente."""
        with pytest.raises(FileNotFoundError):
            predictor.load_data('nonexistent_file.csv')

    @pytest.mark.skipif(
        not os.path.exists(DATA_FILE),
        reason=f"Arquivo {DATA_FILE} não encontrado"
    )
    def test_load_data_real_file(
        self,
        predictor: ObesityPredictor
    ) -> None:
        """Testa carregamento do arquivo real."""
        df = predictor.load_data(DATA_FILE)

        assert df is not None
        assert len(df) > 0
        assert 'Obesity' in df.columns


class TestConfig:
    """Testes para configurações."""

    def test_bmi_thresholds_are_valid(self) -> None:
        """Testa se thresholds de BMI são válidos."""
        assert BMI_THRESHOLDS['underweight'] < BMI_THRESHOLDS['normal']
        assert BMI_THRESHOLDS['normal'] < BMI_THRESHOLDS['overweight']

    def test_bmi_thresholds_are_positive(self) -> None:
        """Testa se thresholds de BMI são positivos."""
        for key, value in BMI_THRESHOLDS.items():
            assert value > 0, f"{key} deve ser positivo"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
