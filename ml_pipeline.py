"""
Pipeline completo de Machine Learning para predição de obesidade.
Inclui: EDA, Feature Engineering, Treinamento e Avaliação.

Este módulo implementa a classe ObesityPredictor que gerencia todo o ciclo
de vida do modelo de ML, desde o carregamento dos dados até a predição.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer
import joblib

from config import (
    DATA_FILE, MODEL_FILE, MODEL_CONFIG,
    NUMERIC_COLS_TO_ROUND, AGE_BINS, AGE_LABELS, BMI_THRESHOLDS
)

# Configuração do logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class ObesityPredictor:
    """
    Classe principal do pipeline de Machine Learning para predição de obesidade.

    Gerencia o modelo treinado, preprocessadores (scaler, encoders, imputer)
    e fornece métodos para treino e predição.

    Attributes:
        model: Modelo de ML treinado (RandomForest ou GradientBoosting)
        scaler: StandardScaler para normalização de features numéricas
        label_encoders: Dict de LabelEncoders para variáveis categóricas
        imputer: SimpleImputer para tratamento de valores faltantes
        feature_names: Lista de nomes das features para consistência
        model_accuracy: Acurácia do modelo no conjunto de teste
    """

    def __init__(self) -> None:
        """Inicializa o ObesityPredictor com preprocessadores vazios."""
        self.model: Optional[Any] = None
        self.scaler: StandardScaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.imputer: SimpleImputer = SimpleImputer(strategy='median')
        self.feature_names: Optional[List[str]] = None
        self.model_accuracy: Optional[float] = None

    def load_data(self, file_path: str = DATA_FILE) -> pd.DataFrame:
        """
        Carrega os dados do CSV.

        Args:
            file_path: Caminho para o arquivo CSV de dados.

        Returns:
            DataFrame com os dados carregados.

        Raises:
            FileNotFoundError: Se o arquivo não existir.
            pd.errors.EmptyDataError: Se o arquivo estiver vazio.
        """
        logger.info("Carregando dados de '%s'...", file_path)

        try:
            df = pd.read_csv(file_path)
            logger.info("Dados carregados: %d registros, %d colunas",
                       df.shape[0], df.shape[1])
            return df
        except FileNotFoundError:
            logger.error("Arquivo não encontrado: %s", file_path)
            raise
        except pd.errors.EmptyDataError:
            logger.error("Arquivo vazio: %s", file_path)
            raise

    @staticmethod
    def categorize_bmi(bmi: float) -> str:
        """
        Categoriza o IMC em faixas clínicas.

        Args:
            bmi: Valor do Índice de Massa Corporal.

        Returns:
            Categoria do IMC: 'Underweight', 'Normal', 'Overweight' ou 'Obese'.
        """
        if bmi < BMI_THRESHOLDS['underweight']:
            return 'Underweight'
        elif bmi < BMI_THRESHOLDS['normal']:
            return 'Normal'
        elif bmi < BMI_THRESHOLDS['overweight']:
            return 'Overweight'
        else:
            return 'Obese'

    def apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica feature engineering nos dados.

        Este método pode ser usado tanto no treinamento quanto na predição,
        garantindo consistência nas features derivadas.

        Args:
            df: DataFrame com os dados originais.

        Returns:
            DataFrame com as features derivadas adicionadas.
        """
        df = df.copy()

        # 1. IMC (Índice de Massa Corporal)
        df['BMI'] = df['Weight'] / (df['Height'] ** 2)

        # 2. Categoria do IMC
        df['BMI_Category'] = df['BMI'].apply(self.categorize_bmi)

        # 3. Normalizar valores numéricos (arredondar)
        for col in NUMERIC_COLS_TO_ROUND:
            if col in df.columns:
                df[col] = df[col].round()

        # 4. Grupo etário
        df['Age_Group'] = pd.cut(
            df['Age'],
            bins=AGE_BINS,
            labels=AGE_LABELS
        )

        # 5. Score de hábitos saudáveis
        df['Healthy_Habits_Score'] = (
            (df['FCVC'] >= 2).astype(int) +      # Come vegetais
            (df['FAVC'] == 'no').astype(int) +   # Não come alimentos calóricos
            (df['NCP'] >= 3).astype(int) +       # Faz 3+ refeições
            (df['SCC'] == 'yes').astype(int)     # Monitora calorias
        )

        # 6. Score de atividade física
        df['Activity_Score'] = (
            df['FAF'] +                          # Frequência atividade física
            (df['TUE'] == 0).astype(int)         # Não usa muito tecnologia
        )

        return df

    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Realiza feature engineering nos dados com logging.

        Wrapper do apply_feature_engineering com mensagens de log
        para uso no treinamento.

        Args:
            df: DataFrame com os dados originais.

        Returns:
            DataFrame com as features derivadas adicionadas.
        """
        logger.info("Iniciando Feature Engineering...")

        df_processed = self.apply_feature_engineering(df)

        logger.info("Feature Engineering concluído. Total de features: %d",
                   df_processed.shape[1])

        return df_processed

    def preprocess_data(
        self,
        df: pd.DataFrame,
        is_training: bool = True
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Preprocessa os dados para treinamento ou predição.

        Realiza encoding de variáveis categóricas, imputação de valores
        faltantes e prepara os dados para o modelo.

        Args:
            df: DataFrame com os dados.
            is_training: Se True, ajusta os encoders/imputer. Se False, usa os existentes.

        Returns:
            Tupla (X, y) onde X são as features processadas e y é o target
            (None se não houver coluna 'Obesity').
        """
        logger.info("Preprocessando dados...")
        df = df.copy()

        # Separar features e target
        if 'Obesity' in df.columns:
            X = df.drop('Obesity', axis=1)
            y = df['Obesity']
        else:
            X = df
            y = None

        # Identificar tipos de colunas
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        # Processar variáveis categóricas
        X_processed = pd.DataFrame()

        for col in categorical_cols:
            if is_training:
                le = LabelEncoder()
                X_processed[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders.get(col)
                if le:
                    # Tratar valores não vistos no treino
                    unique_values = set(le.classes_)
                    X[col] = X[col].astype(str).apply(
                        lambda x: x if x in unique_values else le.classes_[0]
                    )
                    X_processed[col] = le.transform(X[col].astype(str))
                else:
                    X_processed[col] = 0

        # Adicionar colunas numéricas
        for col in numeric_cols:
            X_processed[col] = X[col]

        # Imputação de valores faltantes
        if is_training:
            X_processed = pd.DataFrame(
                self.imputer.fit_transform(X_processed),
                columns=X_processed.columns
            )
            self.feature_names = X_processed.columns.tolist()
        else:
            X_processed = pd.DataFrame(
                self.imputer.transform(X_processed),
                columns=X_processed.columns
            )

        # Garantir ordem das colunas na predição
        if not is_training and self.feature_names:
            X_processed = X_processed.reindex(columns=self.feature_names, fill_value=0)

        logger.info("Preprocessamento concluído. Features: %d", X_processed.shape[1])

        if y is not None:
            return X_processed, y
        return X_processed, None

    def train_model(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> float:
        """
        Treina o modelo de Machine Learning.

        Testa Random Forest e Gradient Boosting, selecionando o melhor.
        Realiza validação cruzada para avaliar generalização.

        Args:
            X: DataFrame com as features.
            y: Series com o target.

        Returns:
            Acurácia do melhor modelo no conjunto de teste.
        """
        logger.info("Treinando modelo...")

        # Dividir dados em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=MODEL_CONFIG['test_size'],
            random_state=MODEL_CONFIG['random_state'],
            stratify=y
        )

        # Normalizar features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Definir modelos candidatos
        models = {
            'Random Forest': RandomForestClassifier(
                **MODEL_CONFIG['random_forest'],
                random_state=MODEL_CONFIG['random_state']
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                **MODEL_CONFIG['gradient_boosting'],
                random_state=MODEL_CONFIG['random_state']
            )
        }

        best_model = None
        best_score = 0.0
        best_name = ""

        for name, model in models.items():
            logger.info("Testando %s...", name)
            model.fit(X_train_scaled, y_train)
            score = model.score(X_test_scaled, y_test)
            logger.info("%s - Acurácia: %.4f", name, score)

            if score > best_score:
                best_score = score
                best_model = model
                best_name = name

        self.model = best_model
        self.model_accuracy = best_score
        logger.info("Melhor modelo: %s com acurácia de %.4f", best_name, best_score)

        # Avaliação detalhada
        y_pred = self.model.predict(X_test_scaled)

        logger.info("Relatório de Classificação:")
        print(classification_report(y_test, y_pred))

        logger.info("Matriz de Confusão:")
        print(confusion_matrix(y_test, y_pred))

        # Validação cruzada
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train,
            cv=MODEL_CONFIG['cv_folds']
        )
        logger.info(
            "Validação Cruzada (%d-fold): %.4f (+/- %.4f)",
            MODEL_CONFIG['cv_folds'], cv_scores.mean(), cv_scores.std() * 2
        )

        return best_score

    def save_model(self, filepath: str = MODEL_FILE) -> None:
        """
        Salva o modelo treinado e todos os preprocessadores.

        Args:
            filepath: Caminho para salvar o arquivo .joblib

        Raises:
            ValueError: Se o modelo não foi treinado.
            IOError: Se houver erro ao salvar o arquivo.
        """
        if self.model is None:
            raise ValueError("Modelo não foi treinado. Execute train_model() primeiro.")

        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'label_encoders': self.label_encoders,
                'imputer': self.imputer,
                'feature_names': self.feature_names,
                'model_accuracy': self.model_accuracy
            }
            joblib.dump(model_data, filepath)
            logger.info("Modelo salvo em: %s", filepath)
        except IOError as e:
            logger.error("Erro ao salvar modelo: %s", e)
            raise

    def load_model(self, filepath: str = MODEL_FILE) -> None:
        """
        Carrega o modelo treinado e todos os preprocessadores.

        Args:
            filepath: Caminho do arquivo .joblib

        Raises:
            FileNotFoundError: Se o arquivo não existir.
            KeyError: Se o arquivo estiver corrompido/incompleto.
        """
        try:
            data = joblib.load(filepath)
            self.model = data['model']
            self.scaler = data['scaler']
            self.label_encoders = data['label_encoders']
            self.feature_names = data['feature_names']

            # Campos que podem não existir em modelos antigos
            self.imputer = data.get('imputer', SimpleImputer(strategy='median'))
            self.model_accuracy = data.get('model_accuracy', None)

            logger.info("Modelo carregado de: %s", filepath)
        except FileNotFoundError:
            logger.error("Arquivo de modelo não encontrado: %s", filepath)
            raise
        except KeyError as e:
            logger.error("Arquivo de modelo corrompido. Chave faltando: %s", e)
            raise

    def predict(
        self,
        X: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Faz predições com o modelo treinado.

        Args:
            X: DataFrame com as features (já preprocessado).

        Returns:
            Tupla (predictions, probabilities) com as classes preditas
            e as probabilidades de cada classe.

        Raises:
            ValueError: Se o modelo não foi carregado/treinado.
        """
        if self.model is None:
            raise ValueError("Modelo não carregado. Execute load_model() primeiro.")

        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)

        return predictions, probabilities

    def get_model_accuracy(self) -> Optional[float]:
        """
        Retorna a acurácia do modelo.

        Returns:
            Acurácia do modelo ou None se não disponível.
        """
        return self.model_accuracy


def main() -> ObesityPredictor:
    """
    Função principal para executar o pipeline completo de ML.

    Returns:
        Instância do ObesityPredictor com o modelo treinado.
    """
    print("=" * 60)
    print("PIPELINE DE MACHINE LEARNING - PREDIÇÃO DE OBESIDADE")
    print("=" * 60)

    # Inicializar pipeline
    pipeline = ObesityPredictor()

    # Passo 1: Carregar dados
    df = pipeline.load_data(DATA_FILE)

    # Passo 2: Feature Engineering
    df_processed = pipeline.feature_engineering(df)

    # Passo 3: Preprocessamento
    X, y = pipeline.preprocess_data(df_processed, is_training=True)

    # Passo 4: Treinamento
    accuracy = pipeline.train_model(X, y)

    # Passo 5: Salvar modelo
    pipeline.save_model(MODEL_FILE)

    print("\n" + "=" * 60)
    print(f"PIPELINE CONCLUÍDO COM SUCESSO!")
    print(f"Acurácia Final: {accuracy:.2%}")
    print("=" * 60)

    return pipeline


if __name__ == "__main__":
    pipeline = main()
