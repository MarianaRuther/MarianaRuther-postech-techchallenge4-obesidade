"""
Pipeline completo de Machine Learning para predição de obesidade
Inclui: EDA, Feature Engineering, Treinamento e Avaliação
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer
import joblib
import warnings
warnings.filterwarnings('ignore')

class ObesityPredictor:
    # classe principal do pipeline de ml - aqui guardo o modelo treinado e os preprocessadores
    def __init__(self):
        self.model = None  # modelo de ml que será treinado
        self.scaler = StandardScaler()  # normalizador de features numéricas
        self.label_encoders = {}  # encoders para variáveis categóricas
        self.feature_names = None  # nomes das features para garantir consistência
        
    def load_data(self, file_path):
        """Carrega os dados do CSV"""
        print("📊 Carregando dados...")
        df = pd.read_csv(file_path)
        print(f"✅ Dados carregados: {df.shape[0]} registros, {df.shape[1]} colunas")
        return df
    
    def feature_engineering(self, df):
        """Realiza feature engineering nos dados"""
        # entrega macro: criação de features derivadas para melhorar a performance do modelo
        print("\n🔧 Iniciando Feature Engineering...")
        df = df.copy()
        
        # crio o imc que é uma feature importante para classificação de obesidade
        df['BMI'] = df['Weight'] / (df['Height'] ** 2)
        print("  ✓ Criado feature: BMI (Índice de Massa Corporal)")
        
        # 2. Categorizar IMC
        def categorize_bmi(bmi):
            if bmi < 18.5:
                return 'Underweight'
            elif bmi < 25:
                return 'Normal'
            elif bmi < 30:
                return 'Overweight'
            else:
                return 'Obese'
        
        df['BMI_Category'] = df['BMI'].apply(categorize_bmi)
        print("  ✓ Criado feature: BMI_Category")
        
        # 3. Normalizar valores numéricos que podem ter outliers
        numeric_cols = ['FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
        for col in numeric_cols:
            if col in df.columns:
                # Arredondar para valores inteiros mais próximos
                df[col] = df[col].round()
        
        # 4. Criar feature de idade categorizada
        df['Age_Group'] = pd.cut(df['Age'], 
                                 bins=[0, 20, 30, 40, 50, 100], 
                                 labels=['Adolescente', 'Jovem', 'Adulto', 'Meia-idade', 'Idoso'])
        print("  ✓ Criado feature: Age_Group")
        
        # crio score combinado de hábitos saudáveis - entrega importante para capturar padrões complexos
        df['Healthy_Habits_Score'] = (
            (df['FCVC'] >= 2).astype(int) +  # Come vegetais
            (df['FAVC'] == 'no').astype(int) +  # Não come alimentos calóricos
            (df['NCP'] >= 3).astype(int) +  # Faz 3+ refeições
            (df['SCC'] == 'yes').astype(int)  # Monitora calorias
        )
        print("  ✓ Criado feature: Healthy_Habits_Score")
        
        # score de atividade física - combina exercício e tempo de tela
        df['Activity_Score'] = (
            df['FAF'] +  # Frequência atividade física
            (df['TUE'] == 0).astype(int)  # Não usa muito tecnologia
        )
        print("  ✓ Criado feature: Activity_Score")
        
        print(f"✅ Feature Engineering concluído. Total de features: {df.shape[1]}")
        return df
    
    def preprocess_data(self, df, is_training=True):
        """Preprocessa os dados para treinamento"""
        # entrega macro: preparação dos dados para o modelo - encoding e normalização
        print("\n🔄 Preprocessando dados...")
        df = df.copy()
        
        # separo features (X) do target (y) que é a coluna Obesity
        if 'Obesity' in df.columns:
            X = df.drop('Obesity', axis=1)
            y = df['Obesity']
        else:
            X = df
            y = None
        
        # identifico automaticamente quais colunas são categóricas e numéricas
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        # processamento de variáveis categóricas usando label encoder
        X_processed = pd.DataFrame()
        
        for col in categorical_cols:
            if is_training:
                # no treinamento, crio o encoder e salvo para usar depois
                le = LabelEncoder()
                X_processed[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
            else:
                # na predição, uso o encoder salvo e trato valores não vistos
                le = self.label_encoders.get(col)
                if le:
                    # se aparecer um valor novo, uso o mais comum do treino
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
        
        # preencho valores faltantes com a mediana para não perder dados
        imputer = SimpleImputer(strategy='median')
        X_processed = pd.DataFrame(
            imputer.fit_transform(X_processed),
            columns=X_processed.columns
        )
        
        # salvo os nomes das features no treino para garantir ordem na predição
        if is_training:
            self.feature_names = X_processed.columns.tolist()
        
        # na predição, reordeno as colunas para bater com o treino
        if not is_training and self.feature_names:
            X_processed = X_processed.reindex(columns=self.feature_names, fill_value=0)
        
        print(f"✅ Preprocessamento concluído. Features: {X_processed.shape[1]}")
        
        if y is not None:
            return X_processed, y
        return X_processed
    
    def train_model(self, X, y):
        """Treina o modelo de Machine Learning"""
        # entrega macro: treinamento do modelo com validação e seleção do melhor algoritmo
        print("\n🤖 Treinando modelo...")
        
        # divido os dados em treino (80%) e teste (20%) mantendo proporção das classes
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # normalizo as features numéricas para melhor performance dos modelos
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # testo dois algoritmos e escolho o melhor - entrega importante para garantir acurácia > 75%
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=200,
                max_depth=10,
                learning_rate=0.1,
                random_state=42
            )
        }
        
        best_model = None
        best_score = 0
        best_name = None
        
        for name, model in models.items():
            print(f"  Testando {name}...")
            model.fit(X_train_scaled, y_train)
            score = model.score(X_test_scaled, y_test)
            print(f"    Acurácia: {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_model = model
                best_name = name
        
        self.model = best_model
        print(f"\n✅ Melhor modelo: {best_name} com acurácia de {best_score:.4f}")
        
        # avalio o modelo com métricas detalhadas para entender performance por classe
        y_pred = self.model.predict(X_test_scaled)
        print("\n📊 Relatório de Classificação:")
        print(classification_report(y_test, y_pred))
        
        print("\n📈 Matriz de Confusão:")
        print(confusion_matrix(y_test, y_pred))
        
        # validação cruzada para garantir que o modelo generaliza bem
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        print(f"\n✅ Validação Cruzada (5-fold): {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return best_score
    
    def save_model(self, filepath='obesity_model.joblib'):
        """Salva o modelo treinado"""
        # entrega macro: persistência do modelo para uso no streamlit
        import joblib
        # salvo tudo junto: modelo, scaler, encoders e nomes das features
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names
        }, filepath)
        print(f"\n💾 Modelo salvo em: {filepath}")
    
    def load_model(self, filepath='obesity_model.joblib'):
        """Carrega o modelo treinado"""
        import joblib
        data = joblib.load(filepath)
        self.model = data['model']
        self.scaler = data['scaler']
        self.label_encoders = data['label_encoders']
        self.feature_names = data['feature_names']
        print(f"✅ Modelo carregado de: {filepath}")
    
    def predict(self, X):
        """Faz predições com o modelo"""
        # entrega macro: função usada pelo streamlit para fazer predições em tempo real
        X_scaled = self.scaler.transform(X)  # normalizo antes de predizer
        predictions = self.model.predict(X_scaled)  # classe predita
        probabilities = self.model.predict_proba(X_scaled)  # probabilidades de cada classe
        return predictions, probabilities


def main():
    """Função principal para executar o pipeline completo"""
    # entrega macro: pipeline completo de ml - do carregamento até salvar o modelo
    print("=" * 60)
    print("🚀 PIPELINE DE MACHINE LEARNING - PREDIÇÃO DE OBESIDADE")
    print("=" * 60)
    
    # inicializo a classe que gerencia todo o pipeline
    pipeline = ObesityPredictor()
    
    # passo 1: carrego os dados do csv
    df = pipeline.load_data('Obesity.csv')
    
    # passo 2: crio features derivadas (imc, scores, categorias)
    df_processed = pipeline.feature_engineering(df)
    
    # passo 3: preprocesso (encoding, normalização, imputação)
    X, y = pipeline.preprocess_data(df_processed, is_training=True)
    
    # passo 4: treino o modelo e avalio - entrega crítica para garantir acurácia > 75%
    accuracy = pipeline.train_model(X, y)
    
    # passo 5: salvo tudo para usar no streamlit
    pipeline.save_model('obesity_model.joblib')
    
    print("\n" + "=" * 60)
    print(f"✅ PIPELINE CONCLUÍDO COM SUCESSO!")
    print(f"📊 Acurácia Final: {accuracy:.2%}")
    print("=" * 60)
    
    return pipeline


if __name__ == "__main__":
    pipeline = main()
