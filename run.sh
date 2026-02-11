#!/bin/bash

# Script para executar o projeto completo - Tech Challenge 4
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🚀 Tech Challenge - Sistema Preditivo de Obesidade"
echo "=================================================="
echo ""

# Para garantir que python3 existe
if ! command -v python3 &>/dev/null; then
    echo "❌ python3 não encontrado. Instale Python 3 e tente novamente."
    exit 1
fi

# Criar venv
VENV_DIR="$SCRIPT_DIR/venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "📁 Criando ambiente virtual..."
    python3 -m venv "$VENV_DIR"
fi

# Ativar venv
source "$VENV_DIR/bin/activate"

# Atualizar pip
echo "📦 Atualizando pip..."
pip install --quiet --upgrade pip

# Opção de confiar em PyPI
PIP_EXTRA="--trusted-host pypi.org --trusted-host files.pythonhosted.org"

# Verificar se as dependências estão instaladas
echo "📦 Verificando dependências..."
if ! python -c "import pandas, sklearn, streamlit, plotly" 2>/dev/null; then
    echo "⚠️  Instalando dependências (pode levar alguns minutos)..."
    if ! pip install $PIP_EXTRA -r requirements.txt; then
        echo "❌ Falha na instalação. Tente: pip install $PIP_EXTRA -r requirements.txt"
        exit 1
    fi
fi

echo ""
echo "🤖 Passo 1: Treinando o modelo..."
if python ml_pipeline.py; then
    echo ""
    echo "✅ Modelo treinado com sucesso!"
    echo ""
    echo "📱 Para executar a aplicação preditiva:"
    echo "   source venv/bin/activate && streamlit run app.py"
    echo ""
    echo "📊 Para executar o dashboard analítico:"
    echo "   source venv/bin/activate && streamlit run dashboard.py"
else
    echo ""
    echo "❌ Erro ao treinar o modelo. Verifique os logs acima."
    exit 1
fi
