#!/bin/bash

# Script para executar o projeto completo

echo "🚀 Tech Challenge - Sistema Preditivo de Obesidade"
echo "=================================================="
echo ""

# Verificar se as dependências estão instaladas
echo "📦 Verificando dependências..."
python3 -c "import pandas, sklearn, streamlit, plotly" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Dependências não encontradas. Instalando..."
    pip install -r requirements.txt
fi

echo ""
echo "🤖 Passo 1: Treinando o modelo..."
python3 ml_pipeline.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Modelo treinado com sucesso!"
    echo ""
    echo "📱 Para executar a aplicação preditiva:"
    echo "   streamlit run app.py"
    echo ""
    echo "📊 Para executar o dashboard analítico:"
    echo "   streamlit run dashboard.py"
else
    echo "❌ Erro ao treinar o modelo. Verifique os logs acima."
fi
