# 🧠 mac_final_prediction_function

Função final de predição para projeto do classificador facial MAC, ou Massive Attribute Classifier, em Python.
Uma solução estática para carregar o modelo, fazer inferências e gerar previsões a partir de imagens ou dados de teste.

## 🚩 Pré-requisitos

Instale as dependências do projeto:
pip install -r requirements.txt

## 🧩 Uso
📌 1. Carregar a função de predição

No seu script Python:
- selecione a imagem a ser classificada
- selecione a quantidade de passagens estocásticas para o cálculo de confiabilidade (m)
- nomeie o modelo armazenado na variável MODELO_HYBRID na lista de listas MODEL
- execute com ´`python pred_hybrid.py`

## 📊 Saída esperada

A função deve retornar, em formato de tabela pandas, algo como:
- Classe prevista
- Probabilidade/confiança
- Rótulo interpretável
