# 🧠 mac_final_prediction_function

Função final de predição para projeto MAC (Machine-Assisted Classification / Hybrid Prediction) em Python.

Uma solução modular para carregar modelos, fazer inferências e gerar previsões a partir de imagens ou dados de teste.

## 🚀 Visão Geral

Este projeto contém uma função final de predição (final prediction function) desenvolvida para ser integrada em pipelines de inferência, aplicações de machine learning e/ou produção. O módulo principal (mac_hybrid) provavelmente contém a lógica principal de predição — possivelmente combinando métodos híbridos (ex.: CNN + MLP, regras + aprendizado de máquina, ensembles etc).

Também há um diretório test_imgs/ com exemplos de imagens usadas para testar a função de predição.

## 🚩 Pré-requisitos

Instale as dependências do projeto:
pip install -r requirements.txt

## 🧩 Uso
📌 1. Carregar a função de predição

No seu script Python:
- selecione a imagem a ser classificada
- selecione a quantidade de passagens estocásticas para o cálculo de confiabilidade (m)
- nomeie o modelo armazenado na variável MODELO_HYBRID na lista de listas MODEL
- execute com python pred_hybrid.py

## 📊 Saída esperada

A função deve retornar, em formato de tabela pandas, algo como:
Classe prevista
Probabilidade/confiança
Rótulo interpretável

Instale dependências e inicie o desenvolvimento com:

pip install -r requirements.txt
