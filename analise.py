import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Carregando os dados
data = pd.read_csv('/content/drive/MyDrive/analise de dados /star_classification.csv')

# Separando as características (X) e os rótulos (y)
X = data.drop('class', axis=1)  # Supondo que 'class' é a coluna que você deseja prever
y = data['class']

# Dividindo os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Criando o modelo de Floresta Aleatória
modelo_rf = RandomForestClassifier(n_estimators=50, random_state=42)

# Treinando o modelo
modelo_rf.fit(X_train, y_train)

# Fazendo predições no conjunto de teste
predicoes = modelo_rf.predict(X_test)

# Calculando a matriz de confusão
Matriz_confusao = confusion_matrix(y_test, predicoes)

# Criando um gráfico de heatmap da matriz de confusão
plt.figure(figsize=(10, 6))
sns.heatmap(Matriz_confusao, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Matriz de Confusão')
plt.xlabel('Valores Preditos')
plt.ylabel('Valores Reais')
plt.show()

# Avaliando o desempenho do modelo
acuracia = accuracy_score(y_test, predicoes)
relatorio_classificacao = classification_report(y_test, predicoes)

print(f'Acurácia: {acuracia}')
print('Relatório de Classificação:\n', relatorio_classificacao)


# Treinando o modelo
modelo_rf = RandomForestClassifier(n_estimators=50, random_state=42)

# Obtendo a acurácia do treinamento e teste em cada número de árvores
acuracia_treinamento = []
acuracia_teste = []

num_arvores = range(1, 51)  # ajuste conforme necessário

for n in num_arvores:
    modelo_rf.set_params(n_estimators=n)
    modelo_rf.fit(X_train, y_train)
    
    # Avaliando a acurácia no conjunto de treinamento
    acuracia_treinamento.append(accuracy_score(y_train, modelo_rf.predict(X_train)))
    
    # Avaliando a acurácia no conjunto de teste
    acuracia_teste.append(accuracy_score(y_test, modelo_rf.predict(X_test)))

# Criando um gráfico de acurácia ao longo do número de árvores
plt.figure(figsize=(10, 6))
plt.plot(num_arvores, acuracia_treinamento, label='Treinamento')
plt.plot(num_arvores, acuracia_teste, label='Teste')
plt.title('Acurácia ao Longo do Número de Árvores na Random Forest')
plt.xlabel('Número de Árvores')
plt.ylabel('Acurácia')
plt.legend()
plt.show()
