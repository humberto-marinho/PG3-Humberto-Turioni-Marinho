import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

# getting the script directory
script_directory = os.path.dirname(os.path.realpath(__file__))

# combinating directory name with name from csv file
file_path = os.path.join(script_directory, 'gossipcop_data.csv')

# loading csv data
data = pd.read_csv(file_path, sep=';', decimal=',')

# dividing data (X) and labels (y)
X = data.iloc[:, :11]
y = data['veracity']

# dividing data between train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criando e treinando o modelo de árvore de decisão
model_tree = DecisionTreeClassifier(random_state=42)
model_tree.fit(X_train, y_train)

#assessing model perfomance
y_pred = model_tree.predict(X_test)
print("Acurácia:", accuracy_score(y_test, y_pred))
print("Relatório de Classificação:\n", classification_report(y_test, y_pred))

# acessing importance of resources
importancies = model_tree.feature_importances_

#printing resources importance
#for i, importancia in enumerate(importancies):
#   print(f'Feature {i+1}: {importancia}')

#Ajustando o tamanho da figura e salvando a árvore de decisão em um arquivo
plt.figure(figsize=(50, 25))
plot_tree(model_tree, filled=True, feature_names=list(X.columns), class_names=["0", "1"])  # adjust max_depth as the necessary
plt.savefig(os.path.join(script_directory, 'tree_gossipcop.png'), dpi=700)  #adjust dpi as the necessary
