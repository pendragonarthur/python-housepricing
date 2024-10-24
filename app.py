import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split

model = linear_model.LinearRegression()
df = pd.read_csv('data/data.csv')

# one hot encoding em colunas categoricas
df_encoded = pd.get_dummies(df, columns=['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']) 

def normalize_data(data): # usando min max scaling
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    n_data = data.copy()
    n_data[numeric_cols] = (data[numeric_cols] - data[numeric_cols].min()) / (data[numeric_cols].max() - data[numeric_cols].min())
    return n_data

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def r2_score(y_true, y_pred):
    return 1 - (np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2))

normalized_data = normalize_data(df_encoded)

X = normalized_data.drop('price', axis=1) # features (variaveis independentes)
y = normalized_data['price'] # target (variavel dependente)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)

y_test_pred = model.predict(X_test)
y_train_pred = model.predict(X_train)

print(f'MSE: {mean_squared_error(y_test, y_test_pred)}')
print(f'R2: {r2_score(y_test, y_test_pred)}')

plt.scatter(y_test, y_test_pred, color='blue', alpha=0.6, label='Previsões')
plt.plot([0, 1], [0, 1], transform=plt.gca().transAxes, color='red', label='Linha de Identidade')
plt.xlabel('Preço Real')
plt.ylabel('Preço Previsto')
plt.title('Previsões vs Preço Real')
plt.legend()
plt.show()