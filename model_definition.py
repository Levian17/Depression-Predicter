import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from torch.utils.data import Dataset, DataLoader

# Cargamos la data del archivo csv
df = pd.read_csv('Student Depression Dataset.csv')
df = df.dropna()

# Dividimos los datos en características (X) y objetivo (y)
X = df.drop(columns=['Depression'])
y = df['Depression']

# Separamos las columnas categóricas de las columnas numéricas
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(exclude=['object']).columns

# Aplicamos One-hot encoding a las columnas categóricas
encoder = OneHotEncoder(sparse_output=False)
encoded_cat_data = encoder.fit_transform(X[categorical_cols])

# Estandarizamos los datos numéricos
scaler = StandardScaler()
scaled_num_data = scaler.fit_transform(X[numerical_cols])

# Unimos los datos procesados
X_processed = np.hstack((encoded_cat_data, scaled_num_data))

# Realizamos la división en entrenamiento y prueba con un 20% para prueba
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=17)

# 2. Creamos un Dataset de PyTorch
class DepressionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.long)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Creamos los DataLoader para el batching
train_dataset = DepressionDataset(X_train, y_train)
test_dataset = DepressionDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 3. Definimos el modelo de la red neuronal
class DepressionModel(nn.Module):
    def __init__(self, input_size):
        super(DepressionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)  # Primera capa
        self.fc2 = nn.Linear(64, 32)          # Segunda capa
        self.fc3 = nn.Linear(32, 2)           # Capa de salida (clasificación binaria)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()      # Sigmoide para la salida de clasificación
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return self.sigmoid(x)  # Retorna las probabilidades para cada clase

# Inicializamos el modelo
input_size = X_train.shape[1]  # Número de características en el dataset procesado
model = DepressionModel(input_size)

# 4. Entrenamos el modelo
criterion = nn.CrossEntropyLoss()  # Para clasificación binaria
optimizer = torch.optim.SGD(params=model.parameters(), 
                            lr=0.1)
# Fase de entrenamiento
num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()  # Ponemos a cero los gradientes
        outputs = model(inputs)  # Pasamos los datos por el modelo
        loss = criterion(outputs, labels)  # Calculamos la pérdida
        loss.backward()  # Propagación hacia atrás
        optimizer.step()  # Actualizamos los pesos
        
        # Seguimiento de la precisión
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")

# 5. Evaluamos el modelo
model.eval()  # Ponemos el modelo en modo evaluación
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Precisión en prueba: {100 * correct / total:.2f}%")

# 6. Clasificamos nuevos datos
# def classify_new_data(model, new_data):
#     model.eval()  # Ponemos el modelo en modo evaluación
#     new_data_tensor = torch.tensor(new_data, dtype=torch.float32)
#     with torch.no_grad():
#         output = model(new_data_tensor)
#         _, predicted_class = torch.max(output.data, 1)
#     return predicted_class.item()  # Retorna la etiqueta de clase predicha (0 o 1)

# # Ejemplo: Predecir para nuevos datos
# new_data = np.array([['']])  # Nuevos datos a clasificar (deben ser preprocesados)
# new_data_processed = np.hstack((encoder.transform(new_data[:, categorical_cols]), scaler.transform(new_data[:, numerical_cols])))
# result = classify_new_data(model, new_data_processed)
# print(f"Clase predicha: {result}")