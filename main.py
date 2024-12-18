from src.GUI import initialize
from src.model_definition import train_and_export_model

# Entrenamos el modelo (si no esta entrenado)
train_and_export_model()

# Desplegamos la GUI
initialize()