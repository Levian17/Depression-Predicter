import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# Cargamos nuestro dataset
df = pd.read_csv('Student Depression Dataset.csv')

# Convertimos la data categorica en data numerica con one-hot-encoding
encoder = OneHotEncoder()  
categorical_columns = df.select_dtypes(include=['object', 'category']).columns 
encoded = encoder.fit_transform(df[categorical_columns]).toarray() 
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_columns))
df = pd.concat([df.drop(columns=categorical_columns), encoded_df], axis=1)

# Dividimos los registros en features y target (X, y)
X = df.drop(columns='Depression')
y = df['Depression']

# Dividimos la data en train y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)

