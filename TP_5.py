import numpy as np
import joblib
import os
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
import glob
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# Chargement des prix close
def load_close_prices(file_path):
    df = pd.read_csv(file_path)
    return df[['Close']]

# Standardisation + split
def scale_and_split(data, split_ratio=0.8):
    scaler = MinMaxScaler()
    train_size = int(len(data) * split_ratio)
    train_data = data[:train_size]
    test_data = data[train_size:]
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)
    return train_scaled, test_scaled, scaler

# Création X et Y
def create_target_features(df_scaled, n_days=30):
    x = []
    y = []
    for i in range(n_days, len(df_scaled)):
        x.append(df_scaled[i - n_days:i, 0])
        y.append(df_scaled[i, 0])
    return np.array(x), np.array(y)

# Pipeline complet pour un fichier
def prepare_dataset(file_path, n_days=30, split_ratio=0.8):
    df = load_close_prices(file_path)
    train_scaled, test_scaled, scaler = scale_and_split(df, split_ratio)
    x_train, y_train = create_target_features(train_scaled, n_days)
    x_test, y_test = create_target_features(test_scaled, n_days)
    return x_train, y_train, x_test, y_test, scaler

# Paramètres
data_folder = "Companies_historical_data"  # Dossier contenant les CSV
output_folder = "datasets"  # Dossier où on sauvegarde les fichiers
os.makedirs(output_folder, exist_ok=True)

n_days = 30
split_ratio = 0.8

# Lister tous les fichiers CSV
file_list = [f for f in os.listdir(data_folder) if f.endswith(".csv")]

# Pour chaque fichier, on prépare et sauvegarde les données
for filename in file_list:
    name = filename.replace("_historical_data.csv", "")  # "apple_historical_data.csv" -> "apple"
    file_path = os.path.join(data_folder, filename)

    # Préparation des données (fonction que tu as déjà)
    x_train, y_train, x_test, y_test, scaler = prepare_dataset(file_path, n_days, split_ratio)

    # Sauvegarde sous forme de fichiers .npy et .pkl
    np.save(os.path.join(output_folder, f"{name}_x_train.npy"), x_train)
    np.save(os.path.join(output_folder, f"{name}_y_train.npy"), y_train)
    np.save(os.path.join(output_folder, f"{name}_x_test.npy"), x_test)
    np.save(os.path.join(output_folder, f"{name}_y_test.npy"), y_test)
    joblib.dump(scaler, os.path.join(output_folder, f"{name}_scaler.pkl"))

    print(f"{name} : fichiers sauvegardés.")
    
def load_dataset(name, folder="datasets"):
    x_train = np.load(f"{folder}/{name}_x_train.npy", allow_pickle=True)
    y_train = np.load(f"{folder}/{name}_y_train.npy", allow_pickle=True)
    x_test = np.load(f"{folder}/{name}_x_test.npy", allow_pickle=True)
    y_test = np.load(f"{folder}/{name}_y_test.npy", allow_pickle=True)
    scaler = joblib.load(f"{folder}/{name}_scaler.pkl")
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
    return x_train, y_train, x_test, y_test, scaler

def build_mlp_model(input_shape, hidden_dims, activation, dropout_rate, optimizer, learning_rate):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=input_shape))
    for dim in hidden_dims:
        model.add(tf.keras.layers.Dense(dim, activation=activation))
        model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(1))
    opt = getattr(tf.keras.optimizers, optimizer)(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss='mean_squared_error')
    return model

def build_rnn_model(input_shape, hidden_dims, activation, dropout_rate, optimizer, learning_rate):
    model = tf.keras.Sequential()
    for i, dim in enumerate(hidden_dims):
        return_seq = i < len(hidden_dims) - 1
        model.add(tf.keras.layers.SimpleRNN(dim, activation=activation, return_sequences=return_seq, input_shape=input_shape if i == 0 else None))
        model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(1))
    opt = getattr(tf.keras.optimizers, optimizer)(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss='mean_squared_error')
    return model

def build_lstm_model(input_shape, hidden_dims, activation, dropout_rate, optimizer, learning_rate):
    model = tf.keras.Sequential()
    for i, dim in enumerate(hidden_dims):
        return_seq = i < len(hidden_dims) - 1
        model.add(tf.keras.layers.LSTM(dim, activation=activation, return_sequences=return_seq, input_shape=input_shape if i == 0 else None))
        model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(1))
    opt = getattr(tf.keras.optimizers, optimizer)(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss='mean_squared_error')
    return model

def train_model(model_type, X_train, y_train, input_shape, hidden_dims, activation, dropout_rate, optimizer, learning_rate, epochs, batch_size):
    if model_type == "MLP":
        model = build_mlp_model(input_shape, hidden_dims, activation, dropout_rate, optimizer, learning_rate)
    elif model_type == "RNN":
        model = build_rnn_model(input_shape, hidden_dims, activation, dropout_rate, optimizer, learning_rate)
    elif model_type == "LSTM":
        model = build_lstm_model(input_shape, hidden_dims, activation, dropout_rate, optimizer, learning_rate)
    else:
        raise ValueError("Modèle non reconnu.")

    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    return model

def predict(model, X_test, y_test, scaler, model_type, entreprise_name=None):
    # 1. Prédiction
    y_pred = model.predict(X_test)

    # 2. Inversion du scaling
    y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1))
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

    # 3. Évaluation
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    print(f"🔍 {entreprise_name or ''} - {model_type} : MAE = {mae:.4f}, RMSE = {rmse:.4f}")

    # 4. Affichage des 10 premières prédictions vs vraies valeurs
    print("Prédictions vs Réel (10 premières valeurs) :")
    for i in range(min(10, len(y_test_inv))):
        print(f"Réel : {y_test_inv[i][0]:.2f} \t Prédit : {y_pred_inv[i][0]:.2f}")

    # 5. Courbe réelle vs prédite (ex: 50 premiers points)
    plt.figure(figsize=(10, 4))
    plt.plot(y_test_inv[:50], label="Réel", marker='o')
    plt.plot(y_pred_inv[:50], label="Prévu", marker='x')
    plt.title(f"{model_type} - Prédiction vs Réel ({entreprise_name or 'Entreprise'})")
    plt.xlabel("Index")
    plt.ylabel("Prix")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # Crée le chemin Résultats/Prédiction/
    folder_path = os.path.join("Résultats", "Prédiction_TP5")
    os.makedirs(folder_path, exist_ok=True)

    # Nom du fichier
    filename = f"TP5_Prediction_{entreprise_name}.png"
    file_path = os.path.join(folder_path, filename)
    plt.close()
    
# Modèles à tester
model_types = ["MLP", "RNN", "LSTM"]

# Fichiers dans le dossier datasets
file_list = sorted([f for f in os.listdir("datasets") if f.endswith("_x_train.npy")])
entreprises = [f.split("_")[0] for f in file_list]

for entreprise in entreprises:
    print(f"Traitement de l’entreprise : {entreprise.upper()}")

    # Charger les données
    x_train, y_train, x_test, y_test, scaler = load_dataset(entreprise)

    for model_type in model_types:
        print(f"\n--- Modèle : {model_type} ---")

        # Entraînement
        model = train_model(
            model_type=model_type,
            X_train=x_train,
            y_train=y_train,
            input_shape=(x_train.shape[1], 1),
            hidden_dims=[50],
            activation="tanh" if model_type != "MLP" else "relu",
            dropout_rate=0.2,
            optimizer="Adam",
            learning_rate=0.001,
            epochs=20,
            batch_size=32
        )

        # Prédiction + Affichage
        predict(model, x_test, y_test, scaler, model_type=model_type, entreprise_name=entreprise)
