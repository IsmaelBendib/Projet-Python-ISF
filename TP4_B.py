import pandas as pd
import glob
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

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

# Création X et Y pour prédiction J+1
def create_target_features(df_scaled, n_days=30):
    x = []
    y = []
    for i in range(n_days, len(df_scaled)):
        x.append(df_scaled[i - n_days:i, 0])
        y.append(df_scaled[i, 0])
    return np.array(x), np.array(y)

# Pipeline pour un fichier
def prepare_dataset(file_path, n_days=30, split_ratio=0.8):
    df = load_close_prices(file_path)
    train_scaled, test_scaled, scaler = scale_and_split(df, split_ratio)
    x_train, y_train = create_target_features(train_scaled, n_days)
    x_test, y_test = create_target_features(test_scaled, n_days)
    return x_train, y_train, x_test, y_test, scaler, test_scaled

# Prédictions à J+2 sur tout le test set
def predict_j_plus_2_series(model, x_test, test_scaled, scaler):
    preds_j2 = []
    y_true_j2 = []

    for i in range(len(x_test) - 2):
        # prédire J+1
        last_window = x_test[i].reshape(1, -1)
        pred_j1_scaled = model.predict(last_window)[0]

        # créer fenêtre J+2
        next_window = np.append(last_window[:, 1:], [[pred_j1_scaled]], axis=1)
        pred_j2_scaled = model.predict(next_window)[0]

        # vrai J+2 dans test_scaled
        true_j2_scaled = test_scaled[i + 2 + 30, 0]  # 30 = taille fenêtre
        y_true_j2.append(true_j2_scaled)
        preds_j2.append(pred_j2_scaled)

    # Inversion des échelles
    preds_j2_inv = scaler.inverse_transform(np.array(preds_j2).reshape(-1, 1)).flatten()
    y_true_j2_inv = scaler.inverse_transform(np.array(y_true_j2).reshape(-1, 1)).flatten()
    return preds_j2_inv, y_true_j2_inv

# Liste des modèles
models = {
    "XGBoost": XGBRegressor(verbosity=0),
    "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=10),
    "LinearRegression": LinearRegression(),
    "KNN": KNeighborsRegressor(n_neighbors=5)
}

# Exécution sur toutes les entreprises et modèles
def regression_j2_all_models(folder_path, n_days=30):
    files = glob.glob(os.path.join(folder_path, "*.csv"))
    results = {}

    for file in files:
        name = os.path.basename(file).replace("_historical_data.csv", "")
        x_train, y_train, x_test, y_test, scaler, test_scaled = prepare_dataset(file, n_days=n_days)

        results[name] = {}
        for model_name, model in models.items():
            model.fit(x_train, y_train)
            preds_j2, y_true_j2 = predict_j_plus_2_series(model, x_test, test_scaled, scaler)
            results[name][model_name] = {"pred": preds_j2, "real": y_true_j2}

    return results

# Tracer les courbes J+2
def plot_j2_predictions_all_models(results):
    for company, model_data in results.items():
        plt.figure(figsize=(12, 6))
        plt.title(f"{company} - Prédictions J+2")

        # Afficher la série réelle (commune à tous les modèles)
        real = list(model_data.values())[0]["real"]
        plt.plot(real, label="Réel J+2", color="black")

        for model_name, data in model_data.items():
            plt.plot(data["pred"], label=f"{model_name} J+2")

        plt.legend()
        plt.tight_layout()
        os.makedirs("Résultats/J+2_all_models", exist_ok=True)
        plt.savefig(f"Résultats/J+2_all_models/{company}_j2_prediction.png")
        plt.close()

# Exécution complète
j2_all_model_results = regression_j2_all_models("Companies_historical_data")
plot_j2_predictions_all_models(j2_all_model_results)
