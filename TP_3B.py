import pandas as pd
import glob
import ta
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, ROCIndicator
from ta.volatility import BollingerBands
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import shap

folder = "Companies_historical_data/"

import yfinance as yf

def download_macro_data(start="2019-03-14", end="2024-03-14"):
    symbols = {
        'VIX': '^VIX',
        'US10Y': '^TNX',
        'SP500': '^GSPC'
    }
    macro_df = pd.DataFrame()
    for name, ticker in symbols.items():
        data = yf.download(ticker, start=start, end=end)['Close']
        data.name = name
        macro_df = pd.concat([macro_df, data], axis=1)
    macro_df.index = pd.to_datetime(macro_df.index)
    macro_df = macro_df.fillna(method='ffill')
    return macro_df

def add_temporal_features(df):
    df = df.copy()
    df['Date'] = df.index
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['Weekday'] = df['Date'].dt.weekday
    df.drop(columns=['Date'], inplace=True)
    return df

def merge_macro_data(company_df, macro_df):
    df = company_df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.merge(macro_df, left_index=True, right_index=True, how='left')
    df = add_temporal_features(df)
    return df

def apply_technical_indicators_to_labeled_data_with_macro(labeled_dict):
    enriched_dict = {}
    macro_df = download_macro_data()  # Données macroéconomiques (avec index datetime)

    for filename, df in labeled_dict.items():
        try:
            df = df.copy()

            # ✅ Étape 1 : créer une colonne Date si elle n'existe pas
            if 'Date' not in df.columns:
                df['Date'] = pd.date_range(start="2019-01-01", periods=len(df), freq='B')

            # ✅ Étape 2 : convertir Date au bon format et la mettre en index
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)

            # ✅ Étape 3 : ajouter les indicateurs techniques
            df_enriched = add_technical_indicators(df)

            # ✅ Étape 4 : fusionner avec macro données + ajouter variables temporelles
            df_enriched = merge_macro_data(df_enriched, macro_df)

            # ✅ Étape 5 : nettoyage
            df_enriched.dropna(inplace=True)
            enriched_dict[filename] = df_enriched

        except Exception as e:
            print(f"Erreur pour {filename} : {e}")

    return enriched_dict


## ----- PARTIE 1 ----
def create_labels(df):
    df = df[['Close']].copy()
    df['Close Horizon'] = df['Close'].shift(-20)
    df['horizon return'] = (df['Close Horizon'] - df['Close']) / df['Close']
    df['label'] = df['horizon return'].apply(
        lambda x: 2 if x > 0.05 else (0 if x < -0.05 else 1)
    )
    return df


def apply_labeling_to_folder(folder_path):
    all_files = glob.glob(os.path.join(folder_path, "*.csv"))
    labeled_data = {}

    for file in all_files:
        try:
            df = pd.read_csv(file)
            if 'Close' not in df.columns:
                print(f"Fichier ignoré (pas de colonne 'Close') : {file}")
                continue
            df_labeled = create_labels(df)
            df_labeled.dropna(inplace=True)
            filename = os.path.basename(file)
            labeled_data[filename] = df_labeled
        except Exception as e:
            print(f"Erreur avec {file} : {e}")

    return labeled_data


labeled_dict = apply_labeling_to_folder(folder)

# Exemple
example_file = list(labeled_dict.keys())[0]
print(f"Exemple pour : {example_file}")
print(labeled_dict[example_file].head())


def add_technical_indicators(df):
    df['SMA 20'] = SMAIndicator(df['Close'], window=20).sma_indicator()
    df['EMA 20'] = EMAIndicator(df['Close'], window=20).ema_indicator()
    df['RSI 14'] = RSIIndicator(df['Close'], window=14).rsi()

    macd = MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD Signal'] = macd.macd_signal()

    boll = BollingerBands(df['Close'])
    df['Bollinger High'] = boll.bollinger_hband()
    df['Bollinger Low'] = boll.bollinger_lband()

    df['Rolling Volatility 20'] = df['Close'].rolling(window=20).std()
    df['ROC 10'] = ROCIndicator(df['Close'], window=10).roc()

    return df


def apply_technical_indicators_to_labeled_data(labeled_dict):
    enriched_dict = {}

    for filename, df in labeled_dict.items():
        try:
            df_enriched = add_technical_indicators(df)
            df_enriched.dropna(inplace=True)
            enriched_dict[filename] = df_enriched
        except Exception as e:
            print(f"Erreur pour {filename} : {e}")

    return enriched_dict


# Étape 1.1.1 : créer les labels
labeled_dict = apply_labeling_to_folder("Companies_historical_data")

# Étape 1.1.2 : ajouter les indicateurs techniques
enriched_dict = apply_technical_indicators_to_labeled_data(labeled_dict)

# Voir un exemple
example_file = list(enriched_dict.keys())[0]
print(enriched_dict[example_file].head())

enriched_dict = apply_technical_indicators_to_labeled_data_with_macro(labeled_dict)

# ---------Exemple avec les nouvelles colonnes
example_file = list(enriched_dict.keys())[0]
df_example = enriched_dict[example_file]

print(f"\nFichier exemple avec les nouvelles données : {example_file}")
print("\nListe des colonnes ajoutées :")
print(df_example.columns.tolist())

print("\nAperçu des premières lignes du DataFrame enrichi :")
print(df_example.head())

def prepare_dataset_for_classification(enriched_dict):
    # Concaténer tous les DataFrames en un seul grand DataFrame
    full_df = pd.concat(enriched_dict.values(), ignore_index=True)

    # Définir X (features) et y (label)
    X = full_df.drop(columns=['label', 'Close Horizon', 'horizon return', 'Close'])
    y = full_df['label']

    # Standardisation des features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test, X.columns


# On suppose que enriched_dict est déjà créé via apply_technical_indicators_to_labeled_data()
X_train, X_test, y_train, y_test, feature_names = prepare_dataset_for_classification(enriched_dict)

print("TP3")
print(f"Taille des données d'entraînement : {X_train.shape}")
print(f"Taille des données de test : {X_test.shape}")
print(f"Noms des features : {list(feature_names)}")


## ------- PARTE 2 ------
def train_and_evaluate_model(model, param_grid, X_train, X_test, y_train, y_test):
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    print(f"\nMeilleurs paramètres : {grid_search.best_params_}")
    print("\nRapport de classification :")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")

    return best_model


rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10],
    'min_samples_split': [2, 5]
}

best_rf = train_and_evaluate_model(RandomForestClassifier(random_state=42), rf_params, X_train, X_test, y_train, y_test)

xgb_params = {
    'n_estimators': [100, 200],
    'max_depth': [3, 6],
    'learning_rate': [0.01, 0.1]
}

best_xgb = train_and_evaluate_model(XGBClassifier(eval_metric='mlogloss'), xgb_params, X_train, X_test, y_train, y_test)
knn_params = {
    'n_neighbors': [3, 5, 10],
    'weights': ['uniform', 'distance']
}

best_knn = train_and_evaluate_model(KNeighborsClassifier(), knn_params, X_train, X_test, y_train, y_test)

# ⚠️ CHANGEMENT 1 : Kernel 'rbf' supprimé car trop lent et parfois instable sur gros datasets
svm_params = {
    'C': [0.1, 1, 10],
    'kernel': ['linear']  # seulement 'linear' pour éviter les lenteurs et bugs
}

# ⚠️ CHANGEMENT 2 : on réduit la taille des données pour éviter que SVC explose en temps de calcul
X_train_svm = X_train[:2000]
y_train_svm = y_train[:2000]
X_test_svm = X_test[:500]
y_test_svm = y_test[:500]

# ⚠️ CHANGEMENT 3 : on entoure le SVC avec un try/except pour capturer d'éventuelles erreurs
try:
    best_svm = train_and_evaluate_model(SVC(), svm_params, X_train_svm, X_test_svm, y_train_svm, y_test_svm)
except Exception as e:
    print(f"Erreur pendant le training SVM : {e}")
    best_svm = None

logreg_params = {
    'C': [0.1, 1, 10],
    'solver': ['liblinear']
}

best_logreg = train_and_evaluate_model(LogisticRegression(), logreg_params, X_train, X_test, y_train, y_test)

results = {
    'Modèle': ['Random Forest', 'XGBoost', 'KNN', 'SVM', 'Logistic Regression'],
    'Meilleure accuracy': [
        accuracy_score(y_test, best_rf.predict(X_test)),
        accuracy_score(y_test, best_xgb.predict(X_test)),
        accuracy_score(y_test, best_knn.predict(X_test)),
        accuracy_score(y_test, best_svm.predict(X_test)),
        accuracy_score(y_test, best_logreg.predict(X_test))
    ]
}

df_results = pd.DataFrame(results)
print(df_results)


## ----- PARTIE 3 ----
def explain_model_with_shap(model, X_train, X_test, feature_names):
    """
    Utilise SHAP pour expliquer un modèle et afficher les summary plots.
    """
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)

    # Importance globale des features
    shap.summary_plot(shap_values, features=X_test, feature_names=feature_names)

    # Importance spécifique pour les classes "Buy" et "Sell"
    if len(shap_values.values.shape) > 2:  # Vérifie si le modèle gère plusieurs classes
        shap.summary_plot(shap_values[:, :, 2], X_test, feature_names=feature_names, title="SHAP pour 'Buy'")
        shap.summary_plot(shap_values[:, :, 0], X_test, feature_names=feature_names, title="SHAP pour 'Sell'")
