#!/usr/bin/env python3
# main.py - Orchestration de la pipeline d'analyse financiere

import json
import logging
import os
import sys
from datetime import datetime
import glob
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import matplotlib
matplotlib.use("qt5agg")


# Creation des dossiers necessaires pour les outputs
os.makedirs("logs", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/pipeline_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Importation des TP
try:
    import TP_1
    import TP_2
    import TP_3
    import TP_4
    import TP_5
    import TP_6
    import TP_7
    import TP_8

    logger.info("All modules imported successfully")
except ImportError as e:
    logger.error(f"Error importing modules: {e}")
    sys.exit(1)


def update_data():
    """Execute le tp_1 pour mettre a jour les donnees financieres (prix, ratios financiers...)"""
    logger.info("Updating historical stock data")

    try:
        # Recupere les data pour la rangee choisie (de 2019 a aujd)
        start_date = '2019-01-01'
        end_date = datetime.now().strftime('%Y-%m-%d')

        # Creation du dossier des donnees historiques des entreprises
        output_dir = 'Companies_historical_data'
        os.makedirs(output_dir, exist_ok=True)

        #
        for company, symbol in TP_1.companies.items():
            try:
                TP_1.get_stock_variations(symbol, start_date, end_date)
                logger.info(f"Updated data for {company}")
            except Exception as e:
                logger.error(f"Error updating data for {company}: {e}")

        logger.info("Dzta update complete")
        return True
    except Exception as e:
        logger.error(f"Error in data update process: {e}")
        return False


def run_clustering():
    """Execute tp_2 pour identifier les entreprises similaires"""
    logger.info("Running clustering analysis")

    clustering_results = {
        'financial': {},
        'risk': {},
        'correlation': {}
    }

    try:
        # Clustering financier (kmeans)
        df_scaled = TP_2.preprocess_for_financial_clustering("financial_ratios.csv")

        # Extractions des resultats de K-means
        centres, affect = TP_2.clust.kmoyennes(4, df_scaled, epsilon=1e-4, iter_max=100, verbose=False)

        # On sauvegarde les assignations a chaque cluster
        for cluster_id, indices in affect.items():
            for idx in indices:
                company = df_scaled.index[idx]
                clustering_results['financial'][company] = {
                    'cluster': cluster_id,
                    'similar_companies': [df_scaled.index[i] for i in indices if df_scaled.index[i] != company]
                }

        # Clustering du risque (hierarchique)
        data_risk_scaled = TP_2.preprocess_risk_clustering("financial_ratios.csv")
        risk_clusters = TP_2.AgglomerativeClustering(n_clusters=3).fit_predict(data_risk_scaled)

        # On sauvegarde les assignations a chaque cluster
        for i, company in enumerate(data_risk_scaled.index):
            cluster_id = risk_clusters[i]
            similar_indices = [j for j in range(len(risk_clusters)) if risk_clusters[j] == cluster_id and j != i]
            clustering_results['risk'][company] = {
                'cluster': cluster_id,
                'similar_companies': [data_risk_scaled.index[j] for j in similar_indices]
            }

        # Correlation des rendements
        returns_df = TP_2.prepare_returns_data("Companies_historical_data")
        corr_matrix = TP_2.clust.correlation_matrix(returns_df)

        # Pour chaque entreprise, on cherche les plus correlees
        for i, company in enumerate(returns_df.columns):
            # Extraction de la correlation de l'entreprise aux autres
            corr_series = pd.Series(corr_matrix[i], index=returns_df.columns)
            # On ordonne et prend les 5 plus elevees
            top_correlated = corr_series.drop(company).nlargest(5)

            clustering_results['correlation'][company] = {
                'similar_companies': top_correlated.index.tolist()
            }

        logger.info("Clustering analysis completed")
    except Exception as e:
        logger.error(f"Error in clustering: {e}")

    return clustering_results


def run_classification():
    """Execute le tp_3 pour obtenir les signaux buy/hold/sell"""
    logger.info("Running classification models")

    classification_results = {}

    try:
        # Preparation des donnees - labels et indicateurs techniques
        labeled_dict = TP_3.apply_labeling_to_folder("Companies_historical_data")
        enriched_dict = TP_3.apply_technical_indicators_to_labeled_data(labeled_dict)

        # Entrainement du modele sur les donnees combinees
        X_train, X_test, y_train, y_test, feature_names = TP_3.prepare_dataset_for_classification(enriched_dict)

        # Definition des parametres du modele
        rf_params = {
            'n_estimators': [100, 200],
            'max_depth': [5, 10],
            'min_samples_split': [2, 5]
        }

        xgb_params = {
            'n_estimators': [100, 200],
            'max_depth': [3, 6],
            'learning_rate': [0.01, 0.1]
        }

        # Entrainement
        rf_model = TP_3.train_and_evaluate_model(
            RandomForestClassifier(random_state=167), rf_params,
            X_train, X_test, y_train, y_test
        )

        xgb_model = TP_3.train_and_evaluate_model(
            XGBClassifier(eval_metric='mlogloss'), xgb_params,
            X_train, X_test, y_train, y_test
        )

        models = {
            'RandomForest': rf_model,
            'XGBoost': xgb_model
        }

        # Prediction pour chaque entreprise
        for filename, data in enriched_dict.items():
            company_name = os.path.basename(filename).split('_')[0]

            # Preparation du dataset
            X_company = data.drop(columns=['label', 'Close Horizon', 'horizon return', 'Close'])

            # Standardisation
            scaler = StandardScaler()
            X_company_scaled = scaler.fit_transform(X_company)

            # Obtenir les predictions de chaque modele
            company_preds = {}
            for model_name, model in models.items():
                preds = model.predict(X_company_scaled)
                # Prediction la plus recente (0:sell, 1:hold, 2:buy)
                signal = preds[-1]
                company_preds[model_name] = int(signal)

            # Aggregation par vote
            signals = list(company_preds.values())
            majority_signal = max(set(signals), key=signals.count)

            classification_results[company_name] = {
                'model_predictions': company_preds,
                'majority_signal': majority_signal,
                'signal_text': {0: 'SELL', 1: 'HOLD', 2: 'BUY'}[majority_signal]
            }

        logger.info("Classification completed")
    except Exception as e:
        logger.error(f"Error in classification: {e}")

    return classification_results


def run_regression():
    """Execute le tp_4 pour predire les prix futurs"""
    logger.info("Running regression models")

    regression_results = {}

    try:
        # Obtenir les resultats des regressions pour chaque entreprise
        results_df, all_preds = TP_4.get_regression_results("Companies_historical_data")

        for company, data in all_preds.items():
            real_values = data.get('real', [])
            pred_values = {}

            for model_name in ['LinearRegression', 'RandomForest', 'KNN', 'XGBoost']:
                if model_name in data:
                    # Trouver la prediction la plus recente
                    pred_values[model_name] = data[model_name][-1]

            # Moyenner les predictions
            if pred_values:
                avg_prediction = sum(pred_values.values()) / len(pred_values)

                # Trouver le dernier prix close
                latest_close = real_values[-1] if real_values else None

                if latest_close:
                    expected_return = (avg_prediction - latest_close) / latest_close

                    regression_results[company] = {
                        'model_predictions': pred_values,
                        'average_prediction': avg_prediction,
                        'latest_close': latest_close,
                        'expected_return': expected_return,
                        'return_percentage': expected_return * 100
                    }

        logger.info("Regression completed")
    except Exception as e:
        logger.error(f"Error in regression: {e}")

    return regression_results


def run_deep_learning():
    """Execute le tp_5 pour les predictions des prix avec deep learning"""
    logger.info("Running deep learning models")

    dl_results = {}

    try:
        # On prend la liste des datasets
        file_list = sorted([f for f in os.listdir("datasets") if f.endswith("_x_train.npy")])
        companies = list(set([f.split("_")[0] for f in file_list]))

        for company in companies:
            # loader le dataset
            x_train, y_train, x_test, y_test, scaler = TP_5.load_dataset(company)

            # Entrainement des modeles
            dl_company_preds = {}
            for model_type in ["MLP", "RNN", "LSTM"]:
                model = TP_5.train_model(
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

                # Predire
                y_pred = model.predict(x_test)
                y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1))

                # Extraire la prediction la plus recente
                latest_pred = y_pred_inv[-1][0]
                dl_company_preds[model_type] = latest_pred

            # Aggregatation des predictions en moyennant
            avg_prediction = sum(dl_company_preds.values()) / len(dl_company_preds)

            # Trouver le dernier prix close et calculer le rendement attendu
            try:
                df = pd.read_csv(f"Companies_historical_data/{company}_historical_data.csv")
                latest_close = df['Close'].iloc[-1]
                expected_return = (avg_prediction - latest_close) / latest_close

                dl_results[company] = {
                    'model_predictions': dl_company_preds,
                    'average_prediction': avg_prediction,
                    'latest_close': latest_close,
                    'expected_return': expected_return,
                    'return_percentage': expected_return * 100
                }
            except Exception as e:
                logger.error(f"Error calculating dl return for {company}: {e}")

        logger.info("Deep learning completed")
    except Exception as e:
        logger.error(f"Error in deep learning: {e}")

    return dl_results


def run_news_analysis():
    """Execute les tp_6 et tp_7 pour l'analyse de sentiment des news"""
    logger.info("Running news sentiment analysis")

    news_results = {}

    try:
        # Mettre a jour les depeches disponibles pour chaque entreprise
        for company_name in TP_1.companies.keys():
            try:
                TP_6.get_news_by_date(company_name)
                logger.info(f"Updated news for {company_name}")
            except Exception as e:
                logger.error(f"Error updating news for {company_name}: {e}")

        # Entrainement du modele NLP
        dataset = TP_7.load_and_prepare_datasets()

        # On utilise finBERT pour l'analyse de sentiment sur les news
        finbert_model, finbert_tokenizer, _, _ = TP_7.train_model(
            "ProsusAI/finbert", dataset, batch_size=8, num_epochs=10
        )

        # On recupere la date du jour pour filter
        today_str = datetime.now().strftime('%Y-%m-%d')

        # On process les news pour chaque entreprise
        for company_name in TP_1.companies.keys():
            company_filename = f"Data/{company_name.replace(' ', '_').replace('&', 'and')}_news.json"

            if os.path.exists(company_filename):
                with open(company_filename, 'r', encoding='utf-8') as f:
                    news_data = json.load(f)

                # On prend les news les plus recentes
                today_news = news_data.get(today_str, [])
                if not today_news:
                    dates = sorted(news_data.keys())
                    if dates:
                        today_news = news_data[dates[-1]]

                # On cherche des sentiments dans les depeches
                sentiments = []
                titles = []
                for item in today_news:
                    title = item.get("title", "")
                    description = item.get("description", "")
                    full_text = f"{title}. {description}" if description else title

                    if full_text.strip():
                        titles.append(title)

                        # Analyse (0: negatif, 1: neutre, 2: positif)
                        inputs = finbert_tokenizer(
                            [full_text],
                            truncation=True,
                            padding="max_length",
                            max_length=128,
                            return_tensors="pt"
                        )
                        with torch.no_grad():
                            outputs = finbert_model(**inputs)
                            sentiment_id = torch.argmax(outputs.logits, dim=-1).item()
                            sentiments.append(sentiment_id)

                # On determine le sentiment dominant
                if sentiments:
                    sentiment_counts = {
                        0: sentiments.count(0),  # negative
                        1: sentiments.count(1),  # neutral
                        2: sentiments.count(2)  # positive
                    }

                    dominant_sentiment = max(sentiment_counts, key=sentiment_counts.get)
                    sentiment_text = {0: "negative", 1: "neutral", 2: "positive"}[dominant_sentiment]

                    # Calcul du score de sentiment (de -1 a 1)
                    total = len(sentiments)
                    sentiment_score = (sentiment_counts[2] - sentiment_counts[0]) / total if total > 0 else 0

                    news_results[company_name] = {
                        'titles': titles,
                        'sentiment_counts': sentiment_counts,
                        'dominant_sentiment': dominant_sentiment,
                        'sentiment_text': sentiment_text,
                        'sentiment_score': sentiment_score
                    }
                else:
                    # Si pas de news, on met neutre
                    news_results[company_name] = {
                        'titles': [],
                        'sentiment_counts': {0: 0, 1: 0, 2: 0},
                        'dominant_sentiment': 1,
                        'sentiment_text': "neutral",
                        'sentiment_score': 0
                    }
            else:
                # Si aucun fichier news
                news_results[company_name] = {
                    'titles': [],
                    'sentiment_counts': {0: 0, 1: 0, 2: 0},
                    'dominant_sentiment': 1,
                    'sentiment_text': "neutral",
                    'sentiment_score': 0
                }

        logger.info("News analysis completed")
    except Exception as e:
        logger.error(f"Error in news analysis: {e}")

    return news_results


def run_sentiment_visualization():
    """Execute le TP8 pour visualiser l'analyse des sentiments et l'effet sur les prix des stocks"""
    logger.info("Running sentiment visualization analysis")

    visualization_results = {}

    try:
        # Selection des entreprises avec le plus de donnees
        news_files = glob.glob("Data/*_news.json")
        company_data = []

        for file in news_files:
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    news_data = json.load(f)

                news_count = sum(len(articles) for articles in news_data.values())
                company_name = os.path.basename(file).replace("_news.json", "").replace("_", " ")
                company_data.append((company_name, news_count))
            except Exception as e:
                logger.error(f"Error processing news file {file}: {e}")

        # On tri les entreprises par le nombre de depeches et on prend le top 5
        top_companies = [company for company, count in sorted(company_data, key=lambda x: x[1], reverse=True)[:5]]

        if top_companies:
            logger.info(f"Running sentiment visualization for top companies: {', '.join(top_companies)}")

            # Chemin vers les modeles
            base_model = "ProsusAI/finbert"  # FinBERT de base

            # On check si une version fine-tunee existe sinon on prend le modele de base
            finetuned_model = "finbert_finetuned"
            if not os.path.exists(finetuned_model):
                logger.warning(f"Fine-tuned model not found at {finetuned_model}, using base model instead")
                finetuned_model = "ProsusAI/finbert"

            # Execution de tout le TP 8
            visualization_results = TP_8.run_sentiment_analysis(
                top_companies,
                base_model,
                finetuned_model
            )

            logger.info("Sentiment visualization analysis completed")
        else:
            logger.warning("No companies with news data found")
    except Exception as e:
        logger.error(f"Error in sentiment visualization: {e}")

    return visualization_results


def aggregate_signals(clustering_results, classification_results, regression_results, dl_results, news_results, visualization_results=None):
    """Combine tous les signaux en une recommendation finale"""
    logger.info("Aggregating signals for investment recommendations")

    recommendations = {}

    for company_name in TP_1.companies.keys():
        try:
            # 1. Classification (0:sell, 1:hold, 2:buy)
            classification_signal = classification_results.get(company_name, {}).get('majority_signal', 1)

            # 2. Rendement attendu de la regression et du deep learning
            regression_return = regression_results.get(company_name, {}).get('expected_return', 0)
            dl_return = dl_results.get(company_name, {}).get('expected_return', 0)

            # On combine la prediction de rendement (en mettant un peu plus de poids au deep learning)
            if dl_return:
                expected_return = regression_return * 0.4 + dl_return * 0.6
            else:
                expected_return = regression_return

            # 3. Score de sentiment (entre -1 et 1)
            sentiment_score = news_results.get(company_name, {}).get('sentiment_score', 0)

            # 4. Si disponible, on inclut l'analyse de sentiment plus poussee du TP8
            enhanced_sentiment_score = 0
            if visualization_results and company_name in visualization_results:
                result = visualization_results[company_name]
                if result['status'] == 'success':
                    # Utilisation du score du modele fine-tune si disponible
                    enhanced_sentiment_score = result['model_b']['sentiment_score']

                    # Ajout au score de sentiment existant
                    if sentiment_score != 0:
                        # On donne plus de poids a ce score
                        sentiment_score = (sentiment_score * 0.3) + (enhanced_sentiment_score * 0.7)
                    else:
                        sentiment_score = enhanced_sentiment_score

            # Conversion de ka classification en echelle -1 a 1
            class_score = (classification_signal - 1)  # 0->-1, 1->0, 2->1

            # On cap les rendements attendus a 10% pour la mise a l'echelle
            return_score = max(min(expected_return * 10, 1), -1)

            # Moyenne ponderee des scores
            final_score = (
                    class_score * 0.4 +  # classification (40%)
                    return_score * 0.4 +  # prediction du prix (40%)
                    sentiment_score * 0.2  # analyse des sentiment (20%)
            )

            # Conversion en recommendation
            if final_score > 0.3:
                recommendation = "BUY"
            elif final_score < -0.3:
                recommendation = "SELL"
            else:
                recommendation = "HOLD"

            # Trouve des entreprises similaires par clustering
            similar_companies = []

            # D'abord clustering financier, puis en risque puis en correlation
            if company_name in clustering_results.get('financial', {}):
                similar_companies = clustering_results['financial'][company_name].get('similar_companies', [])
            elif company_name in clustering_results.get('risk', {}):
                similar_companies = clustering_results['risk'][company_name].get('similar_companies', [])
            elif company_name in clustering_results.get('correlation', {}):
                similar_companies = clustering_results['correlation'][company_name].get('similar_companies', [])

            # Trouver les titres des news
            news_titles = news_results.get(company_name, {}).get('titles', [])

            # Enregistrement resultats
            recommendations[company_name] = {
                'recommendation': recommendation,
                'final_score': final_score,
                'expected_return': expected_return * 100,  # en pourcentage
                'classification_signal': {0: 'SELL', 1: 'HOLD', 2: 'BUY'}[classification_signal],
                'sentiment_analysis': news_results.get(company_name, {}).get('sentiment_text', 'neutral'),
                'similar_companies': similar_companies[:5],  # top 5
                'news_titles': news_titles[:5],  # top 5
                'signals': {
                    'classification': classification_signal,
                    'return_prediction': expected_return,
                    'sentiment': sentiment_score,
                    'enhanced_sentiment': enhanced_sentiment_score if visualization_results and company_name in visualization_results else None
                }
            }
        except Exception as e:
            logger.error(f"Error aggregating signals for {company_name}: {e}")
            recommendations[company_name] = {
                'recommendation': 'HOLD',  # Si erreur, on met "hold" par défault
                'error': str(e)
            }

    return recommendations


def save_recommendations(recommendations):
    """Sauvegarde les recommendations en un ficher output"""
    logger.info("saving recommendations")

    # output json
    output_file = f"outputs/recommendations_{datetime.now().strftime('%Y%m%d')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(recommendations, f, indent=4, ensure_ascii=False)

    # Fichier texte pour etre lu par un humain
    text_file = f"outputs/recommendations_{datetime.now().strftime('%Y%m%d')}.txt"
    with open(text_file, 'w', encoding='utf-8') as f:
        f.write(f"Investment recommendations - {datetime.now().strftime('%Y-%m-%d')}\n")
        f.write("=" * 80 + "\n\n")

        for company, data in sorted(recommendations.items()):
            f.write(f"Company: {company}\n")
            f.write(f"Recommendation: {data['recommendation']}\n")

            if 'expected_return' in data:
                f.write(f"expected return: {data['expected_return']:.2f}%\n")

            if 'sentiment_analysis' in data:
                f.write(f"news sentiment: {data['sentiment_analysis']}\n")

            if 'similar_companies' in data and data['similar_companies']:
                f.write(f"similar companies: {', '.join(data['similar_companies'])}\n")

            if 'news_titles' in data and data['news_titles']:
                f.write("recent news:\n")
                for title in data['news_titles']:
                    f.write(f"  - {title}\n")

            f.write("\n" + "-" * 80 + "\n\n")

    logger.info(f"Recommendations saved to {output_file} and {text_file}")


def main():
    """Fonction main pour tout orchestrer"""
    logger.info("Starting financial analysis pipeline")

    # TP1 : mise a jour des donnees histo
    update_data()

    # TP2 : clustering pour determiner des groupes d'entreprises
    clustering_results = run_clustering()

    # TP3 : modele de classification pour les signaux buy/hold/sell
    classification_results = run_classification()

    # TP4 : modeles de regression pour la prediction des prix
    regression_results = run_regression()

    # TP5: deep learning pour la prediction des prix
    dl_results = run_deep_learning()

    # TP6 & TP7 : chercher et analyser les depeches pour chaque entreprise
    news_results = run_news_analysis()

    # TP8 : visualiser l'impact des news sur les variations de prix de stocks
    visualization_results = run_sentiment_visualization()

    # Aggregation des resultats et generation des recommendations
    recommendations = aggregate_signals(
        clustering_results,
        classification_results,
        regression_results,
        dl_results,
        news_results,
        visualization_results
    )

    # Sauvegarde des resultats
    save_recommendations(recommendations)

    logger.info("Pipeline completed successfully")


if __name__ == "__main__":
    main()
