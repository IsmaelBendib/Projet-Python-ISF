import json
import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import yfinance as yf
from datetime import datetime, timedelta
import pytz
import glob


def get_texts_timestamps(news_data):
    """
    Extract texts and timestamps from news JSON data.

    Parameters:
    -----------
    news_data : dict
        JSON data containing news articles with dates as keys

    Returns:
    --------
    tuple
        (news_texts, news_timestamps) - lists of texts and corresponding timestamps
    """
    news_texts = []
    news_timestamps = []

    # Definition de la timezone pour la conversion
    nyc_tz = pytz.timezone('America/New_York')

    # On process chaque date au format json
    for date, articles in news_data.items():
        for article in articles:
            # Extraction du titre et de la description
            title = article.get('title', '')
            description = article.get('description', '')

            # Creation du texte entier par concatenation titre + description
            full_text = f"{title}. {description}" if description else title

            if full_text.strip():  # On prend que les textes non-vides
                # On prend le timestamp et le converti au fuseau NYC
                timestamp_str = article.get('publishedAt')
                if timestamp_str:
                    # Format ISO
                    utc_timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    # Conversion
                    nyc_timestamp = utc_timestamp.astimezone(nyc_tz)
                    # On arrondit a l'heure la plus proche
                    hour_timestamp = nyc_timestamp.replace(minute=0, second=0, microsecond=0)

                    news_texts.append(full_text)
                    news_timestamps.append(hour_timestamp)

    return news_texts, news_timestamps


def get_sentiments(model_path, texts, batch_size=32):
    """
    Analyze sentiment of news texts using a fine-tuned BERT model.

    Parameters:
    -----------
    model_path : str
        Path to the fine-tuned model
    texts : list
        List of news texts to analyze
    batch_size : int
        Batch size for processing

    Returns:
    --------
    list
        List of sentiment predictions (0: negative, 1: neutral, 2: positive)
    """
    # Chargement du modele et du tokenizer
    tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")

    # Check si model_path est un chemin local path ou un modele de Hugging Face
    if os.path.exists(model_path) or model_path.startswith('./') or model_path.startswith('../'):
        # Si c'est local
        try:
            model = BertForSequenceClassification.from_pretrained(model_path, local_files_only=True)
        except Exception as e:
            print(f"Error loading local model: {e}")
            # Si echec on utilise FinBERT
            model = BertForSequenceClassification.from_pretrained("ProsusAI/finbert")
    else:
        # Modele Hugging Face
        model = BertForSequenceClassification.from_pretrained(model_path)

    model.eval()

    sentiments = []

    # Traitement des batchs
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        # Tokenization
        inputs = tokenizer(batch_texts, truncation=True, padding="max_length",
                           max_length=128, return_tensors="pt")

        # Predictions
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1).tolist()
            sentiments.extend(predictions)

    return sentiments


def align_timestamps(timestamps):
    """
    Align news timestamps with market hours.

    Parameters:
    -----------
    timestamps : list
        List of datetime objects in NYC timezone

    Returns:
    --------
    list
        List of aligned timestamps
    """
    aligned_timestamps = []
    nyc_tz = pytz.timezone('America/New_York')

    for ts in timestamps:
        # On s'assure que le timestamp a conscience du fuseau horaire
        if ts.tzinfo is None:
            ts = nyc_tz.localize(ts)

        # Extraction de l'heure et de la minute
        hour = ts.hour
        minute = ts.minute

        # Ouverture du marche a 9:30, fermeture a 16:00
        if (hour > 9 or (hour == 9 and minute >= 30)) and hour < 16:
            # Si c'est tombe pendant les heures de marche on modifie pas l'horaire
            aligned_timestamps.append(ts)
        elif 16 <= hour < 24:
            # Apres le close mais le meme jour, on assigne au close : 16:00
            aligned_timestamps.append(ts.replace(hour=16, minute=0, second=0, microsecond=0))
        else:
            # Si c'est tombe juste avant l'open, on assigne au close du jour precedent
            previous_day = ts - timedelta(days=1)
            aligned_timestamps.append(previous_day.replace(hour=16, minute=0, second=0, microsecond=0))

    return aligned_timestamps


def plot_comparison(df, sentiments_a, sentiments_b, timestamps, title_a, title_b, company_name):
    """
    Plot comparison between two sentiment models and stock price.

    Parameters:
    -----------
    df : DataFrame
        DataFrame with stock price data (columns: Datetime, Close)
    sentiments_a, sentiments_b : list
        Lists of sentiment predictions from two models
    timestamps : list
        List of news timestamps
    title_a, title_b : str
        Titles for the two subplots
    company_name : str
        Name of the company for the plot title
    """
    # Alignement des timestamps avec les heures de marche
    aligned_timestamps = align_timestamps(timestamps)

    # Creation de la figure avec 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"News Sentiment vs Stock Price - {company_name}", fontsize=16)

    # On plot les prix sur les 2
    ax1.plot(df['Datetime'], df['Close'], 'k-', label='Stock Price')
    ax2.plot(df['Datetime'], df['Close'], 'k-', label='Stock Price')

    # Definition des couleurs pour les segments
    colors = {0: 'red', 1: 'gold', 2: 'green'}
    labels = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

    # On regroupe les sentiments par timestamp
    timestamp_sentiments_a = {}
    timestamp_sentiments_b = {}

    for ts, sent_a, sent_b in zip(aligned_timestamps, sentiments_a, sentiments_b):
        if ts not in timestamp_sentiments_a:
            timestamp_sentiments_a[ts] = []
            timestamp_sentiments_b[ts] = []
        timestamp_sentiments_a[ts].append(sent_a)
        timestamp_sentiments_b[ts].append(sent_b)

    # Tri des timestamps pour le plot
    sorted_timestamps = sorted(timestamp_sentiments_a.keys())

    # Plot sentiments
    for i, ts in enumerate(sorted_timestamps):
        # On cherche le prix le plus proche de l'heure de la news
        closest_time_idx = (df['Datetime'] - ts).abs().argmin()
        price_at_time = df.iloc[closest_time_idx]['Close'] if closest_time_idx is not None else None

        if price_at_time is not None:
            # Plot des sentiments pour le modele A
            for j, sent in enumerate(timestamp_sentiments_a[ts]):
                offset = j * 0.1
                ax1.scatter(ts, price_at_time + offset, color=colors[sent], s=100, zorder=5)

            # Pour le modele B
            for j, sent in enumerate(timestamp_sentiments_b[ts]):
                offset = j * 0.1
                ax2.scatter(ts, price_at_time + offset, color=colors[sent], s=100, zorder=5)

    # Ajout de la legnde
    for sent, color in colors.items():
        ax1.scatter([], [], color=color, s=100, label=labels[sent])
        ax2.scatter([], [], color=color, s=100, label=labels[sent])

    # Configuration des axes
    ax1.set_title(title_a)
    ax2.set_title(title_b)

    # L'axe x doit afficher le temps (a la minute pres)
    date_format = DateFormatter('%Y-%m-%d %H:%M')
    ax2.xaxis.set_major_formatter(date_format)
    plt.gcf().autofmt_xdate()

    # Legendes
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper left')

    # Grilles
    ax1.grid(True, alpha=0.3)
    ax2.grid(True, alpha=0.3)

    # Label Stock Price pour l'axe y
    ax1.set_ylabel('Stock Price')
    ax2.set_ylabel('Stock Price')
    ax2.set_xlabel('Date/Time')

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    # Sauvegarde de la figure
    save_dir = "outputs/sentiment_analysis"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/{company_name}_sentiment_comparison.png", dpi=300, bbox_inches='tight')

    return fig


def analyze_company_sentiment(company_name, model_path_a, model_path_b, title_a="Base FinBERT",
                              title_b="Fine-tuned FinBERT"):
    """
    Perform complete sentiment analysis for a company.

    Parameters:
    -----------
    company_name : str
        Name of the company
    model_path_a, model_path_b : str
        Paths to the two models to compare
    title_a, title_b : str
        Titles for the two models in the plot

    Returns:
    --------
    dict
        Results including sentiments and visualization path
    """
    results = {
        'company': company_name,
        'status': 'failed',
        'error': None
    }

    try:
        # 1. Chargement des depeches de presse
        news_file = f"Data/{company_name.replace(' ', '_').replace('&', 'and')}_news.json"
        if not os.path.exists(news_file):
            results['error'] = f"News file not found: {news_file}"
            return results

        with open(news_file, 'r', encoding='utf-8') as f:
            news_data = json.load(f)

        # 2. Extraction des textes et timestamps
        news_texts, news_timestamps = get_texts_timestamps(news_data)

        if not news_texts:
            results['error'] = "No valid news texts found"
            return results

        # 3. On identifie les sentiments avec les 2 modeles
        sentiments_a = get_sentiments(model_path_a, news_texts)
        sentiments_b = get_sentiments(model_path_b, news_texts)

        # 4. Prix des stocks a la resolution horaire
        symbol = None
        # Mapping du nom de l'entreprise a son symbole
        from TP_1 import companies
        for comp, symb in companies.items():
            if comp.lower() == company_name.lower():
                symbol = symb
                break

        if not symbol:
            results['error'] = f"Symbol not found for company: {company_name}"
            return results

        # On cherche la range de dates pour les news
        if news_timestamps:
            min_date = min(news_timestamps).strftime('%Y-%m-%d')
            max_date = (max(news_timestamps) + timedelta(days=1)).strftime('%Y-%m-%d')
        else:
            min_date = "2025-01-01"
            max_date = datetime.now().strftime('%Y-%m-%d')

        # On recupere l'historique de prix
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=min_date, end=max_date, interval="60m")
        df = df.reset_index()

        # On s'assure que le dataframe a les bonnes colonnes
        if 'Datetime' not in df.columns and 'Date' in df.columns:
            df = df.rename(columns={'Date': 'Datetime'})

        # 5. Plot la comparaison entre les 2 modelese
        fig = plot_comparison(df, sentiments_a, sentiments_b, news_timestamps,
                              title_a, title_b, company_name)

        # 6. Calcule les stats de sentiments pour les 2 modeles
        sentiment_counts_a = {0: sentiments_a.count(0), 1: sentiments_a.count(1), 2: sentiments_a.count(2)}
        sentiment_counts_b = {0: sentiments_b.count(0), 1: sentiments_b.count(1), 2: sentiments_b.count(2)}

        # Calcul du sentiment dominant et du score
        total_a = len(sentiments_a)
        sentiment_score_a = (sentiment_counts_a[2] - sentiment_counts_a[0]) / total_a if total_a > 0 else 0
        dominant_sentiment_a = max(sentiment_counts_a, key=sentiment_counts_a.get)

        total_b = len(sentiments_b)
        sentiment_score_b = (sentiment_counts_b[2] - sentiment_counts_b[0]) / total_b if total_b > 0 else 0
        dominant_sentiment_b = max(sentiment_counts_b, key=sentiment_counts_b.get)

        # 7. Sauvegarde les resultats
        results.update({
            'status': 'success',
            'model_a': {
                'name': title_a,
                'sentiment_counts': sentiment_counts_a,
                'dominant_sentiment': dominant_sentiment_a,
                'sentiment_score': sentiment_score_a
            },
            'model_b': {
                'name': title_b,
                'sentiment_counts': sentiment_counts_b,
                'dominant_sentiment': dominant_sentiment_b,
                'sentiment_score': sentiment_score_b
            },
            'news_count': len(news_texts),
            'visualization_path': f"outputs/sentiment_analysis/{company_name}_sentiment_comparison.png"
        })

    except Exception as e:
        results['error'] = str(e)

    return results


def run_sentiment_analysis(company_list=None, base_model="ProsusAI/finbert", finetuned_model="./finbert_finetuned"):
    """
    Run sentiment analysis for multiple companies.

    Parameters:
    -----------
    company_list : list, optional
        List of companies to analyze. If None, analyze all companies with news data
    base_model : str
        Path or name of the base model
    finetuned_model : str
        Path to the fine-tuned model

    Returns:
    --------
    dict
        Results for all companies
    """
    results = {}

    # Si pas de nom d'entreprise fourni, on fait l'analyse sur toutes les entreprises qui ont des fichiers news
    if company_list is None:
        # Cherche tous les fichiers de news
        news_files = glob.glob("Data/*_news.json")
        company_list = [os.path.basename(f).replace("_news.json", "").replace("_", " ") for f in news_files]

    # Analyse pour chaque entreprise
    for company in company_list:
        print(f"Analyzing sentiment for: {company}")
        company_results = analyze_company_sentiment(
            company,
            base_model,
            finetuned_model,
            "Base FinBERT",
            "Fine-tuned FinBERT"
        )
        results[company] = company_results

    return results


if __name__ == "__main__":
    # Exemple d'utilisation
    top_companies = ["Microsoft", "Apple", "Tesla", "Amazon"]
    results = run_sentiment_analysis(top_companies)

    # Affichage des resultats
    for company, result in results.items():
        print(f"\n{company}:")
        if result['status'] == 'success':
            print(f"  News count: {result['news_count']}")
            print(f"  Base model dominant sentiment: {result['model_a']['dominant_sentiment']}")
            print(f"  Fine-tuned model dominant sentiment: {result['model_b']['dominant_sentiment']}")
        else:
            print(f"  Error: {result['error']}")
