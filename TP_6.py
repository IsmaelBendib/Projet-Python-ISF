import requests
import json
import os
from datetime import datetime, timedelta
# Créer le dossier 'Data' s'il n'existe pas
data_folder = "Data"
os.makedirs(data_folder, exist_ok=True)


# Charger les anciennes actualités si elles existent
def load_existing_news(company_filename):
    if os.path.exists(company_filename):
        with open(company_filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

# Sauvegarder les actualités dans un fichier JSON
def save_news_dict(company_filename, news_dict):
    with open(company_filename, 'w', encoding='utf-8') as f:
        json.dump(news_dict, f, indent=4, ensure_ascii=False)

# Fonction principale : scrapping + mise à jour par date
def get_news_by_date(company_name):
    url = 'https://newsapi.org/v2/everything'
    last_day = datetime.today().strftime('%Y-%m-%d')
    first_day = (datetime.today() - timedelta(days=10)).strftime('%Y-%m-%d')

    api_key = "df14f42316e541fb8c7605245cbbad61"  # Remplacez par votre propre clé

    params = {
        "q": company_name,
        "apiKey": api_key,
        "language": "en",
        "pageSize": 100,
        "from": first_day,
        "to": last_day
    }

    response = requests.get(url, params=params)
    if response.status_code != 200:
        print(f"Erreur {response.status_code} pour {company_name} : {response.text}")
        return

    articles = response.json().get("articles", [])
    print(f"🔎 {len(articles)} articles trouvés pour {company_name}.")

    # Format du fichier
    filename = os.path.join(data_folder, f"{company_name.replace(' ', '_').replace('&', 'and')}_news.json")


    # Charger les anciens articles si le fichier existe
    news_dict = load_existing_news(filename)

    # Parcourir les nouveaux articles
    for article in articles:
        title = article.get("title", "")
        description = article.get("description", "")
        source = article.get("source", {}).get("name", "")
        published_at = article.get("publishedAt", "")

        # Vérification de la mention de l'entreprise
        if company_name.lower() not in (title or "").lower() and company_name.lower() not in (description or "").lower():
            continue

        # Extraire la date (YYYY-MM-DD)
        date = published_at.split("T")[0]

        # Initialiser la date dans le dictionnaire si nécessaire
        if date not in news_dict:
            news_dict[date] = []

        # Éviter les doublons de titre
        if not any(existing["title"] == title for existing in news_dict[date]):
            news_dict[date].append({
                "title": title,
                "description": description,
                "source": source,
                "url": article.get("url"),
                "publishedAt": published_at,
            })

    # Sauvegarde finale
    save_news_dict(filename, news_dict)
    print(f"Fichier mis à jour : {filename}")

# Liste complète des entreprises (issues de ton image)
companies = [
    "Adobe", "Alibaba", "Alphabet", "Amazon", "AMD", "Apple", "ASML", "Baidu", "BYD", "Cisco",
    "ExxonMobil", "Goldman_Sachs", "Hyundai", "IBM", "ICBC", "Intel", "JD.com", "Johnson_&_Johnson",
    "JP_Morgan", "Louis_Vuitton_LVMH", "Meta", "Microsoft", "Netflix", "Nintendo", "NVIDIA",
    "Oracle", "Pfizer", "Qualcomm", "Reliance_Industries", "Samsung", "SAP", "Shell", "Siemens",
    "SoftBank", "Sony", "Tata_Consultancy_Services", "Tencent", "Tesla", "TotalEnergies", "Toyota", "Visa"
]

# Lancer le scrapping pour toutes les entreprises
if __name__ == "__main__":
    for company in companies:
        get_news_by_date(company)
