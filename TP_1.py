# TP1

import yfinance as yf
import pandas as pd
import os


# Dates d'études
start_date = '2019-03-14'
end_date = '2024-03-14'

output_dir = 'Companies_historical_data'
os.makedirs(output_dir, exist_ok=True)

# Dictionnaire des entreprises et leurs symboles boursiers
companies = {
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Amazon": "AMZN",
    "Alphabet": "GOOGL",
    "Meta": "META",
    "Tesla": "TSLA",
    "NVIDIA": "NVDA",
    "Samsung": "005930.KS",
    "Tencent": "TCEHY",
    "Alibaba": "BABA",
    "IBM": "IBM",
    "Intel": "INTC",
    "Oracle": "ORCL",
    "Sony": "SONY",
    "Adobe": "ADBE",
    "Netflix": "NFLX",
    "AMD": "AMD",
    "Qualcomm": "QCOM",
    "Cisco": "CSCO",
    "JP Morgan": "JPM",
    "Goldman Sachs": "GS",
    "Visa": "V",
    "Johnson & Johnson": "JNJ",
    "Pfizer": "PFE",
    "ExxonMobil": "XOM",
    "ASML": "ASML.AS",
    "SAP": "SAP.DE",
    "Siemens": "SIE.DE",
    "Louis Vuitton (LVMH)": "MC.PA",
    "TotalEnergies": "TTE.PA",
    "Shell": "SHEL.L",
    "Baidu": "BIDU",
    "JD.com": "JD",
    "BYD": "BYDDY",
    "ICBC": "1398.HK",
    "Toyota": "TM",
    "SoftBank": "9984.T",
    "Nintendo": "NTDOY",
    "Hyundai": "HYMTF",
    "Reliance Industries": "RELIANCE.NS",
    "Tata Consultancy Services": "TCS.NS"
}

ratios_keys = [
    "forwardPE", "beta", "priceToBook", "priceToSales", "dividendYield", "trailingEps",
    "debtToEquity", "currentRatio", "quickRatio", "returnOnEquity", "returnOnAssets",
    "operatingMargins", "profitMargins"
]

ratios_data = {company: {ratio: None for ratio in ratios_keys} for company in companies.keys()}

# Récupération des données
for company, symbol in companies.items():
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        for ratio in ratios_keys:
            ratios_data[company][ratio] = info.get(ratio, None)
    except Exception as e:
        print(f"Erreur pour {company} ({symbol}): {e}")

df = pd.DataFrame.from_dict(ratios_data, orient='index')

df.to_csv("financial_ratios.csv")

print("Fichier financial_ratios.csv généré avec succès.")


def get_stock_variations(symbol, start_date, end_date):
    try:
        company_data = yf.download(symbol, start=start_date, end=end_date)
        company_data.columns = company_data.columns.get_level_values(0) #enlève la 2eme indexation
        df = company_data[['Close']].copy()
        df['Next Day Close'] = df['Close'].shift(-1)
        df['Rendement'] = (df['Next Day Close'] - df['Close']) / df['Close']
        df = df.dropna(subset=['Rendement'])
        company_name = [name for name, ticker in companies.items() if ticker == symbol][0]
        df.to_csv(f'{output_dir}/{company_name}_historical_data.csv')
        print(f"Les données pour {company_name} ({symbol}) ont été exportées.")
    except Exception as e:
        print(f"Erreur pour {symbol}: {e}")

for company, symbol in companies.items():
    get_stock_variations(symbol, start_date, end_date)

print("Tous les fichiers CSV ont été générés avec succès.")
