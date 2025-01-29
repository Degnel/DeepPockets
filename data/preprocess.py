from data.data_gathering import FinancialDataFetcher

# On fait une méthode qui va directement chercher les données
# et qui les renvoies sous la forme d'un tenseur torch

fetcher = FinancialDataFetcher(period='1y', interval='1h', force_recreate=True)
raw_data = fetcher.fetch_data()