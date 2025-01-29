import yfinance as yf
import pandas as pd
import os

class FinancialDataFetcher:
    def __init__(self, period='max', interval='1d', force_recreate=False):
        self.symbols = {
            'Actions': [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',  # NYSE/NASDAQ
                '7203.T', '6758.T',  # Tokyo Stock Exchange
                '600519.SS', '000001.SS',  # Shanghai Stock Exchange
                '0005.HK', '0939.HK',  # Hong Kong Stock Exchange
                'AI.PA', 'BNP.PA',  # Euronext
                'BARC.L', 'VOD.L',  # London Stock Exchange
                '000001.SZ', '000063.SZ'  # Shenzhen Stock Exchange
            ],
            'Cryptomonnaies': [
                'BTC-USD', 'ETH-USD', 'BNB-USD', 'USDT-USD', 'SOL-USD',
                'ADA-USD', 'XRP-USD', 'DOT-USD', 'DOGE-USD', 'AVAX-USD'
            ],
            # 'Matières premières': [
            #     'GC=F', 'SI=F', 'CL=F', 'NG=F', 'ZC=F',  # Or, Argent, Pétrole, Gaz naturel, Maïs
            #     'ZW=F', 'ZS=F', 'ZL=F', 'HG=F'  # Blé, Soja, Huile de soja, Cuivre
            # ],
            'Monnaies': [
                'EURUSD=X', 'JPYUSD=X', 'GBPUSD=X', 'CHFUSD=X', 'CNYUSD=X', 'HKDUSD=X'
            ]
        }
        self.period = period
        self.interval = interval
        self.force_recreate = force_recreate
        self.data_dir = './data/dataset/raw'
        os.makedirs(self.data_dir, exist_ok=True)

    def get_historical_data(self, symbol):
        stock = yf.Ticker(symbol)
        hist = stock.history(period=self.period, interval=self.interval)
        hist['Volume'] = hist['Volume']
        hist['Close'] = hist['Close']
        hist['Market Cap'] = stock.info.get('marketCap', None)
        hist['Shares Outstanding'] = stock.info.get('sharesOutstanding', None)
        hist['Currency'] = stock.info.get('currency', 'USD')
        return hist

    def fetch_data(self):
        data = {}
        for category, syms in self.symbols.items():
            data[category] = {}
            for sym in syms:
                filename = os.path.join(self.data_dir, f"{category}_{sym}_{self.period}_{self.interval}.csv")
                if os.path.exists(filename) and not self.force_recreate:
                    print(f"Chargement des données existantes pour {sym}...")
                    data[category][sym] = pd.read_csv(filename, index_col=0, parse_dates=True)
                else:
                    print(f"Récupération des données pour {sym}...")
                    data[category][sym] = self.get_historical_data(sym)
                    data[category][sym].to_csv(filename)
        return data