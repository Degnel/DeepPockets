import torch
import pandas as pd
import numpy as np
from data.data_gathering import FinancialDataFetcher


GLOBAL_MONEY_SUPPLY = 255*10e12

class FinancialDataPreprocessor:
    def __init__(self, period='1y', interval='1h', force_recreate=True):
        self.fetcher = FinancialDataFetcher(period=period, interval=interval, force_recreate=force_recreate)

    def fetch_and_preprocess_data(self, indicators, to_usd=True):
        raw_data = self.fetcher.fetch_data()
        if to_usd:
            raw_data = self.convert_to_usd(raw_data)
        preprocessed_data = self.preprocess_data(raw_data, indicators)
        return preprocessed_data

    def convert_to_usd(self, raw_data):
        exchange_rates = raw_data.get('Monnaies', {})
        for sym, data in raw_data['Actions'].items():
            currency = data['Currency'].iloc[0]
            if currency + "USD=X" in exchange_rates:
                exchange_rate = exchange_rates[currency]['Close']
                raw_data['Actions'][sym]['Close'] = data['Close'] / exchange_rate
        return raw_data

    def preprocess_data(self, raw_data, indicators):
        all_data = []
        combined_data = {**raw_data['Actions'], **raw_data['Cryptomonnaies']}
        for _, data in combined_data.items():
            indicator_data = []
            for indicator in indicators:
                if indicator == 'prices':
                    indicator_data.append(data[['Close']].values)
                elif indicator == 'log_prices':
                    indicator_data.append(np.log(data[['Close']].values))
                elif indicator == 'volumes':
                    indicator_data.append(data[['Volume']].values)
                elif indicator == 'normalized_volumes':
                    indicator_data.append(self.normalize_volumes(data[['Volume']].values))
                elif indicator == 'delta':
                    indicator_data.append(data[['Close']].diff().dropna().values)
                elif indicator == 'ratio':
                    indicator_data.append(data[['Close']].pct_change().dropna().values)
                elif indicator == 'proportions':
                    indicator_data.append(self.calculate_proportions(data[['Close']].values))
                elif indicator == 'sigmoid_proportions':
                    indicator_data.append(self.sigmoid(self.calculate_proportions(data[['Close']].values)))
                else:
                    raise ValueError(f"Indicator '{indicator}' is not supported.")

            # Concatenate all indicator data for the current symbol
            combined_data = np.concatenate(indicator_data, axis=1)
            all_data.append(combined_data)

        # Concatenate all data for all symbols
        all_data = np.concatenate(all_data, axis=1)
        return torch.tensor(all_data, dtype=torch.float32)

    def normalize_volumes(self, volumes):
        return (volumes - volumes.mean()) / volumes.std()

    def calculate_proportions(self, data):
        share_count = data[['Shares Outstanding']] = data[['Market Cap']].fillna(0)
        if not share_count:
            share_count = data[['Market Cap']] / data[['Close']].values[-1]
        return share_count * data[['Close']].values / GLOBAL_MONEY_SUPPLY

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

# Example usage
preprocessor = FinancialDataPreprocessor(period='1y', interval='1h')
indicators = ['prices', 'log_prices', 'volumes']
preprocessed_data = preprocessor.fetch_and_preprocess_data(indicators)