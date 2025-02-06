import collections
import collections.abc
if not hasattr(collections, 'Iterable'):
    collections.Iterable = collections.abc.Iterable

import logging
import pandas as pd
import numpy as np
from datetime import datetime
import backtrader as bt
from ib_insync import IB, Stock, util

# Imposta PlaidML per usare la GPU AMD
import plaidml.keras
plaidml.keras.install_backend()

from keras.models import load_model
import openai

from src.indicators import fetch_real_data, clean_data, compute_indicators, compute_stochastic_oscillator, compute_mfi, rank_assets
from src.decision_module import AdvancedDecisionModule
from src.trading_tool import TradingTool
from src.agent import TradingAgent
from src.indicators import load_config

logger = logging.getLogger(__name__)

# Caricamento del modello AI LSTM
model_lstm = load_model("lstm_trading_model.h5")

# Funzione per analizzare il sentiment con GPT-4
def analyze_sentiment(news: str) -> str:
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": f"Analizza il sentiment di questa news: {news}"}]
    )
    return response["choices"][0]["message"]["content"]

# Classe Strategia Backtrader con RSI, MACD, AI e Sentiment Analysis
class AITradingStrategy(bt.Strategy):
    def __init__(self):
        self.data_buffer = []
        self.rsi = bt.indicators.RSI(period=14)
        self.macd = bt.indicators.MACD(period_me1=12, period_me2=26, period_signal=9)
    
    def next(self):
        self.data_buffer.append(self.data.close[0])
        if len(self.data_buffer) < 50:
            return  # Aspetta di avere abbastanza dati per l’LSTM

        # Previsione AI
        X_test = np.array(self.data_buffer[-50:]).reshape(1, 50, 1)
        prediction = model_lstm.predict(X_test)[0][0]

        # Simulazione di una news finanziaria
        news = "La Federal Reserve ha annunciato un aumento dei tassi di interesse."
        sentiment = analyze_sentiment(news)

        # Logica di trading: RSI + MACD + AI + Sentiment
        if "positivo" in sentiment and self.rsi < 30 and self.macd.macd > self.macd.signal and prediction > self.data.close[0]:
            self.buy()
        elif "negativo" in sentiment and self.rsi > 70 and self.macd.macd < self.macd.signal and prediction < self.data.close[0]:
            self.sell()

# Funzione per eseguire backtesting su più asset
class MultiAssetBacktest:
    def __init__(self, assets, start_date, end_date, ib_host="127.0.0.1", ib_port=7497):
        self.ib = IB()
        self.ib.connect(ib_host, ib_port, clientId=1)
        self.assets = assets
        self.start_date = start_date
        self.end_date = end_date

    def run(self):
        cerebro = bt.Cerebro()
        cerebro.addstrategy(AITradingStrategy)

        for asset in self.assets:
            data = bt.feeds.IBData(
                dataname=asset,
                host='127.0.0.1',
                port=7497,
                timeframe=bt.TimeFrame.Days,
                fromdate=datetime.strptime(self.start_date, "%Y-%m-%d"),
                todate=datetime.strptime(self.end_date, "%Y-%m-%d")
            )
            cerebro.adddata(data)

        cerebro.run()
        cerebro.plot()

# Funzione per ottenere tutti gli asset disponibili su IBKR
def get_ibkr_assets():
    return ['AAPL', 'MSFT', 'GOOGL', 'EURUSD', 'BTCUSD', 'ES', 'CL']  # Esempio di asset

# Eseguire il backtest
if __name__ == "__main__":
    config = load_config()
    start_date = config.get("start_date", "2023-01-01")
    end_date = config.get("end_date", "2024-01-01")
    assets = get_ibkr_assets()

    backtest = MultiAssetBacktest(assets, start_date, end_date)
    backtest.run()
