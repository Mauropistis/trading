import logging
import numpy as np
from datetime import datetime
import backtrader as bt
from ib_insync import IB

from indicators import fetch_real_data, compute_indicators
from decision_module import AdvancedDecisionModule
from trading_tool import TradingTool
from agent import TradingAgent
from multi_agent_trading import MultiAgentCoordinator, MultiAgentTradingSystem, TrendFollowerAgent, MeanReversionAgent, RiskManagerAgent, MarketMakerAgent

logger = logging.getLogger(__name__)

# Creazione degli agenti RL
trend_follower = TrendFollowerAgent("models/trend_rl_model")
mean_reversion = MeanReversionAgent("models/mean_reversion_rl_model")
risk_manager = RiskManagerAgent("models/risk_manager_rl_model")
market_maker = MarketMakerAgent("models/market_maker_rl_model")

# Coordinatore multi-agente
multi_agent_coordinator = MultiAgentCoordinator([trend_follower, mean_reversion, risk_manager, market_maker])

# Creazione del sistema di trading multi-agente
multi_agent_trading_system = MultiAgentTradingSystem(multi_agent_coordinator)

# Classe Strategia Backtrader
class AITradingStrategy(bt.Strategy):
    def __init__(self):
        self.data_buffer = []
        self.rsi = bt.indicators.RSI(period=14)
        self.macd = bt.indicators.MACD(period_me1=12, period_me2=26, period_signal=9)
    
    def next(self):
        self.data_buffer.append(self.data.close[0])
        if len(self.data_buffer) < 50:
            return

        X_test = np.array(self.data_buffer[-50:]).reshape(1, 50, 1)
        price = self.data.close[0]
        
        multi_agent_trading_system.execute_trade(X_test, price)

# Backtest Multi-Asset
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

# Avvia il backtest
if __name__ == "__main__":
    assets = ["AAPL", "MSFT", "GOOGL", "EURUSD", "BTCUSD"]
    backtest = MultiAssetBacktest(assets, "2023-01-01", "2024-01-01")
    backtest.run()
