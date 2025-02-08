import numpy as np
import tensorflow as tf
from stable_baselines3 import PPO, SAC

class TrendFollowerAgent:
    def __init__(self, model_path):
        self.model = PPO.load(model_path)

    def decide(self, market_data):
        action, _ = self.model.predict(market_data)
        return "buy" if action == 1 else "sell" if action == 2 else "hold"

class MeanReversionAgent:
    def __init__(self, model_path):
        self.model = PPO.load(model_path)

    def decide(self, market_data):
        action, _ = self.model.predict(market_data)
        return "buy" if action == 1 else "sell" if action == 2 else "hold"

class RiskManagerAgent:
    def __init__(self, model_path):
        self.model = SAC.load(model_path)

    def decide(self, market_data):
        action, _ = self.model.predict(market_data)
        return "modify_sl" if action == 3 else "modify_tp" if action == 4 else "hold"

class MarketMakerAgent:
    def __init__(self, model_path):
        self.model = PPO.load(model_path)

    def decide(self, market_data):
        action, _ = self.model.predict(market_data)
        return "buy" if action == 1 else "sell" if action == 2 else "hold"

class MultiAgentCoordinator:
    def __init__(self, agents):
        self.agents = agents

    def decide_trade(self, market_data):
        decisions = [agent.decide(market_data) for agent in self.agents]
        final_decision = max(set(decisions), key=decisions.count)
        return final_decision

class MultiAgentTradingSystem:
    def __init__(self, coordinator, initial_balance=10000):
        self.coordinator = coordinator
        self.balance = initial_balance
        self.position = 0
        self.stop_loss = 0
        self.take_profit = 0

    def execute_trade(self, market_data, price):
        decision = self.coordinator.decide_trade(market_data)

        if decision == "buy" and self.position == 0:
            self.position = self.balance / price
            self.stop_loss = price * 0.98
            self.take_profit = price * 1.02
            self.balance = 0
            print(f"BUY at {price:.2f}, SL: {self.stop_loss:.2f}, TP: {self.take_profit:.2f}")

        elif decision == "sell" and self.position > 0:
            self.balance = self.position * price
            self.position = 0
            print(f"SELL at {price:.2f}, New Balance: {self.balance:.2f}")

        elif decision == "modify_sl" and self.position > 0:
            self.stop_loss = max(self.stop_loss, price * 0.99)
            print(f"Updated Stop Loss to {self.stop_loss:.2f}")

        elif decision == "modify_tp" and self.position > 0:
            self.take_profit = max(self.take_profit, price * 1.01)
            print(f"Updated Take Profit to {self.take_profit:.2f}")
