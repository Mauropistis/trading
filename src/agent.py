from src.decision_module import AdvancedDecisionModule
from src.trading_tool import TradingTool

class TradingAgent:
    def __init__(self, decision_module: AdvancedDecisionModule = None, trading_tool: TradingTool = None):
        self.decision_module = decision_module if decision_module else AdvancedDecisionModule()
        self.trading_tool = trading_tool if trading_tool else TradingTool()

    def process_market_data(self, market_data: dict, stoch_k: float, stoch_d: float, mfi: float) -> str:
        return self.decision_module.decide(market_data, stoch_k, stoch_d, mfi)

    def execute(self, market_data: dict, stoch_k: float, stoch_d: float, mfi: float) -> str:
        decision = self.process_market_data(market_data, stoch_k, stoch_d, mfi)
        symbol = market_data.get('symbol', 'UNKNOWN')
        command = f"{decision} {symbol} 10"  # Ordine per 10 unità (modificabile tramite configurazione)
        result = self.trading_tool.use(command)
        return result
