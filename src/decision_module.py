class AdvancedDecisionModule:
    def __init__(self, stoch_k_threshold: float = 20, mfi_threshold: float = 20):
        """
        Inizializza il modulo decisionale avanzato.
        
        :param stoch_k_threshold: Soglia per il %K stocastico (es. <20 indica ipervenduto).
        :param mfi_threshold: Soglia per il MFI (es. <20 indica ipervenduto).
        """
        self.stoch_k_threshold = stoch_k_threshold
        self.mfi_threshold = mfi_threshold

    def decide(self, indicators: dict, stoch_k: float, stoch_d: float, mfi: float) -> str:
        """
        Decide l'azione ("buy", "sell" o "hold") utilizzando gli indicatori base e quelli aggiuntivi.
        
        :param indicators: Dizionario contenente gli indicatori base (price, moving_average, RSI, MACD, BB_upper, BB_lower).
        :param stoch_k: Valore attuale del %K stocastico.
        :param stoch_d: Valore attuale del %D stocastico (attualmente non usato, ma disponibile per future logiche).
        :param mfi: Valore attuale del Money Flow Index.
        :return: "buy", "sell" o "hold".
        """
        price = indicators.get('price')
        sma = indicators.get('moving_average')
        rsi = indicators.get('RSI')
        macd_diff = indicators.get('MACD')
        bb_upper = indicators.get('BB_upper')
        bb_lower = indicators.get('BB_lower')

        # Logica avanzata:
        # BUY se: prezzo sotto la SMA (o sotto BB_lower) oppure RSI basso oppure %K in zona ipervenduta,
        # e MFI in zona ipervenduta ed un MACD significativamente negativo.
        if ((price < sma * 0.98 or price < bb_lower or rsi < 35 or stoch_k < self.stoch_k_threshold)
                and mfi < self.mfi_threshold and macd_diff < -0.5):
            return "buy"
        # SELL se: prezzo sopra la SMA (o sopra BB_upper) oppure RSI alto oppure %K in zona ipercomprata,
        # e MFI elevato ed un MACD significativamente positivo.
        elif ((price > sma * 1.02 or price > bb_upper or rsi > 65 or stoch_k > 80)
                and mfi > 80 and macd_diff > 0.5):
            return "sell"
        else:
            return "hold"
