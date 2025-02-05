class TradingTool:
    def name(self):
        return "Trading Tool"

    def description(self):
        return ("Fornisce operazioni di trading. Comandi supportati: "
                "'get_price <simbolo>', 'buy <simbolo> <quantità>' e 'sell <simbolo> <quantità>'.")

    def use(self, command):
        # Simula l'esecuzione del comando (qui potresti integrare una chiamata API)
        return f"Eseguito comando: {command}"
