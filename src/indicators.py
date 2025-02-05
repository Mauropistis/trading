import yfinance as yf
import pandas as pd
import numpy as np
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)

# --- Indicatori di base ---

def compute_sma(series: pd.Series, window: int) -> pd.Series:
    """Calcola la Simple Moving Average (SMA) su una serie temporale."""
    return series.rolling(window=window).mean()

def compute_ema(series: pd.Series, window: int) -> pd.Series:
    """Calcola la Exponential Moving Average (EMA) su una serie temporale."""
    return series.ewm(span=window, adjust=False).mean()

def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Calcola il Relative Strength Index (RSI) su una serie temporale."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(series: pd.Series, fast_window: int = 12, slow_window: int = 26, signal_window: int = 9) -> tuple:
    """Calcola il MACD e la Signal Line su una serie temporale."""
    ema_fast = compute_ema(series, fast_window)
    ema_slow = compute_ema(series, slow_window)
    macd_line = ema_fast - ema_slow
    signal_line = compute_ema(macd_line, signal_window)
    return macd_line, signal_line

def compute_bollinger_bands(series: pd.Series, window: int = 20, num_std: int = 2) -> tuple:
    """Calcola le Bollinger Bands (SMA, banda superiore, banda inferiore) su una serie temporale."""
    sma = compute_sma(series, window)
    rolling_std = series.rolling(window=window).std()
    upper_band = sma + num_std * rolling_std
    lower_band = sma - num_std * rolling_std
    return sma, upper_band, lower_band


def compute_stochastic_oscillator(df: pd.DataFrame, window: int = 14) -> tuple:
    """
    Calcola lo Stochastic Oscillator (%K e %D).
    
    :param df: DataFrame contenente le colonne 'High', 'Low' e 'Close'.
    :param window: Periodo per il calcolo (default 14 giorni).
    :return: Tuple contenente %K e %D (Series).
    """
    low_min = df['Low'].rolling(window=window).min()
    high_max = df['High'].rolling(window=window).max()
    stoch_k = 100 * ((df['Close'] - low_min) / (high_max - low_min))
    stoch_d = stoch_k.rolling(window=3).mean()
    return stoch_k, stoch_d

def compute_mfi(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Calcola il Money Flow Index (MFI).
    
    :param df: DataFrame contenente 'High', 'Low', 'Close' e 'Volume'.
    :param window: Periodo per il calcolo (default 14 giorni).
    :return: Serie contenente il MFI.
    """
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    money_flow = typical_price * df['Volume']
    diff = typical_price.diff()
    positive_flow = money_flow.where(diff > 0, 0)
    negative_flow = money_flow.where(diff < 0, 0)
    pos_mf = positive_flow.rolling(window=window).sum()
    neg_mf = negative_flow.rolling(window=window).sum()
    mfi = 100 - (100 / (1 + pos_mf / neg_mf))
    return mfi

# --- Download e pulizia dei dati ---

def fetch_real_data(ticker: str = 'AAPL', start_date: str = '2022-01-01', end_date: str = '2022-12-31') -> pd.DataFrame:
    """Scarica i dati storici per un ticker utilizzando yfinance."""
    df = yf.download(ticker, start=start_date, end=end_date)
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Rimuove le righe con NaN nella colonna 'Close'."""
    if df['Close'].isnull().to_numpy().sum() > 0:
        logger.info("Valori NaN trovati in 'Close', rimuovo le righe corrispondenti.")
        df = df.dropna(subset=['Close'])
    return df

def compute_indicators(df: pd.DataFrame) -> dict:
    """
    Calcola gli indicatori tecnici base su un DataFrame di dati storici.
    
    :param df: DataFrame con i dati storici.
    :return: Dizionario contenente: price, SMA20, RSI14, MACD diff, BB_upper, BB_lower.
    """
    # Crea una copia del DataFrame per evitare di lavorare su una slice
    df = df.copy()

    df.loc[:, 'SMA20'] = compute_sma(df['Close'], window=20)
    df.loc[:, 'RSI14'] = compute_rsi(df['Close'], window=14)
    macd_line, signal_line = compute_macd(df['Close'])
    df.loc[:, 'MACD'] = macd_line
    df.loc[:, 'Signal'] = signal_line
    sma_bb, upper_bb, lower_bb = compute_bollinger_bands(df['Close'], window=20, num_std=2)
    df.loc[:, 'BB_MA'] = sma_bb
    df.loc[:, 'BB_upper'] = upper_bb
    df.loc[:, 'BB_lower'] = lower_bb

    last = df.iloc[-1]
    indicators = {
        'price': last['Close'].item(),
        'moving_average': last['SMA20'].item(),
        'RSI': last['RSI14'].item(),
        'MACD': (last['MACD'] - last['Signal']).item(),
        'BB_upper': last['BB_upper'].item(),
        'BB_lower': last['BB_lower'].item()
    }
    return indicators



def load_config(config_file: str = "config.json") -> dict:
    """
    Carica la configurazione da un file JSON.
    
    :param config_file: Percorso del file di configurazione.
    :return: Dizionario con i parametri.
    """
    default_config = {
        "ticker": "AAPL",
        "start_date": "1980-01-01",
        "end_date": datetime.now().strftime("%Y-%m-%d"),
        "buy_threshold": 0.02,
        "sell_threshold": 0.02,
        "order_quantity": 10,
        "stoch_k_threshold": 20,
        "mfi_threshold": 20
    }
    try:
        with open(config_file, "r") as f:
            config = json.load(f)
        return config
    except (FileNotFoundError, json.JSONDecodeError):
        print("Problema nel file di configurazione. Uso i parametri di default.")
        logger.error("Problema nel file di configurazione. Uso i parametri di default.")
        logger.error("Configurazione di default: %s", default_config)
        return default_config

def rank_assets(tickers: list, start_date: str, end_date: str, config: dict) -> pd.DataFrame:
    """
    Scarica i dati per una lista di ticker, calcola il ritorno annuale, la volatilità annualizzata,
    il volume medio e gli indicatori base, e calcola uno score (Return/Volatility).
    Ritorna un DataFrame ordinato per score decrescente.
    
    :param tickers: Lista di ticker.
    :param start_date: Data di inizio.
    :param end_date: Data di fine.
    :param config: Configurazione (opzionale per eventuali soglie).
    :return: DataFrame con la classifica degli asset.
    """
    metrics_list = []
    for ticker in tickers:
        try:
            df = fetch_real_data(ticker, start_date, end_date)
        except Exception as e:
            logger.error("Errore nel download dei dati per %s: %s", ticker, e)
            continue
        if df.empty:
            logger.warning("Nessun dato disponibile per %s", ticker)
            continue
        try:
            first_close = float(df['Close'].iloc[0].item())
            last_close = float(df['Close'].iloc[-1].item())
            annual_return = (last_close / first_close) - 1

            daily_returns = df['Close'].pct_change().dropna()
            volatility = float(daily_returns.std().item()) * np.sqrt(252)

            avg_volume = float(df['Volume'].mean().item())
            score = (annual_return / volatility) if volatility != 0 else np.nan

            # Calcola gli indicatori base
            indicators = compute_indicators(df)
        except Exception as e:
            logger.error("Errore nel calcolo delle metriche per %s: %s", ticker, e)
            continue

        metrics_list.append({
            'Ticker': ticker,
            'Price': indicators.get('price'),
            'Moving Average': indicators.get('moving_average'),
            'RSI': indicators.get('RSI'),
            'MACD': indicators.get('MACD'),
            'BB_upper': indicators.get('BB_upper'),
            'BB_lower': indicators.get('BB_lower'),
            'Annual Return': annual_return,
            'Volatility': volatility,
            'Avg Volume': avg_volume,
            'Score': score
        })
    df_metrics = pd.DataFrame(metrics_list)
    df_metrics['Score'] = pd.to_numeric(df_metrics['Score'], errors='coerce')
    df_metrics.sort_values(by='Score', ascending=False, inplace=True)
    return df_metrics
