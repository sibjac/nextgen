import pandas as pd
import pandas_ta as ta
from typing import Any, Dict, Optional
from core.base import BaseStrategy

class TrailingMonsterStrategy(BaseStrategy):
    """
    Trailing Monster Strategy:
    Buy: RSI > overbought, close > SMA, PKAMA bullish, entry conditions met
    Exit: ATR-based trailing stop after minimum profit met
    """

    def __init__(
        self,
        rsi_length: int = 7,
        rsi_overbought: int = 70,
        sma_length: int = 5,
        pkama_length: int = 9,
        pkama_factor: float = 1.0,
        use_self_powered: bool = True,
        atr_length: int = 1,
        atr_multiplier: float = 1.0,
        profit_percentage: float = 1.5
    ):
        self.rsi_length = rsi_length
        self.rsi_overbought = rsi_overbought
        self.sma_length = sma_length
        self.pkama_length = pkama_length
        self.pkama_factor = pkama_factor
        self.use_self_powered = use_self_powered
        self.atr_length = atr_length
        self.atr_multiplier = atr_multiplier
        self.profit_percentage = profit_percentage
        self.last_signal = None

    def compute_pkama(self, close: pd.Series) -> pd.Series:
        # Custom PKAMA logic adapted from your Pine Script
        er = abs(close.diff(self.pkama_length)) / abs(close.diff()).rolling(self.pkama_length).sum()
        pow_val = 1 / er if self.use_self_powered else self.pkama_factor
        per = er ** pow_val
        a = [close.iloc[0]]
        for i in range(1, len(close)):
            a.append(per.iloc[i] * close.iloc[i] + (1 - per.iloc[i]) * a[i - 1])
        return pd.Series(a, index=close.index)

    def on_candle(self, symbol: str, timeframe: str, candles: pd.DataFrame) -> Optional[Dict[str, Any]]:
        if len(candles) < max(self.rsi_length, self.sma_length, self.pkama_length, self.atr_length) + 2:
            return None

        df = candles.copy()
        df['rsi'] = ta.rsi(df['close'], length=self.rsi_length)
        df['sma'] = ta.sma(df['close'], length=self.sma_length)
        df['pkama'] = self.compute_pkama(df['close'])
        df['bullish'] = df['pkama'] > df['pkama'].shift(1)
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=self.atr_length)

        prev, curr = df.iloc[-2], df.iloc[-1]
        signal = None

        # Entry logic
        if (
            curr['rsi'] > self.rsi_overbought and
            curr['close'] > curr['sma'] and
            curr['bullish']
        ):
            signal = {'action': 'BUY', 'confidence': float(curr['rsi']) / 100, 'quantity': 1}

        # Trailing stop/exit logic: implement according to your system's position management
        # Example placeholder:
        # if in_position and profit > min_required and trailing_stop_hit:
        #     signal = {'action': 'SELL', ...}

        if signal and signal != self.last_signal:
            self.last_signal = signal
            return signal
        return None

    def reset(self) -> None:
        self.last_signal = None
