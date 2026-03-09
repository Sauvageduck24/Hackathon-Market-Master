from collections import defaultdict, deque

import numpy as np

DEFAULT_FEE = 0.0003
EPS = 1e-9


class Strategy:
    """Fast intraday momentum scalping strategy for 1-minute candles."""

    def __init__(self):
        self.fast_window = 5
        self.slow_window = 15
        self.vol_window = 20
        self.atr_window = 14
        self.max_history = 120
        self.cooldown_steps = 2

        # Per-pair market history
        self.close_history = defaultdict(lambda: deque(maxlen=self.max_history))
        self.high_history = defaultdict(lambda: deque(maxlen=self.max_history))
        self.low_history = defaultdict(lambda: deque(maxlen=self.max_history))
        self.volume_history = defaultdict(lambda: deque(maxlen=self.max_history))

        # Per-pair position state for exits
        self.position = defaultdict(float)
        self.entry_price = defaultdict(float)
        self.last_trade_step = defaultdict(lambda: -10_000)
        self.step = 0

    def _can_trade(self, pair):
        return (self.step - self.last_trade_step[pair]) >= self.cooldown_steps

    def _mark_trade(self, pair):
        self.last_trade_step[pair] = self.step

    def _sma(self, arr, window):
        if len(arr) < window:
            return None
        return float(np.mean(arr[-window:]))

    def _atr(self, highs, lows, closes, window):
        if len(closes) < window + 1:
            return None

        tr_values = []
        start = len(closes) - window
        for i in range(start, len(closes)):
            prev_close = closes[i - 1]
            high = highs[i]
            low = lows[i]
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            tr_values.append(tr)
        return float(np.mean(tr_values))

    def _entry_signal(self, pair, close, balances, fee):
        closes = np.asarray(self.close_history[pair], dtype=float)
        highs = np.asarray(self.high_history[pair], dtype=float)
        lows = np.asarray(self.low_history[pair], dtype=float)
        volumes = np.asarray(self.volume_history[pair], dtype=float)

        need = max(self.slow_window, self.vol_window, self.atr_window + 1)
        if len(closes) < need or not self._can_trade(pair):
            return None

        fast = self._sma(closes, self.fast_window)
        slow = self._sma(closes, self.slow_window)
        if fast is None or slow is None:
            return None

        # Momentum and breakout filter
        recent_high = float(np.max(closes[-self.fast_window:]))
        volume_now = float(volumes[-1])
        volume_avg = float(np.mean(volumes[-self.vol_window:])) + EPS
        volume_ratio = volume_now / volume_avg

        # Volatility-aware risk size
        atr = self._atr(highs, lows, closes, self.atr_window)
        if atr is None or atr <= EPS:
            return None

        base, quote = pair.split("/")
        quote_bal = float(balances.get(quote, 0.0))

        trend_up = fast > slow
        breakout = close >= recent_high

        if trend_up and breakout and volume_ratio > 1.2 and quote_bal > 0:
            # Spend between 10% and 30% depending on signal quality.
            signal_strength = min((fast - slow) / (slow + EPS), 0.01) / 0.01
            alloc = 0.10 + 0.20 * max(signal_strength, 0.0)
            alloc = min(max(alloc, 0.10), 0.30)
            qty = (quote_bal * alloc) / (close * (1.0 + fee))
            if qty > 0:
                self._mark_trade(pair)
                return {"pair": pair, "side": "buy", "qty": qty}

        return None

    def _exit_signal(self, pair, close, balances, fee):
        closes = np.asarray(self.close_history[pair], dtype=float)
        highs = np.asarray(self.high_history[pair], dtype=float)
        lows = np.asarray(self.low_history[pair], dtype=float)

        if len(closes) < max(self.slow_window, self.atr_window + 1):
            return None

        base, _ = pair.split("/")
        base_bal = float(balances.get(base, 0.0))
        if base_bal <= 0:
            return None

        fast = self._sma(closes, self.fast_window)
        slow = self._sma(closes, self.slow_window)
        atr = self._atr(highs, lows, closes, self.atr_window)
        if fast is None or slow is None or atr is None:
            return None

        # If we have a registered entry, use ATR-based stop and take.
        if self.position[pair] > 0 and self.entry_price[pair] > 0:
            entry = self.entry_price[pair]
            stop = entry - 1.2 * atr
            take = entry + 1.8 * atr

            if close <= stop or close >= take:
                self._mark_trade(pair)
                return {"pair": pair, "side": "sell", "qty": base_bal}

        # Trend reversal fallback exit
        if fast < slow:
            self._mark_trade(pair)
            return {"pair": pair, "side": "sell", "qty": base_bal * 0.75}

        return None

    def on_data(self, market_data, balances):
        self.step += 1
        fee = float(market_data.get("fee", DEFAULT_FEE))
        actions = []

        for pair, data in market_data.items():
            if pair == "fee":
                continue

            close = float(data["close"])
            high = float(data.get("high", close))
            low = float(data.get("low", close))
            volume = float(data.get("volume", 0.0))

            self.close_history[pair].append(close)
            self.high_history[pair].append(high)
            self.low_history[pair].append(low)
            self.volume_history[pair].append(volume)

            # Exit first for faster risk-off behavior on 1-minute bars.
            exit_action = self._exit_signal(pair, close, balances, fee)
            if exit_action is not None:
                actions.append(exit_action)
                continue

            entry_action = self._entry_signal(pair, close, balances, fee)
            if entry_action is not None:
                actions.append(entry_action)

        # Update local position cache using current balances after decisions.
        for pair in self.close_history:
            base, _ = pair.split("/")
            self.position[pair] = float(balances.get(base, 0.0))
            if self.position[pair] <= EPS:
                self.entry_price[pair] = 0.0
            elif self.entry_price[pair] <= EPS and len(self.close_history[pair]) > 0:
                self.entry_price[pair] = float(self.close_history[pair][-1])

        return actions if actions else None
