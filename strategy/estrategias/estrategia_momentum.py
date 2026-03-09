"""Trend-following momentum strategy — complementary to Bollinger (mean
reversion).

Signals are generated using two independent confirmations:
  1. EMA crossover  (fast EMA > slow EMA → bullish trend)
  2. Rate-of-change (ROC above threshold → positive momentum)

A buy requires BOTH conditions.  A sell requires the inverse of both.
This avoids whipsaws that a single-indicator approach would suffer from.

Same interface as estrategia_bollinger: ``Strategy.on_data(market_data,
balances) -> list[dict] | None``.
"""

from collections import defaultdict, deque

import numpy as np

DEFAULT_FEE = 0.0003
EPS = 1e-9


class Strategy:
    """EMA-crossover + ROC momentum strategy for 1-minute candles."""

    def __init__(self):
        self.fast_period = 8
        self.slow_period = 21
        self.roc_period = 10
        self.roc_threshold = 0.005      # 0.5 % minimum ROC to act
        self.max_history = 120
        self.cooldown_steps = 4

        self.price_history = defaultdict(lambda: deque(maxlen=self.max_history))
        self.last_trade_step = defaultdict(lambda: -10_000)
        self.step = 0

    # -- helpers --------------------------------------------------------
    def _can_trade(self, pair):
        return (self.step - self.last_trade_step[pair]) >= self.cooldown_steps

    def _mark_trade(self, pair):
        self.last_trade_step[pair] = self.step

    @staticmethod
    def _ema(arr, period):
        """Simple full-history EMA."""
        if len(arr) < period:
            return None
        alpha = 2.0 / (period + 1)
        result = float(arr[0])
        for v in arr[1:]:
            result = alpha * float(v) + (1.0 - alpha) * result
        return result

    @staticmethod
    def _roc(arr, period):
        """Rate of change over *period* bars."""
        if len(arr) <= period:
            return None
        cur = float(arr[-1])
        prev = float(arr[-period - 1])
        if abs(prev) < EPS:
            return None
        return (cur - prev) / prev

    # -- main entry -----------------------------------------------------
    def on_data(self, market_data, balances):
        self.step += 1
        fee = float(market_data.get("fee", DEFAULT_FEE))
        actions = []

        for pair, data in market_data.items():
            if pair == "fee":
                continue
            close = float(data["close"])
            self.price_history[pair].append(close)

            sig = self._signal(pair, close, balances, fee)
            if sig is not None:
                actions.append(sig)

        return actions if actions else None

    # -- signal logic ---------------------------------------------------
    def _signal(self, pair, close, balances, fee):
        hist = self.price_history[pair]
        warmup = max(self.slow_period + 5, self.roc_period + 1)
        if len(hist) < warmup or not self._can_trade(pair):
            return None

        arr = np.asarray(hist, dtype=float)

        fast_ema = self._ema(arr, self.fast_period)
        slow_ema = self._ema(arr, self.slow_period)
        if fast_ema is None or slow_ema is None:
            return None

        roc = self._roc(arr, self.roc_period)
        if roc is None:
            return None

        base, quote = pair.split("/")
        base_bal = float(balances.get(base, 0.0))
        quote_bal = float(balances.get(quote, 0.0))

        # --- BUY: trend up + positive momentum -------------------------
        if fast_ema > slow_ema and roc > self.roc_threshold and quote_bal > 0:
            strength = min(abs(roc) / 0.02, 1.0)
            risk_frac = min(0.08 + 0.12 * strength, 0.20)
            qty = (quote_bal * risk_frac) / (close * (1.0 + fee))
            if qty > 0:
                self._mark_trade(pair)
                return {"pair": pair, "side": "buy", "qty": qty}

        # --- SELL: trend down + negative momentum -----------------------
        if fast_ema < slow_ema and roc < -self.roc_threshold and base_bal > 0:
            strength = min(abs(roc) / 0.02, 1.0)
            sell_frac = min(0.30 + 0.30 * strength, 0.80)
            qty = base_bal * sell_frac
            if qty > 0:
                self._mark_trade(pair)
                return {"pair": pair, "side": "sell", "qty": qty}

        return None
