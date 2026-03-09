from collections import defaultdict, deque
import numpy as np

DEFAULT_FEE = 0.0003  # 3 bps
EPS = 1e-9


class Strategy:
    def __init__(self):
        self.window = 30
        self.warmup = 40
        self.max_history = 120
        self.base_band = 1.25
        self.cooldown_steps = 3

        self.price_history = defaultdict(lambda: deque(maxlen=self.max_history))
        self.last_trade_step = defaultdict(lambda: -10_000)
        self.step = 0

    def _can_trade(self, pair):
        return (self.step - self.last_trade_step[pair]) >= self.cooldown_steps

    def _mark_trade(self, pair):
        self.last_trade_step[pair] = self.step

    def _mean_reversion_signal(self, pair, close, balances, fee):
        hist = self.price_history[pair]
        if len(hist) < self.warmup or not self._can_trade(pair):
            return None

        arr = np.asarray(hist, dtype=float)
        recent = arr[-self.window:]
        mu = recent.mean()
        sigma = recent.std(ddof=1)

        if sigma < EPS:
            return None

        z = (close - mu) / sigma
        trend = (arr[-1] - arr[-self.window]) / (abs(arr[-self.window]) + EPS)
        band = self.base_band + min(abs(trend) * 6.0, 0.75)

        base, quote = pair.split("/")
        base_bal = float(balances.get(base, 0.0))
        quote_bal = float(balances.get(quote, 0.0))

        # Comprar cuando hay desviación negativa, pero evitar caídas demasiado violentas.
        if z < -band and trend > -0.04 and quote_bal > 0:
            risk_fraction = min(0.10 + 0.05 * abs(z), 0.25)
            qty = (quote_bal * risk_fraction) / (close * (1.0 + fee))
            if qty > 0:
                self._mark_trade(pair)
                return {"pair": pair, "side": "buy", "qty": qty}

        # Vender parcialmente con desviación positiva.
        if z > band and base_bal > 0:
            sell_fraction = min(0.25 + 0.10 * abs(z), 0.80)
            qty = base_bal * sell_fraction
            if qty > 0:
                self._mark_trade(pair)
                return {"pair": pair, "side": "sell", "qty": qty}

        return None

    def _arbitrage_signal(self, market_data, balances, fee):
        needed = ["token_1/fiat", "token_2/fiat", "token_1/token_2"]
        if not all(k in market_data for k in needed):
            return None
        if not self._can_trade("token_1/token_2"):
            return None

        p1 = float(market_data["token_1/fiat"]["close"])
        p2 = float(market_data["token_2/fiat"]["close"])
        p12 = float(market_data["token_1/token_2"]["close"])

        implied = p1 / (p2 + EPS)

        # Umbral considerando fee total de ida y vuelta + margen mínimo.
        edge = max(0.0025, 4.0 * fee)

        # token_1/token_2 barato => comprar token_1 con token_2
        if p12 < implied * (1.0 - edge):
            token2_bal = float(balances.get("token_2", 0.0))
            spend_token2 = token2_bal * 0.20
            qty_token1 = spend_token2 / (p12 * (1.0 + fee) + EPS)
            if qty_token1 > 0:
                self._mark_trade("token_1/token_2")
                return {"pair": "token_1/token_2", "side": "buy", "qty": qty_token1}

        # token_1/token_2 caro => vender token_1 por token_2
        if p12 > implied * (1.0 + edge):
            token1_bal = float(balances.get("token_1", 0.0))
            qty_token1 = token1_bal * 0.20
            if qty_token1 > 0:
                self._mark_trade("token_1/token_2")
                return {"pair": "token_1/token_2", "side": "sell", "qty": qty_token1}

        return None

    def on_data(self, market_data, balances):
        self.step += 1
        fee = float(market_data.get("fee", DEFAULT_FEE))
        actions = []

        for pair, data in market_data.items():
            if pair == "fee":
                continue
            close = float(data["close"])
            self.price_history[pair].append(close)

            signal = self._mean_reversion_signal(pair, close, balances, fee)
            if signal is not None:
                actions.append(signal)

        arb = self._arbitrage_signal(market_data, balances, fee)
        if arb is not None:
            actions.append(arb)

        return actions if actions else None