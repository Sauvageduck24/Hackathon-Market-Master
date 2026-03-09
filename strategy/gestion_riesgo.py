"""Risk manager — translates accepted consensus signals into sized orders.

Pure rules + math, no ML:
  - Base allocation fraction scales linearly with consensus_score.
  - Hard cap: no single trade may exceed *max_trade_pct* of total portfolio.
  - Exposure guard: total non-fiat exposure must stay below *max_exposure_pct*.
  - Sell sizing also scales with consensus strength (partial vs full exit).
"""

EPS = 1e-9


class RiskManager:
    """Stateless sizer — all state comes from balances / market_data."""

    def __init__(
        self,
        max_trade_pct=0.20,
        min_trade_pct=0.05,
        max_exposure_pct=0.80,
    ):
        self.max_trade_pct = max_trade_pct
        self.min_trade_pct = min_trade_pct
        self.max_exposure_pct = max_exposure_pct

    # ------------------------------------------------------------------
    def size(self, signals, balances, market_data):
        """Convert consensus signals into executable orders.

        Args:
            signals:     list of {"pair", "side", "consensus_score", ...}.
            balances:    {"fiat": float, "token_1": float, ...}.
            market_data: {"token_1/fiat": {"close": ...}, "fee": float, ...}.

        Returns:
            List of {"pair", "side", "qty"}.
        """
        if not signals:
            return []

        fee = float(market_data.get("fee", 0.0003))
        portfolio = self._portfolio_value(balances, market_data)
        if portfolio <= EPS:
            return []

        exposure = self._current_exposure(balances, market_data, portfolio)
        actions = []

        for sig in signals:
            pair = sig["pair"]
            side = sig["side"]
            score = sig.get("consensus_score", 0.5)
            base, quote = pair.split("/")

            mkt = market_data.get(pair)
            if mkt is None:
                continue
            close = float(mkt.get("close", 0))
            if close <= EPS:
                continue

            if side == "buy":
                action = self._size_buy(
                    pair, base, quote, close, fee, score,
                    balances, portfolio, exposure,
                )
            else:
                action = self._size_sell(pair, base, score, balances)

            if action is not None:
                actions.append(action)
                # Update running exposure after accepting a buy
                if side == "buy":
                    added = action["qty"] * close / portfolio
                    exposure[pair] = exposure.get(pair, 0.0) + added

        return actions

    # ------------------------------------------------------------------
    def _size_buy(self, pair, base, quote, close, fee, score,
                  balances, portfolio, exposure):
        quote_bal = float(balances.get(quote, 0.0))
        if quote_bal <= EPS:
            return None

        # Linear scaling: min_trade_pct at score=0 → max_trade_pct at score=1
        alloc = self.min_trade_pct + (
            (self.max_trade_pct - self.min_trade_pct) * score
        )

        # Exposure guard
        total_exp = sum(exposure.values())
        room = max(0.0, self.max_exposure_pct - total_exp)
        alloc = min(alloc, room)
        if alloc < self.min_trade_pct:
            return None

        spend = min(portfolio * alloc, quote_bal)
        qty = spend / (close * (1.0 + fee))
        if qty <= 0:
            return None
        return {"pair": pair, "side": "buy", "qty": qty}

    @staticmethod
    def _size_sell(pair, base, score, balances):
        base_bal = float(balances.get(base, 0.0))
        if base_bal <= EPS:
            return None

        # 30 %–80 % of position depending on consensus strength
        frac = 0.30 + 0.50 * score
        frac = min(frac, 1.0)
        qty = base_bal * frac
        if qty <= 0:
            return None
        return {"pair": pair, "side": "sell", "qty": qty}

    # ------------------------------------------------------------------
    @staticmethod
    def _portfolio_value(balances, market_data):
        """Estimate total portfolio value denominated in fiat."""
        value = float(balances.get("fiat", 0.0))
        for pair, mkt in market_data.items():
            if pair == "fee" or not isinstance(mkt, dict):
                continue
            base, quote = pair.split("/")
            if quote == "fiat":
                close = float(mkt.get("close", 0))
                if close > 0:
                    value += float(balances.get(base, 0.0)) * close
        return value

    @staticmethod
    def _current_exposure(balances, market_data, portfolio):
        """Fraction of portfolio held in each non-fiat asset."""
        exp = {}
        if portfolio <= EPS:
            return exp
        for pair, mkt in market_data.items():
            if pair == "fee" or not isinstance(mkt, dict):
                continue
            base, quote = pair.split("/")
            if quote == "fiat":
                close = float(mkt.get("close", 0))
                if close > 0:
                    exp[pair] = float(balances.get(base, 0.0)) * close / portfolio
        return exp
