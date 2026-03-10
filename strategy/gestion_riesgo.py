"""Risk manager that translates accepted consensus signals into sized orders.

Pure rules + math, no ML:
    - Base allocation fraction scales linearly with consensus_score.
    - Hard cap: no single trade may exceed *max_trade_pct* of total portfolio.
    - Exposure guard: total non-fiat exposure must stay below *max_exposure_pct*.
    - Sell sizing scales with consensus strength (partial vs full exit).

Emergency exits are disabled by design (`emergency_sell_frac = 0.0`) and related
parameters are fixed so the optimizer does not spend trials on no-op dimensions.
"""

EPS = 1e-9
MIN_ACTION_QTY = 0.01
MIN_SELL_QTY = {
    "token_2": 0.01,
    "default": 0.001,
}


class RiskManager:
    """Portfolio-aware order sizer."""

    DEFAULT_FEE = 0.0003
    FIXED_PARAMS = {
        "take_profit_pct": 0.0,
        "emergency_sell_frac": 0.0,
        "sell_max_frac": 0.85,
        "stop_loss_pct": 0.0,
        "warmup_ticks": 0,
    }

    def __init__(self):
        self.params = {
            "max_trade_pct": 0.15,
            "min_trade_pct": 0.05,
            "max_exposure_pct": 0.80,
            "sell_min_frac": 0.50,
        }
        self._sync_params()

    def _sync_params(self):
        p = dict(self.FIXED_PARAMS)
        p.update(self.params)

        min_trade = min(max(float(p["min_trade_pct"]), 0.0), 1.0)
        max_trade = min(max(float(p["max_trade_pct"]), min_trade), 1.0)
        max_exp = min(max(float(p["max_exposure_pct"]), 0.0), 1.0)
        sell_min = min(max(float(p["sell_min_frac"]), 0.0), 1.0)
        sell_max = min(max(float(p["sell_max_frac"]), sell_min), 1.0)
        stop_loss = max(float(p["stop_loss_pct"]), 0.0)
        take_profit = max(float(p["take_profit_pct"]), 0.0)
        warmup_ticks = max(int(p.get("warmup_ticks", 0)), 0)
        emergency_frac = min(max(float(p["emergency_sell_frac"]), 0.0), 1.0)

        self.min_trade_pct = min_trade
        self.max_trade_pct = max_trade
        self.max_exposure_pct = max_exp
        self.sell_min_frac = sell_min
        self.sell_max_frac = sell_max
        # Kept as attributes for compatibility, but emergency logic is disabled.
        self.stop_loss_pct = stop_loss
        self.take_profit_pct = take_profit
        self.warmup_ticks = warmup_ticks
        self.emergency_sell_frac = emergency_frac

        self.params["min_trade_pct"] = self.min_trade_pct
        self.params["max_trade_pct"] = self.max_trade_pct
        self.params["max_exposure_pct"] = self.max_exposure_pct
        self.params["sell_min_frac"] = self.sell_min_frac

    def set_params(self, params):
        if not params:
            return
        tunables = {k: v for k, v in params.items() if k not in self.FIXED_PARAMS}

        # Constraint: keep trade bounds ordered to preserve linear sizing span.
        # If an optimizer proposes an inverted range, fold it into a valid one.
        if "min_trade_pct" in tunables and "max_trade_pct" in tunables:
            min_trade = float(tunables["min_trade_pct"])
            max_trade = float(tunables["max_trade_pct"])
            if min_trade > max_trade:
                tunables["min_trade_pct"], tunables["max_trade_pct"] = max_trade, min_trade

        if tunables:
            self.params.update(tunables)
        self._sync_params()

    def reset(self):
        """Compatibility no-op: manager is stateless across steps."""
        return

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

        fee = float(market_data.get("fee", self.DEFAULT_FEE))
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
                    pair, quote, close, fee, score,
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

        return [a for a in actions if float(a.get("qty", 0.0)) >= MIN_ACTION_QTY]

    # ------------------------------------------------------------------
    def _size_buy(self, pair, quote, close, fee, score,
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

    def _size_sell(self, pair, base, score, balances):
        base_bal = float(balances.get(base, 0.0))
        if base_bal <= EPS:
            return None

        # Fraction of position depending on consensus strength.
        spread = max(self.sell_max_frac - self.sell_min_frac, 0.0)
        frac = self.sell_min_frac + spread * score
        frac = min(frac, 1.0)
        qty = base_bal * frac

        # Avoid dust cascades: if the computed slice is too small, exit all-or-nothing.
        min_q = MIN_SELL_QTY.get(base, MIN_SELL_QTY["default"])
        if qty < min_q:
            qty = base_bal if base_bal >= min_q else 0.0

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