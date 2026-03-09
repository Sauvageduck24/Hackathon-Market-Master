"""Risk manager — translates accepted consensus signals into sized orders.

Pure rules + math, no ML:
  - Base allocation fraction scales linearly with consensus_score.
  - Hard cap: no single trade may exceed *max_trade_pct* of total portfolio.
  - Exposure guard: total non-fiat exposure must stay below *max_exposure_pct*.
  - Sell sizing also scales with consensus strength (partial vs full exit).
    - Emergency exits via dynamic stop-loss / take-profit on unrealized PnL.
"""

EPS = 1e-9


class RiskManager:
    """Sizer with lightweight state for entry-price and emergency exits."""

    DEFAULT_FEE = 0.0003

    def __init__(self):
        self.params = {
            "max_trade_pct": 0.20,
            "min_trade_pct": 0.05,
            "max_exposure_pct": 0.80,
            "sell_min_frac": 0.30,
            "sell_max_frac": 0.80,
            "stop_loss_pct": 0.20,
            "take_profit_pct": 0.30,
            "emergency_sell_frac": 1.00,
        }
        self._avg_entry_fiat = {}
        self._prev_balances = None
        self._last_close_by_pair = {}
        self._sync_params()

    def _sync_params(self):
        min_trade = min(max(float(self.params["min_trade_pct"]), 0.0), 1.0)
        max_trade = min(max(float(self.params["max_trade_pct"]), min_trade), 1.0)
        max_exp = min(max(float(self.params["max_exposure_pct"]), 0.0), 1.0)
        sell_min = min(max(float(self.params["sell_min_frac"]), 0.0), 1.0)
        sell_max = min(max(float(self.params["sell_max_frac"]), sell_min), 1.0)
        stop_loss = max(float(self.params["stop_loss_pct"]), 0.0)
        take_profit = max(float(self.params["take_profit_pct"]), 0.0)
        emergency_frac = min(max(float(self.params["emergency_sell_frac"]), 0.0), 1.0)

        self.min_trade_pct = min_trade
        self.max_trade_pct = max_trade
        self.max_exposure_pct = max_exp
        self.sell_min_frac = sell_min
        self.sell_max_frac = sell_max
        self.stop_loss_pct = stop_loss
        self.take_profit_pct = take_profit
        self.emergency_sell_frac = emergency_frac

        self.params["min_trade_pct"] = self.min_trade_pct
        self.params["max_trade_pct"] = self.max_trade_pct
        self.params["max_exposure_pct"] = self.max_exposure_pct
        self.params["sell_min_frac"] = self.sell_min_frac
        self.params["sell_max_frac"] = self.sell_max_frac
        self.params["stop_loss_pct"] = self.stop_loss_pct
        self.params["take_profit_pct"] = self.take_profit_pct
        self.params["emergency_sell_frac"] = self.emergency_sell_frac

    def set_params(self, params):
        if not params:
            return
        self.params.update(params)
        self._sync_params()

    def reset(self):
        """Clear per-run state while preserving tuned parameters."""
        self._avg_entry_fiat = {}
        self._prev_balances = None
        self._last_close_by_pair = {}

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
        self._update_last_prices(market_data)
        self._update_position_tracking(balances)

        emergency_actions, blocked_pairs = self._emergency_actions(balances)

        if not signals:
            return emergency_actions

        fee = float(market_data.get("fee", self.DEFAULT_FEE))
        portfolio = self._portfolio_value(balances, market_data)
        if portfolio <= EPS:
            return []

        exposure = self._current_exposure(balances, market_data, portfolio)
        actions = []

        for sig in signals:
            pair = sig["pair"]
            if pair in blocked_pairs:
                continue
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

        return emergency_actions + actions

    # ------------------------------------------------------------------
    def _update_last_prices(self, market_data):
        for pair, mkt in market_data.items():
            if pair == "fee" or not isinstance(mkt, dict):
                continue
            close = float(mkt.get("close", 0.0))
            if close > EPS:
                self._last_close_by_pair[pair] = close

    def _pair_price(self, pair):
        close = self._last_close_by_pair.get(pair)
        if close is None:
            return None
        close = float(close)
        if close <= EPS:
            return None
        return close

    def _base_price_fiat(self, base):
        direct_pair = f"{base}/fiat"
        direct = self._pair_price(direct_pair)
        if direct is not None:
            return direct

        # Fallback for token_2 when only token_1/fiat and token_1/token_2 exist.
        if base == "token_2":
            p_1_fiat = self._pair_price("token_1/fiat")
            p_1_2 = self._pair_price("token_1/token_2")
            if p_1_fiat is not None and p_1_2 is not None:
                return p_1_fiat / p_1_2
        return None

    def _sell_pair_for_base(self, base):
        direct_pair = f"{base}/fiat"
        if self._pair_price(direct_pair) is not None:
            return direct_pair

        # As emergency fallback, sell base against token_2 if available.
        cross_pair = f"{base}/token_2"
        if self._pair_price(cross_pair) is not None:
            return cross_pair
        return None

    def _update_position_tracking(self, balances):
        if self._prev_balances is None:
            self._prev_balances = dict(balances)
            for asset, qty in balances.items():
                if asset == "fiat":
                    continue
                qty = float(qty)
                if qty <= EPS:
                    continue
                price = self._base_price_fiat(asset)
                if price is not None:
                    self._avg_entry_fiat[asset] = price
            return

        for asset, cur_qty_raw in balances.items():
            if asset == "fiat":
                continue

            prev_qty = float(self._prev_balances.get(asset, 0.0))
            cur_qty = float(cur_qty_raw)
            price = self._base_price_fiat(asset)

            if cur_qty <= EPS:
                self._avg_entry_fiat.pop(asset, None)
            elif prev_qty <= EPS:
                if price is not None:
                    self._avg_entry_fiat[asset] = price
            elif cur_qty > prev_qty + EPS:
                # Approximate weighted average cost after inventory increase.
                if price is not None:
                    prev_avg = self._avg_entry_fiat.get(asset, price)
                    added = cur_qty - prev_qty
                    new_avg = ((prev_qty * prev_avg) + (added * price)) / cur_qty
                    self._avg_entry_fiat[asset] = new_avg

        self._prev_balances = dict(balances)

    def _emergency_actions(self, balances):
        actions = []
        blocked_pairs = set()

        if self.emergency_sell_frac <= 0.0:
            return actions, blocked_pairs

        for asset, qty_raw in balances.items():
            if asset == "fiat":
                continue

            qty = float(qty_raw)
            if qty <= EPS:
                continue

            entry = self._avg_entry_fiat.get(asset)
            price = self._base_price_fiat(asset)
            if entry is None or price is None or entry <= EPS:
                continue

            pnl_pct = (price / entry) - 1.0
            stop_hit = pnl_pct <= -self.stop_loss_pct
            take_hit = self.take_profit_pct > 0.0 and pnl_pct >= self.take_profit_pct
            if not stop_hit and not take_hit:
                continue

            pair = self._sell_pair_for_base(asset)
            if pair is None:
                continue

            sell_qty = qty * self.emergency_sell_frac
            if sell_qty <= EPS:
                continue

            blocked_pairs.add(pair)
            actions.append({
                "pair": pair,
                "side": "sell",
                "qty": sell_qty,
                "reason": "stop_loss" if stop_hit else "take_profit",
                "unrealized_pnl_pct": pnl_pct,
            })

        return actions, blocked_pairs

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

    def _size_sell(self, pair, base, score, balances):
        base_bal = float(balances.get(base, 0.0))
        if base_bal <= EPS:
            return None

        # Fraction of position depending on consensus strength.
        spread = max(self.sell_max_frac - self.sell_min_frac, 0.0)
        frac = self.sell_min_frac + spread * score
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