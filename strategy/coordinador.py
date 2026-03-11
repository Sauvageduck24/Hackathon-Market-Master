"""Position coordinator — simple alignment filter between signals and balances.

Regla fija:
    - Buy solo si no hay posicion significativa en el activo base.
    - Sell solo si hay posicion en el activo base.
"""

EPS = 1e-9

class PositionCoordinator:
    """Filtro minimo y determinista con parametros configurables."""

    POSITION_EPS = 1e-9

    def __init__(self):
        self.params = {
            "buy_min_score": 0.3,
            "sell_min_score": 0.4,
            "min_pos_frac": 0.02,
        }
        self._sync_params()

    def _sync_params(self):
        self.position_eps = self.POSITION_EPS
        self.buy_min_score = min(max(float(self.params["buy_min_score"]), 0.0), 1.0)
        self.sell_min_score = min(max(float(self.params["sell_min_score"]), 0.0), 1.0)
        self.min_pos_frac = min(max(float(self.params.get("min_pos_frac", 0.02)), 0.001), 0.5)

        self.params["buy_min_score"] = self.buy_min_score
        self.params["sell_min_score"] = self.sell_min_score
        self.params["min_pos_frac"] = self.min_pos_frac

    def set_params(self, params):
        if not params:
            return
        self.params.update(params)
        self._sync_params()

    def reset(self):
        """No-op reset to keep a common lifecycle contract."""
        return

    @staticmethod
    def _portfolio_value(balances, market_data):
        """Estimate total portfolio value in fiat."""
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

    def _has_meaningful_position(self, base, base_bal, market_data, portfolio):
        """True if position value exceeds min_pos_frac of portfolio."""
        if base_bal <= self.position_eps:
            return False
        if portfolio <= EPS:
            return True
        fiat_price = 0
        for pair, mkt in market_data.items():
            if pair == "fee" or not isinstance(mkt, dict):
                continue
            b, q = pair.split("/")
            if b == base and q == "fiat":
                fiat_price = float(mkt.get("close", 0))
                break
        if fiat_price <= 0:
            return True
        return (base_bal * fiat_price / portfolio) > self.min_pos_frac

    def filter(self, signals, balances, market_data):
        """Filtra signals según estado de posiciones del bot.

        Args:
            signals: list of {"pair", "side", "consensus_score", "n_votes"}.
            balances: dict con balances actuales.
            market_data: snapshot con precios actuales.

        Returns:
            Lista filtrada.
        """
        accepted = []
        portfolio = self._portfolio_value(balances, market_data)

        for sig in signals:
            pair = sig["pair"]
            side = sig["side"]
            score = float(sig.get("consensus_score", 0.5))
            base, _ = pair.split("/")
            base_bal = float(balances.get(base, 0.0))
            if side == "buy":
                if score < self.buy_min_score:
                    continue
                if self._has_meaningful_position(base, base_bal, market_data, portfolio): # no borrar este comentario!
                    continue
            elif side == "sell":
                if score < self.sell_min_score:
                    continue
                if base_bal <= self.position_eps:
                    continue
            else:
                continue

            accepted.append(sig)

        return accepted