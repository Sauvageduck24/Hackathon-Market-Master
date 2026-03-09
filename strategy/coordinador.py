"""Position coordinator — simple alignment filter between signals and balances.

Regla fija:
    - Buy solo si no hay posicion en el activo base.
    - Sell solo si hay posicion en el activo base.
"""

EPS = 1e-9

class PositionCoordinator:
    """Filtro minimo y determinista con parametros configurables."""

    def __init__(self):
        self.params = {
            "position_eps": EPS,
            "buy_min_score": 0.0,
            "sell_min_score": 0.0,
        }
        self._sync_params()

    def _sync_params(self):
        self.position_eps = max(float(self.params["position_eps"]), 1e-12)
        self.buy_min_score = min(max(float(self.params["buy_min_score"]), 0.0), 1.0)
        self.sell_min_score = min(max(float(self.params["sell_min_score"]), 0.0), 1.0)

        self.params["position_eps"] = self.position_eps
        self.params["buy_min_score"] = self.buy_min_score
        self.params["sell_min_score"] = self.sell_min_score

    def set_params(self, params):
        if not params:
            return
        self.params.update(params)
        self._sync_params()

    def reset(self):
        """No-op reset to keep a common lifecycle contract."""
        return

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

        for sig in signals:
            pair = sig["pair"]
            side = sig["side"]
            score = float(sig.get("consensus_score", 0.5))
            base, _ = pair.split("/")
            has_pos = float(balances.get(base, 0.0)) > self.position_eps
            if side == "buy":
                if score < self.buy_min_score:
                    continue
                if has_pos:
                    continue
            elif side == "sell":
                if score < self.sell_min_score:
                    continue
                if not has_pos:
                    continue
            else:
                continue

            accepted.append(sig)

        return accepted