"""Strategy class wrapper for single-strategy mode.

Exports only `Strategy`, so `strategy.main` can do:
    from .strategy import Strategy
"""

import importlib
import json
from pathlib import Path
import sys

try:
    from .coordinador import PositionCoordinator
    from .gestion_riesgo import RiskManager
except Exception:
    from coordinador import PositionCoordinator
    from gestion_riesgo import RiskManager


class Strategy:
    """Thin adapter around `estrategia_unificada.Strategy`."""

    _STRATEGY_NAME = "estrategia_unificada"

    def __init__(self):
        self._load_weights = True
        self._weights_loaded = False
        self._weights_status_logged = False

        mod = None
        for mod_name in (
            ".estrategia_unificada",
            "strategy.estrategia_unificada",
            "estrategia_unificada",
            ".estrategias.estrategia_unificada",
            "strategy.estrategias.estrategia_unificada",
            "estrategias.estrategia_unificada",
        ):
            try:
                if mod_name.startswith("."):
                    mod = importlib.import_module(mod_name, package=__package__)
                else:
                    mod = importlib.import_module(mod_name)
                break
            except Exception:
                continue

        if mod is None or not hasattr(mod, "Strategy"):
            raise ImportError("No se pudo importar estrategia_unificada.Strategy")

        self._strategy = mod.Strategy()
        self._coordinator = PositionCoordinator()
        self._risk = RiskManager()
        self._bind_main_api()

    @staticmethod
    def _weights_candidates():
        strategy_dir = Path(__file__).resolve().parent
        return [
            strategy_dir / "best_params.json",
            Path("strategy") / "best_params.json",
            Path("data") / "best_params.json",
            Path("best_params.json"),
        ]

    def _set_load_weights(self, load_weights=True):
        """Internal toggle to enable/disable loading persisted JSON weights."""
        self._load_weights = bool(load_weights)
        if self._load_weights and not self._weights_loaded:
            self._weights_status_logged = False

    def set_load_weights(self, load_weights=True):
        """Public toggle used by training flow in exchange.trade."""
        self._set_load_weights(load_weights)

    def _extract_strategy_params(self, params):
        """Normalize different parameter payload shapes to strategy-only params."""
        if not isinstance(params, dict):
            return None

        if "estrategias" in params:
            estrategias = params.get("estrategias") or {}
            return estrategias.get(self._STRATEGY_NAME)

        if self._STRATEGY_NAME in params:
            maybe_cfg = params.get(self._STRATEGY_NAME)
            if isinstance(maybe_cfg, dict):
                return maybe_cfg

        return params

    def _maybe_load_weights_from_json(self):
        """Load saved params from JSON once, only when enabled."""
        if self._weights_loaded:
            return

        if not self._load_weights:
            if not self._weights_status_logged:
                print("[Strategy] No se cargan parametros: load_weights desactivado.")
                self._weights_status_logged = True
            return

        candidates = self._weights_candidates()

        for path in candidates:
            if not path.exists():
                continue
            try:
                with path.open("r", encoding="utf-8") as f:
                    payload = json.load(f)
            except Exception:
                continue

            if not self._weights_status_logged:
                print(f"[Strategy] Cargando parametros desde: {path}")
                self._weights_status_logged = True
            self.set_params(payload)
            self._weights_loaded = True
            return

        if not self._weights_status_logged:
            print("[Strategy] No se cargan parametros: best_params.json no encontrado.")
            self._weights_status_logged = True

    def _bind_main_api(self):
        """Expose optimization hooks on `strategy.main` at import time.

        `exchange.trade` expects module-level functions in `strategy.main`.
        Since `main.py` only instantiates this class, we publish those
        callables dynamically from here.
        """
        main_mod = sys.modules.get("strategy.main")
        if main_mod is None:
            return
        setattr(main_mod, "reset", self.reset)
        setattr(main_mod, "set_params", self.set_params)
        setattr(main_mod, "set_load_weights", self.set_load_weights)
        setattr(main_mod, "get_hyperparams_template", self.get_hyperparams_template)

    def reset(self):
        # Reset weight-loading state between backtest runs/trials.
        self._weights_loaded = False
        self._weights_status_logged = False

        if hasattr(self._strategy, "reset"):
            self._strategy.reset()
        if hasattr(self._coordinator, "reset"):
            self._coordinator.reset()
        if hasattr(self._risk, "reset"):
            self._risk.reset()

    def set_params(self, params):
        if not params:
            return

        strategy_params = self._extract_strategy_params(params)

        if strategy_params and hasattr(self._strategy, "set_params"):
            self._strategy.set_params(strategy_params)

        coordinator_params = params.get("coordinador", {}) if isinstance(params, dict) else {}
        if hasattr(self._coordinator, "set_params"):
            self._coordinator.set_params(coordinator_params)

        risk_params = params.get("gestion_riesgo", {}) if isinstance(params, dict) else {}
        if hasattr(self._risk, "set_params"):
            self._risk.set_params(risk_params)

    @staticmethod
    def _params_without_fixed(instance):
        if not hasattr(instance, "params") or not isinstance(instance.params, dict):
            return {}

        fixed_params = getattr(instance, "FIXED_PARAMS", {})
        fixed_keys = set(fixed_params.keys()) if isinstance(fixed_params, dict) else set()
        return {k: v for k, v in dict(instance.params).items() if k not in fixed_keys}

    @staticmethod
    def _sanitize_actions(actions):
        """Keep only exchange-compatible fields in outgoing orders."""
        if not actions:
            return []

        sanitized = []
        for action in actions:
            if not isinstance(action, dict):
                continue
            if not all(k in action for k in ("pair", "side", "qty")):
                continue
            sanitized.append(
                {
                    "pair": action["pair"],
                    "side": action["side"],
                    "qty": action["qty"],
                }
            )
        return sanitized

    def get_hyperparams_template(self):
        self._maybe_load_weights_from_json()
        strategy_params = self._params_without_fixed(self._strategy)
        coordinator_params = self._params_without_fixed(self._coordinator)
        risk_params = self._params_without_fixed(self._risk)

        return {
            "estrategias": {self._STRATEGY_NAME: strategy_params},
            "coordinador": coordinator_params,
            "gestion_riesgo": risk_params,
        }

    def on_data(self, market_data, balances):
        self._maybe_load_weights_from_json()
        signals = self._strategy.on_data(market_data, balances)
        if not signals:
            return None

        bypass_actions = []
        pipeline_signals = []
        for signal in signals:
            if isinstance(signal, dict) and signal.get("bypass_pipeline") is True:
                bypass_actions.append(signal)
            else:
                pipeline_signals.append(signal)

        filtered_signals = []
        if pipeline_signals:
            filtered_signals = self._coordinator.filter(pipeline_signals, balances, market_data)

        sized_orders = []
        if filtered_signals:
            sized_orders = self._risk.size(filtered_signals, balances, market_data)

        actions = bypass_actions + sized_orders
        actions = self._sanitize_actions(actions)

        return actions if actions else None