"""Strategy entry point required by the exchange engine.

Orchestrates the full signal pipeline:
  1. Collect raw signals from all strategies in estrategias/
  2. Blend into consensus signals  (estrategia_blend)
  3. Filter through position rules  (coordinador)
  4. Size trades via risk manager   (gestion_riesgo)
"""

import importlib
import os

from .estrategia_blend import SignalBlender
from .coordinador import PositionCoordinator
from .gestion_riesgo import RiskManager


def _load_strategies():
    """Dynamically import every Strategy class from estrategias/."""
    strategies = {}
    pkg_dir = os.path.join(os.path.dirname(__file__), "estrategias")
    for fname in sorted(os.listdir(pkg_dir)):
        if fname.startswith("_") or not fname.endswith(".py"):
            continue
        mod_name = fname[:-3]
        mod = importlib.import_module(
            f".estrategias.{mod_name}", package="strategy"
        )
        if hasattr(mod, "Strategy"):
            strategies[mod_name] = mod.Strategy()
    return strategies


_strategies = _load_strategies()
_blender = SignalBlender()
_coordinator = PositionCoordinator()
_risk_mgr = RiskManager()


def on_data(market_data, balances):
    """API required by the exchange engine."""
    # 1. Raw signals from every strategy
    all_signals = {}
    for name, strat in _strategies.items():
        sigs = strat.on_data(market_data, balances)
        all_signals[name] = sigs if sigs else []

    # 2. Blend → one consensus signal per pair
    blended = _blender.combine(all_signals, market_data)

    # 3. Filter through position / cooldown rules
    filtered = _coordinator.filter(blended, balances)

    # 4. Size each accepted signal
    actions = _risk_mgr.size(filtered, balances, market_data)

    return actions if actions else None