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


def reset():
    """Reset all strategy components to a clean per-backtest state."""
    for strat in _strategies.values():
        if hasattr(strat, "reset"):
            strat.reset()
    if hasattr(_coordinator, "reset"):
        _coordinator.reset()
    if hasattr(_risk_mgr, "reset"):
        _risk_mgr.reset()


def set_params(params):
    """Apply optimized params to strategy components.

    Expected shape:
        {
            "estrategias": {"estrategia_momentum": {...}, ...},
            "coordinador": {...},
            "gestion_riesgo": {...}
        }
    """
    if not params:
        return

    strat_params = params.get("estrategias", {})
    for name, cfg in strat_params.items():
        strat = _strategies.get(name)
        if strat is not None and hasattr(strat, "set_params"):
            strat.set_params(cfg)

    coord_params = params.get("coordinador")
    if coord_params and hasattr(_coordinator, "set_params"):
        _coordinator.set_params(coord_params)

    risk_params = params.get("gestion_riesgo")
    if risk_params and hasattr(_risk_mgr, "set_params"):
        _risk_mgr.set_params(risk_params)


def get_hyperparams_template():
    """Return current tunable params from all strategy components."""
    estrategias = {}
    for name, strat in _strategies.items():
        if hasattr(strat, "params") and isinstance(strat.params, dict):
            estrategias[name] = dict(strat.params)

    coordinador = {}
    if hasattr(_coordinator, "params") and isinstance(_coordinator.params, dict):
        coordinador = dict(_coordinator.params)

    gestion_riesgo = {}
    if hasattr(_risk_mgr, "params") and isinstance(_risk_mgr.params, dict):
        gestion_riesgo = dict(_risk_mgr.params)

    return {
        "estrategias": estrategias,
        "coordinador": coordinador,
        "gestion_riesgo": gestion_riesgo,
    }

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
    filtered = _coordinator.filter(blended, balances, market_data)

    # 4. Size each accepted signal
    actions = _risk_mgr.size(filtered, balances, market_data)

    """if actions:
        print('baalances:'  , balances)
        print(all_signals)
        print(actions)

        if all_signals['estrategia_scalping_momentum']:
            time.sleep(5)"""

    return actions if actions else None