"""CLI: python -m exchange.engine path/to/submission.tgz"""
import argparse
import math
import importlib
import json
import os
import sys
import tarfile
import tempfile
from collections import defaultdict
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm

try:
    import optuna
except Exception:
    optuna = None

# --- Constantes y Funciones Matemáticas (NUEVO) --------------------------
MINUTES_PER_YEAR = 365 * 24 * 60
ANNUALIZATION_FACTOR = np.sqrt(MINUTES_PER_YEAR)
EPSILON = 1e-9
DEFAULT_RISK_FREE = 0.0

def sharpe(returns: np.ndarray, risk_free: float = DEFAULT_RISK_FREE):
    if len(returns) == 0: return 0.0
    excess = returns - risk_free / MINUTES_PER_YEAR
    return ANNUALIZATION_FACTOR * excess.mean() / (excess.std(ddof=1) + EPSILON)

def max_drawdown(equity: np.ndarray):
    if len(equity) == 0: return 0.0
    cummax = np.maximum.accumulate(equity)
    dd = (equity - cummax) / cummax
    return dd.min()

def score_from_metrics(metrics: dict) -> tuple[float, dict[str, float]]:
    """Compute Kaggle score and expose each component for reporting."""
    if metrics.get("trade_count", 0) == 0:
        return float("-inf"), {
            "sharpe_contribution": 0.0,
            "drawdown_penalty": 0.0,
            "turnover_penalty": 0.0,
        }
    sharpe_component = 0.7 * metrics["sharpe"]
    drawdown_component = 0.2 * abs(metrics["max_dd"])
    turnover_component = 0.1 * (metrics["turnover"] / 1e6)
    score = sharpe_component - drawdown_component - turnover_component
    return score, {
        "sharpe_contribution": sharpe_component,
        "drawdown_penalty": drawdown_component,
        "turnover_penalty": turnover_component,
    }

def _reload_strategy_module(submission_dir: Path):
    """Reload strategy package from scratch to avoid cross-trial state bleed."""
    submission_path = str(submission_dir)
    if submission_path not in sys.path:
        sys.path.insert(0, submission_path)

    for mod_name in list(sys.modules.keys()):
        if mod_name == "strategy" or mod_name.startswith("strategy."):
            del sys.modules[mod_name]

    return importlib.import_module("strategy.main")


_STRATEGY_MODULE_CACHE = {}


def _get_strategy_module(submission_dir: Path):
    """Get cached strategy module for a submission path."""
    cache_key = str(submission_dir.resolve())
    if cache_key not in _STRATEGY_MODULE_CACHE:
        _STRATEGY_MODULE_CACHE[cache_key] = _reload_strategy_module(submission_dir)
    return _STRATEGY_MODULE_CACHE[cache_key]


def _flatten_numeric_params(obj, path=()):
    """Flatten nested numeric params to {(path_tuple): value}."""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(_flatten_numeric_params(v, path + (str(k),)))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out[path] = obj
    return out


def _set_nested_value(dst, path, value):
    cur = dst
    for key in path[:-1]:
        if key not in cur or not isinstance(cur[key], dict):
            cur[key] = {}
        cur = cur[key]
    cur[path[-1]] = value


def _suggest_from_default(trial, flat_key, default):
    """Suggest a value around the default without hardcoding parameter names."""
    name = "__".join(flat_key)

    if isinstance(default, int):
        low = max(1, int(round(default * 0.5)))
        high = max(low + 1, int(round(default * 2.0)))
        return int(trial.suggest_int(name, low, high))

    val = float(default)
    abs_val = abs(val)
    if abs_val > 0 and abs_val < 1e-4:
        low = max(abs_val * 0.1, 1e-12)
        high = max(abs_val * 10.0, low * 10.0)
        sampled = trial.suggest_float(name, low, high, log=True)
    elif abs_val <= 1.0:
        low = max(1e-9, abs_val * 0.25)
        high = max(low * 1.5, min(1.0, abs_val * 4.0 if abs_val > 0 else 1.0))
        sampled = trial.suggest_float(name, low, high)
    else:
        low = max(1e-9, abs_val * 0.5)
        high = max(low * 1.5, abs_val * 2.0)
        sampled = trial.suggest_float(name, low, high)

    return sampled if val >= 0 else -sampled

# --- Core Engine ---------------------------------------------------------

class Trader:
    """Trader supporting multiple trading pairs and currencies."""

    def __init__(self, balances, fee):
        # Initialize balances for each currency
        self.balances = balances
        self.fee = fee

        # Track market prices for each pair
        self.prices = {
            "token_1/fiat": None,
            "token_2/fiat": None,
            "token_1/token_2": None
        }

        # First and last prices for reporting
        self.first_prices = {
            "token_1/fiat": None,
            "token_2/fiat": None,
            "token_1/token_2": None
        }

        # Store the first update timestamp for each pair
        self.first_update = {
            "token_1/fiat": False,
            "token_2/fiat": False,
            "token_1/token_2": False
        }

        # Track portfolio value history
        self.equity_history = []
        self.turnover = 0.0
        self.trade_count = 0
        self.total_fees_paid = 0.0  # Track total fees paid

    def update_market(self, pair, price_data):
        """Update market prices for a specific trading pair"""
        # Store the updated price
        self.prices[pair] = price_data["close"]

        # Store first price for each pair (for reporting)
        if not self.first_update[pair]:
            self.first_prices[pair] = price_data["close"]
            self.first_update[pair] = True

        # Calculate total portfolio value (in fiat)
        equity = self.calculate_portfolio_value()
        self.equity_history.append(equity)

    def calculate_portfolio_value(self):
        """Calculate total portfolio value in fiat currency"""
        value = self.balances["fiat"]

        # Add token_1 value if we have price data
        if self.prices["token_1/fiat"] is not None:
            value += self.balances["token_1"] * self.prices["token_1/fiat"]

        # Add token_2 value if we have price data
        if self.prices["token_2/fiat"] is not None:
            value += self.balances["token_2"] * self.prices["token_2/fiat"]
        # If token_2/fiat price not available but token_1/fiat and token_1/token_2 are available
        elif self.prices["token_1/fiat"] is not None and self.prices["token_1/token_2"] is not None:
            token2_value_in_token1 = self.balances["token_2"] / self.prices["token_1/token_2"]
            value += token2_value_in_token1 * self.prices["token_1/fiat"]

        return value

    def execute(self, order):
        """Execute a trading order across any supported pair"""
        pair = order["pair"]  # e.g., "token_1/fiat"
        side = order["side"]  # "buy" or "sell"
        qty = float(order["qty"])

        # Split the pair into base and quote currencies
        base, quote = pair.split("/")

        # Get current price for the pair
        price = self.prices[pair]
        if price is None:
            return  # Can't trade without a price

        executed = False

        if side == "buy":
            # Calculate total cost including fee
            base_cost = qty * price
            fee_amount = base_cost * self.fee
            total_cost = base_cost + fee_amount

            # Check if we have enough of the quote currency
            if self.balances[quote] >= total_cost:
                # Deduct quote currency (e.g., fiat)
                self.balances[quote] -= total_cost

                # Add base currency (e.g., token_1)
                self.balances[base] += qty

                # Track turnover and fees
                self.turnover += total_cost
                self.total_fees_paid += fee_amount
                executed = True

        elif side == "sell":
            # Check if we have enough of the base currency
            if self.balances[base] >= qty:
                # Calculate proceeds after fee
                base_proceeds = qty * price
                fee_amount = base_proceeds * self.fee
                net_proceeds = base_proceeds - fee_amount

                # Add quote currency (e.g., fiat)
                self.balances[quote] += net_proceeds

                # Deduct base currency (e.g., token_1)
                self.balances[base] -= qty

                # Track turnover and fees
                self.turnover += base_proceeds
                self.total_fees_paid += fee_amount
                executed = True

        # Count successful trades
        if executed:
            self.trade_count += 1

def run_backtest(
    submission_dir: Path,
    combined_data: pd.DataFrame,
    fee: float,
    balances: dict[str, float],
    strategy_params: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Run a backtest with multiple trading pairs.

    Args:
        submission_dir: Path to the strategy directory
        combined_data: DataFrame containing market data for multiple pairs
        fee: Trading fee (in basis points, e.g., 2 = 0.02%)
        balances: Dictionary of {pair: amount} containing initial balances
    """
    strat_mod = _get_strategy_module(submission_dir)

    if strategy_params and hasattr(strat_mod, "set_params"):
        strat_mod.set_params(strategy_params)

    if hasattr(strat_mod, "reset"):
        strat_mod.reset()

    trader = Trader(balances, fee)

    # Record initial balances for display
    initial_balances = balances.copy()

    # Initialize prices with first data point for each pair
    combined_data.sort_values("timestamp", inplace=True)
    first_prices = {k: df.iloc[0]['close'] for k, df in combined_data.groupby("symbol")}

    # Calculate true initial portfolio value including all assets
    initial_portfolio_value = initial_balances["fiat"]
    if "token_1/fiat" in first_prices and initial_balances["token_1"] > 0:
        initial_portfolio_value += initial_balances["token_1"] * first_prices["token_1/fiat"]
    if "token_2/fiat" in first_prices and initial_balances["token_2"] > 0:
        initial_portfolio_value += initial_balances["token_2"] * first_prices["token_2/fiat"]

    trader.equity_history = [initial_portfolio_value]

    # Build a timestamp -> {pair -> row} index in one vectorized pass.
    rows_by_ts = defaultdict(dict)
    for rec in combined_data.to_dict("records"):
        rows_by_ts[rec["timestamp"]][rec["symbol"]] = rec

    timestamps = sorted(rows_by_ts)
    all_actions = []
    trade_id_counter = 0

    for timestamp in tqdm(timestamps, desc="Backtest"):
        market_data = {"fee": fee}
        for pair, row_dict in rows_by_ts[timestamp].items():
            market_data[pair] = row_dict
            trader.update_market(pair, row_dict)

        # Get strategy decision based on all available market data and current balances
        actions = strat_mod.on_data(market_data, balances)
        if actions is None:
            continue

        # Avoid per-row DataFrame concat; collect and build once at the end.
        for action in actions:
            trader.execute(action)
            trade_id_counter += 1
            all_actions.append({
                "id": str(trade_id_counter),
                "timestamp": timestamp,
                "pair": action["pair"],
                "side": action["side"],
                "qty": action["qty"],
            })

    result = pd.DataFrame(all_actions, columns=["id", "timestamp", "pair", "side", "qty"])

    # --- CÁLCULO DE MÉTRICAS AL FINALIZAR EL BUCLE (NUEVO) ---
    equity_curve = np.array(trader.equity_history)
    rets = np.diff(equity_curve) / equity_curve[:-1] if len(equity_curve) > 1 else np.array([])
    
    initial_equity = equity_curve[0] if len(equity_curve) > 0 else 0.0
    final_equity = equity_curve[-1] if len(equity_curve) > 0 else 0.0
    absolute_pnl = final_equity - initial_equity
    percentage_pnl = (absolute_pnl / initial_equity) * 100 if initial_equity > 0 else 0.0

    metrics = {
        "sharpe": sharpe(rets),
        "max_dd": max_drawdown(equity_curve),
        "turnover": trader.turnover,
        "absolute_pnl": absolute_pnl,
        "percentage_pnl": percentage_pnl,
        "initial_equity": initial_equity,
        "final_equity": final_equity,
        "trade_count": trader.trade_count,
        "total_fees_paid": trader.total_fees_paid,
    }

    return result, metrics


# --- CLI --------------------------------------------------------------

def main(args: argparse.Namespace):
    if not os.path.exists(args.data):
        print(f"Error: {args.data} file doesn't exist.")
        sys.exit(1)

    data_df = pd.read_csv(args.data)
    base_balances = {
        "fiat": args.fiat_balance,
        "token_1": args.token1_balance,
        "token_2": args.token2_balance,
    }

    data_dir = Path(args.data).parent
    params_file = data_dir / "best_params.json"

    with tempfile.TemporaryDirectory() as td:
        with tarfile.open(args.submission) as tar:
            tar.extractall(path=td, filter="data")

        submission_dir = Path(td) / "submission"
        best_params = None

        if args.optimize:
            if optuna is None:
                print("Error: Optuna no está disponible. Instala dependencias y vuelve a intentar.")
                sys.exit(1)

            print(f"\nIniciando optimización de hiperparámetros ({args.optimize_trials} trials)...")

            template_mod = _get_strategy_module(submission_dir)
            if not hasattr(template_mod, "get_hyperparams_template"):
                print("Error: strategy.main no expone get_hyperparams_template().")
                sys.exit(1)

            param_template = template_mod.get_hyperparams_template()
            flat_template = _flatten_numeric_params(param_template)
            if not flat_template:
                print("Error: No se encontraron hiperparámetros numéricos optimizables.")
                sys.exit(1)

            def objective(trial):
                trial_params = {}
                for key_path, default in flat_template.items():
                    sampled = _suggest_from_default(trial, key_path, default)
                    _set_nested_value(trial_params, key_path, sampled)

                _, trial_metrics = run_backtest(
                    submission_dir,
                    data_df.copy(),
                    args.fee / 10000,
                    base_balances.copy(),
                    strategy_params=trial_params,
                )
                trial_score, _ = score_from_metrics(trial_metrics)
                return trial_score

            #sampler = optuna.samplers.RandomSampler()
            sampler = optuna.samplers.CmaEsSampler()

            study = optuna.create_study(
                direction="maximize",
                sampler=sampler
            )

            study.optimize(objective, n_trials=args.optimize_trials, n_jobs=1, catch=(Exception,))

            completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            if not completed_trials:
                print("Error: Ningun trial de Optuna se completo exitosamente.")
                print("Revisa los warnings de los trials fallidos para identificar la causa raiz.")
                sys.exit(1)

            best_params = {}
            for key_path, default in flat_template.items():
                trial_key = "__".join(key_path)
                sampled = study.best_params.get(trial_key, default)
                _set_nested_value(best_params, key_path, sampled)

            with open(params_file, "w") as f:
                json.dump(best_params, f, indent=2)
            print(f"Hiperparámetros guardados en: {params_file}")
            print("Mejores hiperparámetros encontrados:")
            print(json.dumps(best_params, indent=2))

        else:
            if params_file.exists():
                with open(params_file) as f:
                    best_params = json.load(f)
                print(f"Cargando hiperparámetros desde: {params_file}")

        # Run final backtest (con mejores parámetros si optimize=True)
        res_df, metrics = run_backtest(
            submission_dir,
            data_df.copy(),
            args.fee / 10000,
            base_balances.copy(),
            strategy_params=best_params,
        )

        score, score_components = score_from_metrics(metrics)

        ordered_res = {
            "score": score,
            "score_components": score_components,
            "pnl": {
                "absolute": metrics["absolute_pnl"],
                "percentage": metrics["percentage_pnl"],
                "initial_equity": metrics["initial_equity"],
                "final_equity": metrics["final_equity"]
            },
            "trading": {
                "sharpe": metrics["sharpe"],
                "max_drawdown": metrics["max_dd"],
                "turnover": metrics["turnover"],
                "trade_count": metrics["trade_count"],
                "total_fees_paid": metrics["total_fees_paid"],
            }
        }
        if best_params is not None:
            ordered_res["optimized"] = True
            ordered_res["best_params"] = best_params

        # Formatear números para una mejor visualización JSON
        def format_numbers(obj):
            if isinstance(obj, dict):
                return {k: format_numbers(v) for k, v in obj.items()}
            elif isinstance(obj, (float, np.floating)):
                return round(float(obj), 4)
            return obj

        print("\n" + "="*45)
        print("📊 RESULTADOS DEL BACKTEST (KAGGLE SCORE)")
        print("="*45)
        print(json.dumps(format_numbers(ordered_res), indent=2))
        print("="*45)

        # Save resulting trades to csv
        res_df.to_csv(args.output, index=False)
        print(f"\n✅ Trades guardados con éxito en: {args.output}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("submission", help="Path to submission file")
    p.add_argument("--data", help="Path to data file", default="test.csv")
    p.add_argument("--output", help="Path to output file", default="submission.csv")
    p.add_argument("--token1_balance", help="Initial token_1 balance", type=float, default=0.0)
    p.add_argument("--token2_balance", help="Initial token_2 balance", type=float, default=0.0)
    p.add_argument("--fiat_balance", help="Initial fiat balance", type=float, default=10000.0)
    p.add_argument("--fee", help="Trading fee (in basis points, e.g., 3 = 0.03% = 0.0003)", type=float, default=3.0)
    p.add_argument("--optimize", help="Optimize momentum hyperparameters with Optuna", action="store_true")
    p.add_argument("--optimize-trials", help="Number of Optuna trials", type=int, default=10)
    args = p.parse_args()

    main(args)