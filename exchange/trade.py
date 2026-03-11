"""CLI: python -m exchange.trade path/to/submission.tgz"""
import argparse
import importlib
import json
import os
import sys
import tarfile
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import optuna
except Exception:
    optuna = None

# --- Helpers -------------------------------------------------------------

MINUTES_PER_YEAR = 365 * 24 * 60
ANNUALIZATION_FACTOR = np.sqrt(MINUTES_PER_YEAR)
EPSILON = 1e-9
DEFAULT_RISK_FREE = 0.0
DEFAULT_FEE = 0.0003
BOOTSTRAP_SIMULATIONS = 200
BOOTSTRAP_SEED = 42

def sharpe(returns: np.ndarray, risk_free: float = DEFAULT_RISK_FREE):
    if len(returns) == 0:
        return 0.0
    excess = returns - risk_free / MINUTES_PER_YEAR
    return ANNUALIZATION_FACTOR * excess.mean() / (excess.std(ddof=1) + EPSILON)


def max_drawdown(equity: np.ndarray):
    if len(equity) == 0:
        return 0.0
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


def _prepare_backtest_data(data_dict: dict) -> dict:
    """Build a reusable timestamp index to avoid repeated groupby/iterrows."""
    first_prices = {}
    all_data = []

    for pair, df in data_dict.items():
        if not df.empty:
            first_prices[pair] = df.iloc[0]["close"]
        df_copy = df.copy()
        df_copy["pair"] = pair
        all_data.append(df_copy)

    combined_data = pd.concat(all_data)
    combined_data = combined_data.sort_values("timestamp")

    timeline = []
    current_ts = None
    current_rows = []
    for row_data in combined_data.to_dict("records"):
        ts = row_data["timestamp"]
        if current_ts is None:
            current_ts = ts
        if ts != current_ts:
            timeline.append((current_ts, current_rows))
            current_ts = ts
            current_rows = []
        current_rows.append(row_data)

    if current_rows:
        timeline.append((current_ts, current_rows))

    return {
        "first_prices": first_prices,
        "timeline": timeline,
    }


def _clean_start_balances(base_balances: dict, first_prices: dict) -> dict:
    """Convert initial token holdings into fiat and zero-out token balances."""
    fiat_value = float(base_balances.get("fiat", 0.0))
    token_1_balance = float(base_balances.get("token_1", 0.0))
    token_2_balance = float(base_balances.get("token_2", 0.0))

    token1fiat = first_prices.get("token_1/fiat")
    token2fiat = first_prices.get("token_2/fiat")
    token1token2 = first_prices.get("token_1/token_2")

    if token1fiat is not None and token_1_balance > 0.0:
        fiat_value += token_1_balance * float(token1fiat)

    if token2fiat is not None and token_2_balance > 0.0:
        fiat_value += token_2_balance * float(token2fiat)
    elif token1fiat is not None and token1token2 is not None and token_2_balance > 0.0:
        # token_1/token_2 means token_1 per token_2, so token_2 in fiat is token_1/fiat divided by token_1/token_2.
        token2_in_token1 = token_2_balance / float(token1token2)
        fiat_value += token2_in_token1 * float(token1fiat)

    return {
        "fiat": fiat_value,
        "token_1": 0.0,
        "token_2": 0.0,
    }


def _bootstrap_sharpe_stats(
    returns: np.ndarray,
    simulations: int = BOOTSTRAP_SIMULATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """Return mean/std of bootstrapped Sharpe values from full-backtest returns."""
    if len(returns) == 0:
        return 0.0, 0.0

    rng = np.random.default_rng(seed)
    mc_sharpes = []
    for _ in range(simulations):
        sample = rng.choice(returns, size=len(returns), replace=True)
        mc_sharpes.append(sharpe(sample))

    sharpes = np.array(mc_sharpes)
    return float(sharpes.mean()), float(sharpes.std(ddof=1))


# --- Core Engine ---------------------------------------------------------

class Trader:
    """Trader supporting multiple trading pairs and currencies."""

    def __init__(self):
        # Initialize balances for each currency
        self.balances = {"fiat": 0.0, "token_1": 0.0, "token_2": 0.0}

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

        # Trading fee
        self.fee = DEFAULT_FEE

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
    data_dict: dict,
    trader=None,
    strategy_params: dict | None = None,
    prepared_data: dict | None = None,
    force_strategy_reload: bool = False,
    load_weights: bool = True,
):
    """Run a backtest with multiple trading pairs."""
    strat_mod = _reload_strategy_module(submission_dir) if force_strategy_reload else _get_strategy_module(submission_dir)

    if hasattr(strat_mod, "set_load_weights"):
        strat_mod.set_load_weights(load_weights)

    if strategy_params and hasattr(strat_mod, "set_params"):
        strat_mod.set_params(strategy_params)

    if hasattr(strat_mod, "reset"):
        strat_mod.reset()

    # Initialize multi-asset trader if not provided
    if trader is None:
        trader = Trader()

    # Record initial balances for display
    initial_balances = trader.balances.copy()

    if prepared_data is None:
        prepared_data = _prepare_backtest_data(data_dict)

    # Initialize prices with first data point for each pair
    first_prices = prepared_data["first_prices"]

    # Calculate true initial portfolio value including all assets
    initial_portfolio_value = initial_balances["fiat"]
    if "token_1/fiat" in first_prices and initial_balances["token_1"] > 0:
        initial_portfolio_value += initial_balances["token_1"] * first_prices["token_1/fiat"]
    if "token_2/fiat" in first_prices and initial_balances["token_2"] > 0:
        initial_portfolio_value += initial_balances["token_2"] * first_prices["token_2/fiat"]

    # Start equity history with correct initial portfolio value
    trader.equity_history = [initial_portfolio_value]

    # Process data timestamp by timestamp
    all_actions = []
    trade_id_counter = 0
    timeline_iter = prepared_data["timeline"]
    for timestamp, rows_at_timestamp in tqdm(timeline_iter, desc="Backtest", dynamic_ncols=True):
        market_data = {}
        for row_data in rows_at_timestamp:
            pair = row_data["pair"]
            trader.update_market(pair, row_data)
            row_data["fee"] = trader.fee
            market_data[pair] = row_data

        actions = strat_mod.on_data(market_data, trader.balances)

        if actions is None:
            continue

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

    # Calculate performance metrics
    equity_curve = np.array(trader.equity_history)
    rets = np.diff(equity_curve) / equity_curve[:-1] if len(equity_curve) > 1 else np.array([])
    initial_equity = equity_curve[0] if len(equity_curve) > 0 else 0.0
    final_equity = equity_curve[-1] if len(equity_curve) > 0 else 0.0
    absolute_pnl = final_equity - initial_equity
    percentage_pnl = (absolute_pnl / initial_equity) * 100 if initial_equity > 0 else 0.0

    # Store current prices for result reporting
    current_prices = {
        "token_1/fiat": trader.prices.get("token_1/fiat"),
        "token_2/fiat": trader.prices.get("token_2/fiat"),
        "token_1/token_2": trader.prices.get("token_1/token_2")
    }

    # Calculate what the value would be if we had simply held the initial assets
    hodl_value = initial_balances["fiat"]
    if trader.prices["token_1/fiat"] is not None and initial_balances["token_1"] > 0:
        hodl_value += initial_balances["token_1"] * trader.prices["token_1/fiat"]
    if trader.prices["token_2/fiat"] is not None and initial_balances["token_2"] > 0:
        hodl_value += initial_balances["token_2"] * trader.prices["token_2/fiat"]

    # Calculate HODL performance
    hodl_absolute_pnl = hodl_value - initial_equity
    hodl_percentage_pnl = (hodl_absolute_pnl / initial_equity) * 100 if initial_equity > 0 else 0.0

    return {
        "sharpe": sharpe(rets),
        "max_dd": max_drawdown(equity_curve),
        "turnover": trader.turnover,
        "absolute_pnl": absolute_pnl,
        "percentage_pnl": percentage_pnl,
        "initial_equity": initial_equity,
        "final_equity": final_equity,
        "initial_balances": initial_balances,
        "final_balances": trader.balances,
        "initial_fiat_value": initial_equity,
        "final_fiat_value": final_equity,
        "total_fees_paid": trader.total_fees_paid,
        "trade_count": trader.trade_count,
        "current_prices": current_prices,
        "hodl_absolute_pnl": hodl_absolute_pnl,
        "hodl_percentage_pnl": hodl_percentage_pnl,
        "hodl_value": hodl_value,
        "equity_curve": equity_curve.tolist(),
        "trades": all_actions,
    }


# --- CLI --------------------------------------------------------------

def main(args: argparse.Namespace):
    data_dict = {}

    if args.data:
        if not os.path.exists(args.data):
            print(f"Error: {args.data} file doesn't exist.")
            sys.exit(1)

        csv_df = pd.read_csv(args.data)
        pair_col = None
        if "symbol" in csv_df.columns:
            pair_col = "symbol"
        elif "pair" in csv_df.columns:
            pair_col = "pair"

        if pair_col is None:
            print("Error: CSV data must include a 'symbol' or 'pair' column.")
            sys.exit(1)

        for pair, pair_df in csv_df.groupby(pair_col):
            df = pair_df.copy()
            if pair_col != "pair":
                df["pair"] = pair
            data_dict[pair] = df
    else:
        if os.path.exists(args.token1fiat):
            data_dict["token_1/fiat"] = pd.read_parquet(args.token1fiat)

        if os.path.exists(args.token2fiat):
            data_dict["token_2/fiat"] = pd.read_parquet(args.token2fiat)

        if os.path.exists(args.token1token2):
            data_dict["token_1/token_2"] = pd.read_parquet(args.token1token2)

    if not data_dict:
        print("Error: No data files found. Please provide at least one valid data file.")
        sys.exit(1)

    base_balances = {
        "fiat": args.fiat_balance,
        "token_1": args.token1_balance,
        "token_2": args.token2_balance,
    }

    strategy_dir = Path(__file__).resolve().parents[1] / "strategy"
    params_file = strategy_dir / "best_params.json"

    prepared_backtest_data = _prepare_backtest_data(data_dict)
    with tempfile.TemporaryDirectory() as td:
        with tarfile.open(args.submission) as tar:
            tar.extractall(path=td, filter="data")

        submission_dir = Path(td) / "submission"
        best_params = None

        if args.optimize:
            if optuna is None:
                print("Error: Optuna no esta disponible. Instala dependencias y vuelve a intentar.")
                sys.exit(1)

            print(f"\nIniciando optimizacion de hiperparametros ({args.optimize_trials} trials)...")

            template_mod = _get_strategy_module(submission_dir)
            if hasattr(template_mod, "set_load_weights"):
                template_mod.set_load_weights(False)
            if not hasattr(template_mod, "get_hyperparams_template"):
                print("Error: strategy.main no expone get_hyperparams_template().")
                sys.exit(1)

            param_template = template_mod.get_hyperparams_template()
            flat_template = _flatten_numeric_params(param_template)
            if not flat_template:
                print("Error: No se encontraron hiperparametros numericos optimizables.")
                sys.exit(1)

            objective_balances = base_balances
            if args.optimize_clean_start:
                objective_balances = _clean_start_balances(
                    base_balances,
                    prepared_backtest_data["first_prices"],
                )
                print(
                    "Optuna clean start activado: "
                    f"fiat={objective_balances['fiat']:.2f}, token_1=0, token_2=0"

                )

            first_prices = prepared_backtest_data["first_prices"]
            tv_cleanup_fixed = 0.0
            if args.optimize_clean_start:
                tv_cleanup_fixed = 100 * first_prices["token_1/fiat"] + 10 * first_prices["token_2/fiat"]

            # Hard bounds for parameters that directly drive excess turnover.
            # These override _suggest_from_default to prevent Optuna from exploring
            # high-TV regions (e.g. very short cooldowns or very large trade budgets).
            _HARD_LIMITS = {
                "estrategias__estrategia_unificada__cd_arb":   (240, 1440),
                "estrategias__estrategia_unificada__cd_lag":   (240, 1440),
                "estrategias__estrategia_unificada__cd_mr":    (240, 1440),
                "estrategias__estrategia_unificada__cd_mom":   (180, 1440),
                "estrategias__estrategia_unificada__cd_panic": (240, 1440),
                "estrategias__estrategia_unificada__budget_arb":  (0.003, 0.020),
                "estrategias__estrategia_unificada__budget_lag":  (0.003, 0.020),
                "estrategias__estrategia_unificada__budget_mr":   (0.003, 0.020),
                "estrategias__estrategia_unificada__budget_mom":  (0.003, 0.020),
                "gestion_riesgo__max_trade_pct":   (0.002, 0.010),
                "gestion_riesgo__min_trade_pct":   (0.001, 0.006),
                "coordinador__sell_min_score":     (0.60,  0.95),
                "coordinador__buy_min_score":      (0.15,  0.55),
            }

            def objective(trial):
                # Aligned with Kaggle score: robust_sharpe, max_dd penalty, turnover penalty.
                trial_params = {}
                for key_path, default in flat_template.items():
                    key_str = "__".join(key_path)
                    if key_str in _HARD_LIMITS:
                        lo, hi = _HARD_LIMITS[key_str]
                        if isinstance(default, int):
                            sampled = trial.suggest_int(key_str, int(lo), int(hi))
                        else:
                            sampled = trial.suggest_float(key_str, lo, hi)
                    else:
                        sampled = _suggest_from_default(trial, key_path, default)
                    _set_nested_value(trial_params, key_path, sampled)

                trial_trader = Trader()
                trial_trader.balances = objective_balances.copy()
                trial_trader.fee = args.fee / 10000
                trial_trader.equity_history = []

                trial_metrics = run_backtest(
                    submission_dir,
                    data_dict,
                    trader=trial_trader,
                    strategy_params=trial_params,
                    prepared_data=prepared_backtest_data,
                    force_strategy_reload=True,
                    load_weights=False,
                )

                if trial_metrics.get("trade_count", 0) == 0:
                    return float("-inf")

                equity = np.array(trial_metrics["equity_curve"])
                all_rets = np.diff(equity) / equity[:-1] if len(equity) > 1 else np.array([])
                rets = all_rets[np.abs(all_rets) > 1e-7]
                sharpe_mean, sharpe_std = _bootstrap_sharpe_stats(rets)
                robust_sharpe = sharpe_mean - 0.5 * sharpe_std

                trial_max_dd = abs(max_drawdown(equity))
                trade_penalty = 0.01 * max(0, 20 - trial_metrics["trade_count"])
                tv_adj = trial_metrics["turnover"]
                if args.optimize_clean_start:
                    tv_adj += tv_cleanup_fixed
                return 0.7 * robust_sharpe - 0.2 * trial_max_dd - 0.1 * (tv_adj / 1e6) - trade_penalty

            sampler = optuna.samplers.CmaEsSampler()
            study = optuna.create_study(direction="maximize", sampler=sampler)
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

            strategy_dir.mkdir(parents=True, exist_ok=True)
            with open(params_file, "w", encoding="utf-8") as f:
                json.dump(best_params, f, indent=2)
            print(f"Hiperparametros guardados en: {params_file}")
            print("Mejores hiperparametros encontrados:")
            print(json.dumps(best_params, indent=2))

        else:
            if params_file.exists():
                with open(params_file, encoding="utf-8") as f:
                    best_params = json.load(f)
                print(f"Cargando hiperparametros desde: {params_file}")

        trader = Trader()
        trader.balances = base_balances.copy()
        trader.fee = args.fee / 10000
        trader.equity_history = []

        metrics = run_backtest(
            submission_dir,
            data_dict,
            trader=trader,
            strategy_params=best_params,
            prepared_data=prepared_backtest_data,
        )

        score, score_components = score_from_metrics(metrics)

        ordered_res = {
            "score": score,
            "pnl": {
                "absolute": metrics["absolute_pnl"],
                "percentage": metrics["percentage_pnl"],
                "initial_equity": metrics["initial_equity"],
                "final_equity": metrics["final_equity"],
            },
            "balances": {
                "initial": {
                    **metrics["initial_balances"],
                    "total_in_fiat": metrics["initial_fiat_value"],
                },
                "final": {
                    **metrics["final_balances"],
                    "total_in_fiat": metrics["final_fiat_value"],
                },
            },
            "prices": {
                "initial": {
                    "token_1/fiat": trader.first_prices.get("token_1/fiat"),
                    "token_2/fiat": trader.first_prices.get("token_2/fiat"),
                    "token_1/token_2": trader.first_prices.get("token_1/token_2"),
                },
                "final": metrics["current_prices"],
            },
            "trading": {
                "sharpe": metrics["sharpe"],
                "max_drawdown": metrics["max_dd"],
                "turnover": metrics["turnover"],
                "trade_count": metrics["trade_count"],
                "total_fees_paid": metrics["total_fees_paid"],
            },
            "hodl_pnl": {
                "absolute": metrics["hodl_absolute_pnl"],
                "percentage": metrics["hodl_percentage_pnl"],
                "initial_equity": metrics["initial_equity"],
                "final_equity": metrics["hodl_value"],
            },
            "score_components": score_components,
        }

        if best_params is not None:
            ordered_res["optimized"] = True
            ordered_res["best_params"] = best_params

        def format_numbers(obj):
            if isinstance(obj, dict):
                return {k: format_numbers(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [format_numbers(item) for item in obj]
            if isinstance(obj, (float, np.floating)):
                return round(float(obj), 4)
            return obj

        print("\n" + "=" * 45)
        print("RESULTADOS DEL BACKTEST (KAGGLE SCORE)")
        print("=" * 45)
        print(json.dumps(format_numbers(ordered_res), indent=2))
        print("=" * 45)

        # Save resulting trades to csv.
        # Relative output paths are resolved against the token1fiat data directory.
        output_path = Path(args.output)
        if not output_path.is_absolute():
            if args.data:
                output_path = Path(args.data).resolve().parent / output_path
            else:
                output_path = Path(args.token1fiat).resolve().parent / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        trades_df = pd.DataFrame(metrics.get("trades", []), columns=["id", "timestamp", "pair", "side", "qty"])
        trades_df.to_csv(output_path, index=False)
        print(f"\nTrades guardados con exito en: {output_path}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("submission", help="Path to submission file")
    p.add_argument("--data", help="Path to CSV data file", default="data/test.csv")
    p.add_argument("--token1fiat", help="Path to token_1/fiat data", default="data/token1fiat_1m.parquet")
    p.add_argument("--token2fiat", help="Path to token_2/fiat data", default="data/token2fiat_1m.parquet")
    p.add_argument("--token1token2", help="Path to token_1/token_2 data", default="data/token1token2_1m.parquet")
    p.add_argument("--output", help="Path to output file", default="submission.csv")
    p.add_argument("--token1_balance", help="Initial token_1 balance", type=float, default=0.0)
    p.add_argument("--token2_balance", help="Initial token_2 balance", type=float, default=0.0)
    p.add_argument("--fiat_balance", help="Initial fiat balance", type=float, default=10000.0)
    p.add_argument("--fee", help="Trading fee (in basis points, e.g., 3 = 0.03% = 0.0003)", type=float, default=3.0)
    p.add_argument("--optimize", help="Optimize momentum hyperparameters with Optuna", action="store_true")
    p.add_argument(
        "--optimize-clean-start",
        help="Use fiat-only clean balances during Optuna objective while keeping real balances for final backtest",
        action="store_true",
    )
    p.add_argument("--optimize-trials", help="Number of Optuna trials", type=int, default=10)
    cli_args = p.parse_args()

    main(cli_args)
