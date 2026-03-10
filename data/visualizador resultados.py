import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

MINUTES_PER_YEAR = 365 * 24 * 60
ANNUALIZATION_FACTOR = np.sqrt(MINUTES_PER_YEAR)
EPSILON = 1e-9
DEFAULT_RISK_FREE = 0.0
DEFAULT_FEE = 0.0003

PAIRS = ["token_1/fiat", "token_2/fiat", "token_1/token_2"]


def sharpe(returns: np.ndarray, risk_free: float = DEFAULT_RISK_FREE) -> float:
    if len(returns) == 0:
        return 0.0
    excess = returns - risk_free / MINUTES_PER_YEAR
    return float(ANNUALIZATION_FACTOR * excess.mean() / (excess.std(ddof=1) + EPSILON))


def max_drawdown(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return 0.0
    cummax = np.maximum.accumulate(equity)
    dd = (equity - cummax) / cummax
    return float(dd.min())


def score_from_metrics(metrics: dict) -> tuple[float, dict[str, float]]:
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

    return float(score), {
        "sharpe_contribution": float(sharpe_component),
        "drawdown_penalty": float(drawdown_component),
        "turnover_penalty": float(turnover_component),
    }


class ReplayTrader:
    def __init__(self, fee: float, fiat: float, token_1: float, token_2: float):
        self.fee = float(fee)
        self.balances = {
            "fiat": float(fiat),
            "token_1": float(token_1),
            "token_2": float(token_2),
        }
        self.prices = {pair: None for pair in PAIRS}
        self.turnover = 0.0
        self.total_fees_paid = 0.0
        self.trade_count = 0
        self.equity_history: list[float] = []

    def portfolio_value(self) -> float:
        value = self.balances["fiat"]
        if self.prices["token_1/fiat"] is not None:
            value += self.balances["token_1"] * self.prices["token_1/fiat"]

        if self.prices["token_2/fiat"] is not None:
            value += self.balances["token_2"] * self.prices["token_2/fiat"]
        elif self.prices["token_1/fiat"] is not None and self.prices["token_1/token_2"] is not None:
            token2_in_token1 = self.balances["token_2"] / self.prices["token_1/token_2"]
            value += token2_in_token1 * self.prices["token_1/fiat"]

        return float(value)

    def update_market(self, pair: str, close_price: float) -> None:
        self.prices[pair] = float(close_price)
        self.equity_history.append(self.portfolio_value())

    def execute(self, pair: str, side: str, qty: float) -> bool:
        if pair not in self.prices or self.prices[pair] is None:
            return False

        base, quote = pair.split("/")
        price = float(self.prices[pair])
        qty = float(qty)

        if side == "buy":
            base_cost = qty * price
            fee_amount = base_cost * self.fee
            total_cost = base_cost + fee_amount
            if self.balances[quote] >= total_cost:
                self.balances[quote] -= total_cost
                self.balances[base] += qty
                self.turnover += total_cost
                self.total_fees_paid += fee_amount
                self.trade_count += 1
                return True
            return False

        if side == "sell":
            if self.balances[base] >= qty:
                base_proceeds = qty * price
                fee_amount = base_proceeds * self.fee
                net_proceeds = base_proceeds - fee_amount
                self.balances[quote] += net_proceeds
                self.balances[base] -= qty
                self.turnover += base_proceeds
                self.total_fees_paid += fee_amount
                self.trade_count += 1
                return True
            return False

        return False


def load_best_params(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def simulate(market_df: pd.DataFrame, trades_df: pd.DataFrame, trader: ReplayTrader) -> dict:
    market_df = market_df.copy()
    trades_df = trades_df.copy()

    market_df["timestamp"] = pd.to_datetime(market_df["timestamp"])
    trades_df["timestamp"] = pd.to_datetime(trades_df["timestamp"])
    trades_df["side"] = trades_df["side"].str.lower()

    market_df = market_df.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    trades_df = trades_df.sort_values(["timestamp", "id"]).reset_index(drop=True)

    first_prices = {}
    for pair in PAIRS:
        pair_rows = market_df[market_df["symbol"] == pair]
        if not pair_rows.empty:
            first_prices[pair] = float(pair_rows.iloc[0]["close"])

    initial_equity = (
        trader.balances["fiat"]
        + trader.balances["token_1"] * first_prices.get("token_1/fiat", 0.0)
        + trader.balances["token_2"] * first_prices.get("token_2/fiat", 0.0)
    )
    trader.equity_history = [float(initial_equity)]

    trades_by_ts = {ts: grp for ts, grp in trades_df.groupby("timestamp", sort=False)}
    executed_rows = []

    grouped_market = list(market_df.groupby("timestamp", sort=True))
    for ts, group in tqdm(grouped_market, desc="Simulando mercado", unit="ts"):
        for row in group.itertuples(index=False):
            trader.update_market(str(row.symbol), float(row.close))

        if ts not in trades_by_ts:
            continue

        ts_trades = list(trades_by_ts[ts].itertuples(index=False))
        for tr in tqdm(ts_trades, desc=f"Ejecutando trades {ts}", unit="trade", leave=False):
            ok = trader.execute(str(tr.pair), str(tr.side), float(tr.qty))
            executed_rows.append(
                {
                    "id": str(tr.id),
                    "timestamp": ts,
                    "pair": str(tr.pair),
                    "side": str(tr.side),
                    "qty": float(tr.qty),
                    "executed": bool(ok),
                }
            )

    equity_curve = np.asarray(trader.equity_history, dtype=float)
    returns = np.diff(equity_curve) / equity_curve[:-1] if len(equity_curve) > 1 else np.asarray([])

    metrics = {
        "sharpe": sharpe(returns),
        "max_dd": max_drawdown(equity_curve),
        "turnover": float(trader.turnover),
        "trade_count": int(trader.trade_count),
        "total_fees_paid": float(trader.total_fees_paid),
        "initial_equity": float(equity_curve[0]) if len(equity_curve) > 0 else 0.0,
        "final_equity": float(equity_curve[-1]) if len(equity_curve) > 0 else 0.0,
        "final_balances": trader.balances,
        "equity_curve": equity_curve,
    }
    score, score_components = score_from_metrics(metrics)

    metrics["score"] = score
    metrics["score_components"] = score_components
    metrics["executed_trades"] = pd.DataFrame(executed_rows)
    return metrics


def plot_results(market_df: pd.DataFrame, executed_trades: pd.DataFrame, equity_curve: np.ndarray) -> None:
    market_df = market_df.copy()
    market_df["timestamp"] = pd.to_datetime(market_df["timestamp"])

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    pair_titles = {
        "token_1/fiat": "token_1/fiat (precio + operaciones)",
        "token_2/fiat": "token_2/fiat (precio + operaciones)",
        "token_1/token_2": "token_1/token_2 (precio)",
    }

    for idx, pair in enumerate(PAIRS):
        ax = axes[idx]
        pair_df = market_df[market_df["symbol"] == pair]
        ax.plot(pair_df["timestamp"], pair_df["close"], linewidth=1.0, label=pair)

        if not executed_trades.empty:
            pair_trades = executed_trades[(executed_trades["pair"] == pair) & (executed_trades["executed"])]
            buys = pair_trades[pair_trades["side"] == "buy"]
            sells = pair_trades[pair_trades["side"] == "sell"]

            if not buys.empty:
                buy_prices = []
                for t in buys["timestamp"]:
                    p = pair_df[pair_df["timestamp"] == t]["close"]
                    buy_prices.append(float(p.iloc[-1]) if not p.empty else np.nan)
                ax.scatter(buys["timestamp"], buy_prices, marker="^", s=35, label="buy")

            if not sells.empty:
                sell_prices = []
                for t in sells["timestamp"]:
                    p = pair_df[pair_df["timestamp"] == t]["close"]
                    sell_prices.append(float(p.iloc[-1]) if not p.empty else np.nan)
                ax.scatter(sells["timestamp"], sell_prices, marker="v", s=35, label="sell")

        ax.set_title(pair_titles[pair])
        ax.grid(alpha=0.2)
        ax.legend(loc="best")

    axes[3].plot(range(len(equity_curve)), equity_curve, linewidth=1.1, label="equity")
    axes[3].set_title("Equity curve")
    axes[3].set_xlabel("tick")
    axes[3].grid(alpha=0.2)
    axes[3].legend(loc="best")

    fig.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualizador de operaciones + metrica Kaggle")
    parser.add_argument("--market", default="data/test.csv", help="CSV de mercado (test.csv)")
    parser.add_argument("--trades", default="data/submission.csv", help="CSV de operaciones (submission.csv)")
    parser.add_argument("--params", default="data/best_params.json", help="JSON de mejores parametros")
    parser.add_argument("--fee", type=float, default=DEFAULT_FEE, help="Fee decimal (ej: 0.0003)")
    parser.add_argument("--fiat", type=float, default=10000.0, help="Saldo inicial fiat")
    parser.add_argument("--token1", type=float, default=100.0, help="Saldo inicial token_1")
    parser.add_argument("--token2", type=float, default=10.0, help="Saldo inicial token_2")
    args = parser.parse_args()

    market_path = Path(args.market)
    trades_path = Path(args.trades)
    params_path = Path(args.params)

    market_df = pd.read_csv(market_path)
    trades_df = pd.read_csv(trades_path)
    best_params = load_best_params(params_path)

    trader = ReplayTrader(
        fee=args.fee,
        fiat=args.fiat,
        token_1=args.token1,
        token_2=args.token2,
    )

    metrics = simulate(market_df, trades_df, trader)
    executed_df = metrics["executed_trades"]

    plot_results(market_df, executed_df, metrics["equity_curve"])

    requested_trades = int(len(trades_df))
    executed_trades = int(executed_df["executed"].sum()) if not executed_df.empty else 0
    failed_trades = requested_trades - executed_trades

    summary = {
        "score": float(metrics["score"]),
        "score_components": metrics["score_components"],
        "trading": {
            "sharpe": float(metrics["sharpe"]),
            "max_drawdown": float(metrics["max_dd"]),
            "turnover": float(metrics["turnover"]),
            "trade_count": int(metrics["trade_count"]),
            "total_fees_paid": float(metrics["total_fees_paid"]),
        },
        "equity": {
            "initial": float(metrics["initial_equity"]),
            "final": float(metrics["final_equity"]),
            "absolute_pnl": float(metrics["final_equity"] - metrics["initial_equity"]),
            "percentage_pnl": float(
                ((metrics["final_equity"] - metrics["initial_equity"]) / metrics["initial_equity"] * 100.0)
                if metrics["initial_equity"] > 0
                else 0.0
            ),
        },
        "execution": {
            "requested_trades": requested_trades,
            "executed_trades": executed_trades,
            "failed_trades": failed_trades,
        },
        "balances": metrics["final_balances"],
        "assumptions": {
            "fee": float(args.fee),
            "initial_balances": {
                "fiat": float(args.fiat),
                "token_1": float(args.token1),
                "token_2": float(args.token2),
            },
        },
        "best_params": best_params,
    }

    print("=" * 52)
    print("VISUALIZADOR DE RESULTADOS")
    print("=" * 52)
    print(json.dumps(summary, indent=2))
    print("=" * 52)


if __name__ == "__main__":
    main()
