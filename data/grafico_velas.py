from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle
from tqdm import tqdm


DAYS_TO_PLOT = 1


def load_data(base_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    test_path = base_dir / "test.csv"
    submission_path = base_dir / "submission.csv"

    test_df = pd.read_csv(test_path)
    submission_df = pd.read_csv(submission_path)

    test_df["timestamp"] = pd.to_datetime(test_df["timestamp"])
    submission_df["timestamp"] = pd.to_datetime(submission_df["timestamp"])

    test_start = test_df["timestamp"].min()
    submission_start = submission_df["timestamp"].min()

    if pd.notna(test_start):
        test_end = test_start + pd.Timedelta(days=DAYS_TO_PLOT)
        test_df = test_df[test_df["timestamp"] < test_end]

    if pd.notna(submission_start):
        submission_end = submission_start + pd.Timedelta(days=DAYS_TO_PLOT)
        submission_df = submission_df[submission_df["timestamp"] < submission_end]

    return test_df.sort_values("timestamp"), submission_df.sort_values("timestamp")


def draw_candles(ax: plt.Axes, pair_df: pd.DataFrame, pair_name: str) -> None:
    x = mdates.date2num(pair_df["timestamp"])

    if len(x) > 1:
        width = (x[1] - x[0]) * 0.7
    else:
        width = 1 / (24 * 60) * 0.7

    candle_rows = zip(
        x,
        pair_df["open"],
        pair_df["high"],
        pair_df["low"],
        pair_df["close"],
    )

    for xi, open_, high, low, close in tqdm(
        candle_rows,
        total=len(pair_df),
        desc=f"Velas {pair_name}",
        leave=False,
    ):
        color = "#2ca02c" if close >= open_ else "#d62728"
        ax.vlines(xi, low, high, color=color, linewidth=1)

        body_low = min(open_, close)
        body_height = abs(close - open_)
        if body_height == 0:
            body_height = max(1e-8, pair_df["close"].mean() * 1e-6)

        rect = Rectangle(
            (xi - width / 2, body_low),
            width,
            body_height,
            facecolor=color,
            edgecolor=color,
            linewidth=1,
        )
        ax.add_patch(rect)

    ax.set_title(f"Velas: {pair_name}")
    ax.set_ylabel("Precio")
    ax.grid(True, alpha=0.2)


def overlay_trades(ax: plt.Axes, pair_df: pd.DataFrame, pair_trades: pd.DataFrame) -> None:
    if pair_trades.empty:
        return

    pair_prices = pair_df[["timestamp", "close"]].sort_values("timestamp")
    trades_with_price = pd.merge_asof(
        pair_trades.sort_values("timestamp"),
        pair_prices,
        on="timestamp",
        direction="backward",
    ).dropna(subset=["close"])

    buys = trades_with_price[trades_with_price["side"].str.lower() == "buy"]
    sells = trades_with_price[trades_with_price["side"].str.lower() == "sell"]

    if not buys.empty:
        ax.scatter(
            buys["timestamp"],
            buys["close"],
            marker="^",
            s=50,
            color="#1f77b4",
            label="Buy",
            zorder=3,
        )

    if not sells.empty:
        ax.scatter(
            sells["timestamp"],
            sells["close"],
            marker="v",
            s=50,
            color="#ff7f0e",
            label="Sell",
            zorder=3,
        )

    if not buys.empty or not sells.empty:
        ax.legend(loc="upper left")


def plot(test_df: pd.DataFrame, submission_df: pd.DataFrame, pairs: list[str]) -> None:
    fig, axes = plt.subplots(len(pairs), 1, figsize=(14, 4 * len(pairs)), sharex=True)
    if len(pairs) == 1:
        axes = [axes]

    pair_iterator = zip(axes, pairs)
    for ax, pair in tqdm(pair_iterator, total=len(pairs), desc="Procesando pares"):
        pair_df = test_df[test_df["symbol"] == pair].sort_values("timestamp")
        pair_trades = submission_df[submission_df["pair"] == pair].sort_values("timestamp")

        if pair_df.empty:
            ax.set_title(f"Sin datos en test.csv para {pair}")
            ax.axis("off")
            continue

        draw_candles(ax, pair_df, pair)
        overlay_trades(ax, pair_df, pair_trades)

    axes[-1].set_xlabel("Tiempo")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grafica velas de test.csv y marca operaciones de submission.csv"
    )
    parser.add_argument(
        "--pair",
        action="append",
        help="Par a mostrar (ej: token_1/fiat). Repetir para varios.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent

    test_df, submission_df = load_data(base_dir)

    available_pairs = sorted(test_df["symbol"].dropna().unique().tolist())
    pairs = args.pair if args.pair else available_pairs

    plot(test_df, submission_df, pairs)


if __name__ == "__main__":
    main()
