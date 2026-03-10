"""Bulk 1-minute Binance downloader (2024-01-01 UTC -> now).

Genera:
- data/btcusdt_1m_hard.csv
- data/ethusdt_1m_hard.csv
- data/ethbtc_1m_hard.csv
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

BASE_URL = "https://api.binance.com/api/v3/klines"
INTERVAL = "1m"
ONE_MINUTE_MS = 60_000
MAX_LIMIT = 1000
DEFAULT_START = datetime(2022, 1, 1, tzinfo=timezone.utc)


@dataclass(frozen=True)
class Job:
    symbol: str
    csv_symbol: str
    output_name: str


JOBS = [
    Job(symbol="BTCUSDT", csv_symbol="BTC/USDT", output_name="btcusdt_1m_hard.csv"),
    Job(symbol="ETHUSDT", csv_symbol="ETH/USDT", output_name="ethusdt_1m_hard.csv"),
    Job(symbol="ETHBTC", csv_symbol="ETH/BTC", output_name="ethbtc_1m_hard.csv"),
]


def build_session() -> requests.Session:
    """Create a requests session with retry/backoff for API reliability."""
    session = requests.Session()
    retries = Retry(
        total=8,
        connect=8,
        read=8,
        backoff_factor=0.6,
        status_forcelist=[418, 429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=16, pool_maxsize=16)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def fetch_symbol(symbol: str, csv_symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Download 1m klines for one symbol in batches."""
    session = build_session()
    frames: List[pd.DataFrame] = []
    cursor = start_ms

    expected = max((end_ms - start_ms) // ONE_MINUTE_MS, 0)
    pbar = tqdm(total=expected, desc=f"{symbol}", position=0, leave=True)

    try:
        while cursor < end_ms:
            remaining = (end_ms - cursor) // ONE_MINUTE_MS
            if remaining <= 0:
                break

            limit = min(MAX_LIMIT, remaining)
            params = {
                "symbol": symbol,
                "interval": INTERVAL,
                "startTime": cursor,
                "limit": limit,
            }

            response = session.get(BASE_URL, params=params, timeout=30)
            if response.status_code != 200:
                raise RuntimeError(
                    f"Binance error {response.status_code} for {symbol}: {response.text[:250]}"
                )

            batch = response.json()
            if not batch:
                break

            frame = pd.DataFrame(
                batch,
                columns=[
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "close_time",
                    "quote_asset_volume",
                    "number_of_trades",
                    "taker_buy_base_asset_volume",
                    "taker_buy_quote_asset_volume",
                    "ignore",
                ],
            )
            frames.append(frame)

            last_open_time = int(batch[-1][0])
            cursor = last_open_time + ONE_MINUTE_MS
            pbar.update(len(batch))

            # Si Binance devuelve menos de lo solicitado, no hay mas datos en este rango.
            if len(batch) < limit:
                break

    finally:
        pbar.close()
        session.close()

    if not frames:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "symbol"])

    df = pd.concat(frames, ignore_index=True)
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    # Keep timestamp format aligned with scripts/download.py (without timezone suffix).
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["open"] = pd.to_numeric(df["open"], errors="coerce")
    df["high"] = pd.to_numeric(df["high"], errors="coerce")
    df["low"] = pd.to_numeric(df["low"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df["symbol"] = csv_symbol
    return df


def run() -> None:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    start_ms = int(DEFAULT_START.timestamp() * 1000)
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    print(f"Descarga hard desde {DEFAULT_START.isoformat()} hasta ahora (UTC)")
    print("Simbolos: BTCUSDT, ETHUSDT, ETHBTC")

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(fetch_symbol, job.symbol, job.csv_symbol, start_ms, end_ms): job
            for job in JOBS
        }

        for future in as_completed(futures):
            job = futures[future]
            try:
                df = future.result()
                out_path = data_dir / job.output_name
                df.to_csv(out_path, index=False)
                print(f"OK {job.symbol}: {len(df):,} filas -> {out_path}")
            except Exception as exc:  # noqa: BLE001
                print(f"ERROR {job.symbol}: {exc}")


if __name__ == "__main__":
    run()
