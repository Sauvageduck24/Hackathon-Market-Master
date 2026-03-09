"""Grabs Binance 1-minute OHLCV via CCXT and saves as CSV."""
import ccxt
import pandas as pd
import os
import argparse
from datetime import datetime, timedelta, timezone
from tqdm import tqdm

ONE_MINUTE_IN_MILLIS = 60_000

def fetch(symbol: str, start_ts: int, end_ts: int) -> pd.DataFrame:
    """Fetch OHLCV between timestamps (inclusive) in milliseconds."""
    
    # 1. Activamos el limitador nativo de CCXT. ¡Te protege de los baneos!
    exchange = ccxt.binance({
        'enableRateLimit': True, 
    })
    
    frames = []
    since_ts = start_ts
    
    # Calculamos el total de velas de 1m esperadas para configurar la barra
    total_expected_candles = (end_ts - start_ts) // ONE_MINUTE_IN_MILLIS
    
    # 2. Inicializamos la barra de progreso con tqdm
    with tqdm(total=total_expected_candles, desc=f"Descargando {symbol}") as pbar:
        while since_ts < end_ts:
            remaining_minutes = (end_ts - since_ts) // ONE_MINUTE_IN_MILLIS
            if remaining_minutes <= 0:
                break
            
            # Binance permite un máximo de 1000 velas por petición
            limit_minutes = min(1000, remaining_minutes)
            
            # Descargamos el lote (el rate limit de CCXT actúa aquí automáticamente)
            batch = exchange.fetch_ohlcv(symbol, timeframe="1m", since=since_ts, limit=limit_minutes)
            
            if not batch:
                break
                
            frames.append(pd.DataFrame(batch, columns=["timestamp", "open", "high", "low", "close", "volume"]))
            
            # El siguiente bloque empezará un minuto después de la última vela obtenida
            since_ts = batch[-1][0] + ONE_MINUTE_IN_MILLIS
            
            # Actualizamos la barra de progreso por el número de velas descargadas
            pbar.update(len(batch))

    if not frames:
        print(f"\nNo se encontraron datos para {symbol} en ese rango temporal.")
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["symbol"] = symbol
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("symbol", help="Token pair (ej. BTC/USDT)")
    # Nota: Actualizado a timezone.utc (utcfromtimestamp quedará obsoleto en futuras versiones de Python)
    parser.add_argument("--start", type=int, help="UNIX-ms", default=int((datetime.now(timezone.utc) - timedelta(days=1)).timestamp() * 1e3))
    parser.add_argument("--end", type=int, help="UNIX-ms", default=int(datetime.now(timezone.utc).timestamp() * 1e3))
    parser.add_argument("--output", help="Output csv file path")
    args = parser.parse_args()

    print(f"Iniciando descarga para {args.symbol}...")
    df = fetch(args.symbol, args.start, args.end)
    
    if not df.empty:
        os.makedirs("../data", exist_ok=True)
        
        # Use provided output path or generate default
        if args.output:
            out_path = args.output
        else:
            out_path = f"../data/{args.symbol.replace('/', '').lower()}.csv"
        
        df.to_csv(out_path, index=False)
        print(f"\n¡Éxito! Guardadas {len(df):,} filas en {out_path}")