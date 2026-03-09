import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np

# -- Load data --
df = pd.read_csv(r"c:\Users\pabsa\OneDrive\Escritorio\trading-hackaton-kaggle-main\data\test.csv")
btc = df[df["symbol"] == "token_2/fiat"].copy()
btc = btc.sort_values("timestamp").reset_index(drop=True)

print(f"Total BTC rows: {len(btc)}")
print(f"Timestamp range: {btc['timestamp'].iloc[0]}  to  {btc['timestamp'].iloc[-1]}")
print(f"Price range: {btc['close'].min():.2f}  -  {btc['close'].max():.2f}")
print()

# -- Compute indicators --
btc["long_sma"]  = btc["close"].rolling(1440).mean()
btc["med_sma"]   = btc["close"].rolling(360).mean()
btc["med_sma_60_ago"] = btc["med_sma"].shift(60)

# Only analyse after warmup (tick >= 1440 so all windows are valid)
warmup = 1440
w = btc.iloc[warmup:].copy()
print(f"Ticks after warmup (tick >= {warmup}): {len(w)}")
print(f"  long_sma NaN count: {w['long_sma'].isna().sum()}")
print(f"  med_sma_60_ago NaN count: {w['med_sma_60_ago'].isna().sum()}")
print()

# -- Individual conditions --
price = w["close"].values
long_sma = w["long_sma"].values
med_now  = w["med_sma"].values
med_ago  = w["med_sma_60_ago"].values

cond_dip      = price < long_sma * 0.98
cond_med_rise = med_now > med_ago
cond_both     = cond_dip & cond_med_rise

print("=== Condition counts (strict: price < long_sma * 0.98) ===")
print(f"  price < long_sma * 0.98 :  {np.nansum(cond_dip):>6d} / {len(w)}")
print(f"  med_sma_now > med_sma_60 : {np.nansum(cond_med_rise):>6d} / {len(w)}")
print(f"  BOTH                     : {np.nansum(cond_both):>6d} / {len(w)}")
print()

if np.nansum(cond_both) > 0:
    idxs = np.where(cond_both)[0][:5]
    print("First 5 ticks where BOTH are true:")
    for i in idxs:
        row = w.iloc[i]
        ratio = row["close"] / row["long_sma"]
        print(f"  tick {warmup+i:>6d} | ts={row['timestamp']} | "
              f"close={row['close']:.2f}  long_sma={row['long_sma']:.2f}  "
              f"ratio={ratio:.5f}  med_now={row['med_sma']:.2f}  med_60ago={row['med_sma_60_ago']:.2f}")
else:
    print("  --> No ticks satisfy BOTH conditions simultaneously.")
print()

# -- Relaxed thresholds --
print("=== Relaxed thresholds ===")
for mult_label, mult in [("0.99", 0.99), ("1.00 (just med_rising)", 1.00)]:
    cond_r = (price < long_sma * mult) & cond_med_rise
    cnt = int(np.nansum(cond_r))
    print(f"  price < long_sma * {mult_label}: {cnt:>6d} ticks")
    if cnt > 0:
        idxs_r = np.where(cond_r)[0][:5]
        for i in idxs_r:
            row = w.iloc[i]
            ratio = row["close"] / row["long_sma"]
            print(f"      tick {warmup+i:>6d} | close={row['close']:.2f}  "
                  f"long_sma={row['long_sma']:.2f}  ratio={ratio:.5f}")
print()

# -- Ratio distribution when med_rising --
print("=== price / long_sma  distribution WHEN med_rising is True ===")
mask_rising = cond_med_rise & ~np.isnan(long_sma) & ~np.isnan(med_ago)
if mask_rising.any():
    ratios = price[mask_rising] / long_sma[mask_rising]
    print(f"  count : {len(ratios)}")
    print(f"  min   : {np.min(ratios):.6f}")
    print(f"  5th%  : {np.percentile(ratios, 5):.6f}")
    print(f"  25th% : {np.percentile(ratios, 25):.6f}")
    print(f"  median: {np.median(ratios):.6f}")
    print(f"  75th% : {np.percentile(ratios, 75):.6f}")
    print(f"  95th% : {np.percentile(ratios, 95):.6f}")
    print(f"  max   : {np.max(ratios):.6f}")
else:
    print("  No ticks with med_rising = True (and valid SMAs).")

# -- Ratio distribution overall --
print()
print("=== price / long_sma  distribution OVERALL (after warmup) ===")
valid = ~np.isnan(long_sma)
ratios_all = price[valid] / long_sma[valid]
print(f"  count : {len(ratios_all)}")
print(f"  min   : {np.min(ratios_all):.6f}")
print(f"  5th%  : {np.percentile(ratios_all, 5):.6f}")
print(f"  25th% : {np.percentile(ratios_all, 25):.6f}")
print(f"  median: {np.median(ratios_all):.6f}")
print(f"  75th% : {np.percentile(ratios_all, 75):.6f}")
print(f"  95th% : {np.percentile(ratios_all, 95):.6f}")
print(f"  max   : {np.max(ratios_all):.6f}")
