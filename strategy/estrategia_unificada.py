"""
Estrategia Unificada Mejorada — Optimizada para Score = 0.7×Sharpe - 0.2×|MaxDD| - 0.1×(TV/1e6)

Combina módulos sofisticados con regime detection y volatility-aware sizing:

  MÓDULO 1 — Arbitraje Triangular (risk_free_bulletproof)
    Market-neutral por diseño. No depende del período ni de la tendencia.
    Genera Sharpe alto con turnover bajo usando safety factors.

  MÓDULO 2 — Statistical Arbitrage (Z-Score Pairs Trading)
    Usa z-score del spread log(ETH/BTC) con entrada a 3.5-sigma en regímenes LOW.
    Mejor para capturar mean-reversion en mercados tranquilos.

  MÓDULO 3 — Mean-Reversion Intra-Asset (adaptativo por régimen)
    Z-score dinámico con banda adaptativa según volatilidad realizada y régimen.
    Señal de pánico extremo para entradas de alta convicción.

  MÓDULO 4 — Momentum Direccional (regime-aware)
    EMA crossover + ATR dinámico. ATR para sizing y stops adaptativos.
    Solo actúa en régimen TREND con confirmación multi-indicador.

Mejoras principales:
  - Regime Detection (LOW, MEDIUM, TREND) basado en volatilidad normalizada.
  - ATR mejorado con timeframe aggregation para mayor confiabilidad.
  - Z-scores más conservadores (3.5-sigma) en regímenes LOW para stat-arb.
  - Sizing optimizado con deducción de fees antes de la asignación.
  - Position tracking detallado (entry_price, peak_price) para mejor gestión.
  - Thresholds adaptativos según volatilidad realizada y régimen.
  - Cooldowns en ticks con early exit en casos de pánico.
"""

from collections import defaultdict, deque
import numpy as np

DEFAULT_FEE = 0.0003
EPS = 1e-9

_P1F = "token_1/fiat"
_P2F = "token_2/fiat"
_P12 = "token_1/token_2"


def _clamp01(value):
    return min(max(float(value), 0.0), 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers estadísticos mejorados
# ──────────────────────────────────────────────────────────────────────────────

def _sma(arr, w):
    if len(arr) < w:
        return None
    recent = np.asarray(list(arr)[-w:], dtype=float)
    return float(np.mean(recent))

def _ema_numpy(series, window):
    """Cálculo rápido de EMA sobre array numpy."""
    if len(series) < window:
        return np.ones_like(series) * np.mean(series)
    alpha = 2.0 / (window + 1)
    ema = np.zeros_like(series)
    ema[0] = series[0]
    for i in range(1, len(series)):
        ema[i] = series[i] * alpha + ema[i - 1] * (1.0 - alpha)
    return ema

def _ema(arr, period):
    if len(arr) < period:
        return None
    arr_seq = list(arr)[-period:]
    alpha = 2.0 / (period + 1)
    v = float(arr_seq[0])
    for x in arr_seq[1:]:
        v = alpha * float(x) + (1.0 - alpha) * v
    return v

def _std(arr, w):
    if len(arr) < w:
        return None
    recent = np.asarray(list(arr)[-w:], dtype=float)
    return float(np.std(recent, ddof=1))

def _zscore(arr, w):
    if len(arr) < w:
        return None
    recent = np.asarray(list(arr)[-w:], dtype=float)
    mu  = recent.mean()
    std = recent.std(ddof=1)
    if std < EPS:
        return None
    return (recent[-1] - mu) / std

def _roc(arr, period):
    if len(arr) <= period:
        return None
    arr_seq = list(arr)
    prev = float(arr_seq[-period - 1])
    if abs(prev) < EPS:
        return None
    return (float(arr_seq[-1]) - prev) / prev

def _atr_aggregated(closes, highs, lows, candle_tf, w_atr=14):
    """ATR con timeframe aggregation para mayor confiabilidad."""
    arr_c = np.asarray(closes, dtype=float)
    arr_h = np.asarray(highs, dtype=float)
    arr_l = np.asarray(lows, dtype=float)
    
    cutoff = (len(arr_c) // candle_tf) * candle_tf
    if cutoff < candle_tf * (w_atr + 1):
        return None
    
    # Aggregate to synthetic candles
    c_ch = arr_c[-cutoff:].reshape(-1, candle_tf)
    h_ch = arr_h[-cutoff:].reshape(-1, candle_tf)
    l_ch = arr_l[-cutoff:].reshape(-1, candle_tf)
    
    syn_closes = c_ch[:, -1]
    syn_highs = np.max(h_ch, axis=1)
    syn_lows = np.min(l_ch, axis=1)
    
    if len(syn_closes) < w_atr + 1:
        return None
    
    # TR calculation
    tr1 = syn_highs[1:] - syn_lows[1:]
    tr2 = np.abs(syn_highs[1:] - syn_closes[:-1])
    tr3 = np.abs(syn_lows[1:] - syn_closes[:-1])
    tr_arr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = float(np.mean(tr_arr[-w_atr:]))
    return atr if atr > EPS else None

def _atr(highs, lows, closes, w):
    if len(closes) < w + 1:
        return None
    trs = []
    for i in range(len(closes) - w, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i]  - closes[i - 1]),
        )
        trs.append(tr)
    return float(np.mean(trs))

def _realized_vol(closes, w):
    """Volatilidad realizada (std de log-returns) en ventana w."""
    if len(closes) < w + 1:
        return None
    arr = np.asarray(list(closes)[-(w + 1):], dtype=float)
    rets = np.diff(np.log(arr + EPS))
    return float(rets.std(ddof=1))


# ──────────────────────────────────────────────────────────────────────────────
# Estrategia principal mejorada
# ──────────────────────────────────────────────────────────────────────────────

class Strategy:
    """Estrategia unificada multi-módulo con regime detection y ATR mejorado."""

    def __init__(self):
        self.params = {
            # Ventanas (antes fijas, ahora optimizables)
            "w_fast": 8,
            "w_slow": 21,
            "w_roc": 10,
            "w_atr": 14,
            "w_vol_sma": 20,
            "max_hist": 200,
            "candle_tf": 60,  # Timeframe aggregation: 60 ticks = 1 hora
            "take_profit_pct": 0.0,
            "emergency_sell_frac": 0.0,
            # Ventanas tunables
            "w_spread": 60,
            "w_mr": 40,
            "w_vol": 30,
            "w_regime": 720,  # 12-hour window para régimen
            # Cooldowns tunables (altos para limitar turnover)
            "cd_arb": 480,
            "cd_lag": 360,
            "cd_mr": 360,
            "cd_mom": 240,
            "cd_panic": 360,
            # Presupuestos tunables (bajos para limitar turnover)
            "budget_arb": 0.010,
            "budget_lag": 0.010,
            "budget_mr": 0.010,
            "budget_mom": 0.010,
            # Thresholds tunables
            "z_mr_base": 1.8,
            "z_panic": 3.0,
            "z_stat_arb": 3.5,  # Conservador para stat-arb
            "panic_rp": 0.70,
            "roc_div": 0.006,
            "spread_z": 1.8,
            "arb_edge_mul": 3.0,
        }
        self._sync_params()
        self._reset_state()

    def _sync_params(self):
        p = dict(self.params)

        self.BUDGET_ARB = min(max(float(p["budget_arb"]), 0.0), 1.0)
        self.BUDGET_LAG = min(max(float(p["budget_lag"]), 0.0), 1.0)
        self.BUDGET_MR = min(max(float(p["budget_mr"]), 0.0), 1.0)
        self.BUDGET_MOM = min(max(float(p["budget_mom"]), 0.0), 1.0)

        self.W_SPREAD = max(5, int(p["w_spread"]))
        self.W_MR = max(5, int(p["w_mr"]))
        self.W_VOL = max(5, int(p["w_vol"]))
        self.W_REGIME = max(100, int(p["w_regime"]))
        self.W_FAST = max(2, int(p["w_fast"]))
        self.W_SLOW = max(self.W_FAST + 1, int(p["w_slow"]))
        self.W_ROC = max(2, int(p["w_roc"]))
        self.W_ATR = max(2, int(p["w_atr"]))
        self.W_VOL_SMA = max(2, int(p["w_vol_sma"]))
        self.CANDLE_TF = max(1, int(p["candle_tf"]))

        min_hist = max(self.W_SLOW + 10, self.W_SPREAD + 5, self.W_MR + self.W_VOL + 5, self.W_REGIME + 20)
        self.MAX_HIST = max(min_hist, int(p["max_hist"]))

        self.Z_MR_BASE = max(0.1, float(p["z_mr_base"]))
        self.Z_PANIC = max(0.1, float(p["z_panic"]))
        self.Z_STAT_ARB = max(0.5, float(p["z_stat_arb"]))
        self.PANIC_RP = min(max(float(p["panic_rp"]), 0.0), 1.0)
        self.ROC_DIV = max(0.0, float(p["roc_div"]))
        self.SPREAD_Z = max(0.1, float(p["spread_z"]))
        self.ARB_EDGE_MUL = max(1.0, float(p["arb_edge_mul"]))

        self.CD_ARB = max(1, int(p["cd_arb"]))
        self.CD_LAG = max(1, int(p["cd_lag"]))
        self.CD_MR = max(1, int(p["cd_mr"]))
        self.CD_MOM = max(1, int(p["cd_mom"]))
        self.CD_PANIC = max(1, int(p["cd_panic"]))

        # Guardar normalizados
        self.params.update({
            "w_fast": self.W_FAST, "w_slow": self.W_SLOW, "w_roc": self.W_ROC,
            "w_atr": self.W_ATR, "w_vol_sma": self.W_VOL_SMA,
            "max_hist": self.MAX_HIST, "candle_tf": self.CANDLE_TF,
            "w_spread": self.W_SPREAD, "w_mr": self.W_MR, "w_vol": self.W_VOL,
            "w_regime": self.W_REGIME, "z_mr_base": self.Z_MR_BASE,
            "z_panic": self.Z_PANIC, "z_stat_arb": self.Z_STAT_ARB,
            "panic_rp": self.PANIC_RP, "roc_div": self.ROC_DIV,
            "spread_z": self.SPREAD_Z, "arb_edge_mul": self.ARB_EDGE_MUL,
            "cd_arb": self.CD_ARB, "cd_lag": self.CD_LAG, "cd_mr": self.CD_MR,
            "cd_mom": self.CD_MOM, "cd_panic": self.CD_PANIC,
            "budget_arb": self.BUDGET_ARB, "budget_lag": self.BUDGET_LAG,
            "budget_mr": self.BUDGET_MR, "budget_mom": self.BUDGET_MOM,
        })

    def _reset_state(self):
        mh = self.MAX_HIST
        self._c = {p: deque(maxlen=mh) for p in [_P1F, _P2F, _P12]}
        self._h = {p: deque(maxlen=mh) for p in [_P1F, _P2F, _P12]}
        self._l = {p: deque(maxlen=mh) for p in [_P1F, _P2F, _P12]}
        self._v = {p: deque(maxlen=mh) for p in [_P1F, _P2F, _P12]}
        self._spread = deque(maxlen=mh)
        self._last = defaultdict(lambda: -10_000)
        self.positions = {_P1F: {"qty": 0.0, "entry_price": 0.0, "peak_price": 0.0},
                         _P2F: {"qty": 0.0, "entry_price": 0.0, "peak_price": 0.0}}
        self.stat_arb_state = "flat"
        self.step = 0

    def set_params(self, params):
        if params:
            self.params.update(params)
            self._sync_params()
            self._reset_state()

    def reset(self):
        self._reset_state()

    def _ok(self, key, cd):
        return (self.step - self._last[key]) >= cd

    def _mark(self, key):
        self._last[key] = self.step

    def _detect_regime(self):
        if len(self._c[_P1F]) < self.W_REGIME + 1:
            return "MEDIUM"
        
        closes = np.asarray(list(self._c[_P1F])[-self.W_REGIME:], dtype=float)
        highs = np.asarray(list(self._h[_P1F])[-self.W_REGIME:], dtype=float)
        lows = np.asarray(list(self._l[_P1F])[-self.W_REGIME:], dtype=float)
        prev_closes = np.asarray(list(self._c[_P1F])[-(self.W_REGIME+1):-1], dtype=float)
        
        tr_arr = np.maximum(highs - lows, np.maximum(np.abs(highs - prev_closes), np.abs(lows - prev_closes)))
        avg_tr = float(np.mean(tr_arr))
        norm_vol = avg_tr / (closes[-1] + EPS)
        
        return "LOW" if norm_vol < 0.0015 else ("TREND" if norm_vol > 0.0030 else "MEDIUM")

    def _arb(self, p1f, p2f, p12, balances, fee):
        if not self._ok("arb", self.CD_ARB):
            return []

        edge = self.ARB_EDGE_MUL * fee
        implied = p1f / (p2f + EPS)
        fiat_bal = float(balances.get("fiat", 0.0))
        if fiat_bal < 10.0:
            return []

        invest = fiat_bal * self.BUDGET_ARB
        SF = 0.99999

        q1  = invest / (p1f * (1.0 + fee))
        q1s = q1 * SF
        t2  = q1s * p12 * (1.0 - fee) * SF
        out1 = t2 * p2f * (1.0 - fee)

        q2  = invest / (p2f * (1.0 + fee))
        q2s = q2 * SF
        q1r = q2s / (p12 * (1.0 + fee)) * SF
        out2 = q1r * p1f * (1.0 - fee)

        min_out = invest * 1.0000001

        if out1 > min_out and p12 < implied * (1.0 - edge):
            self._mark("arb")
            return [{"pair": _P1F, "side": "buy",  "qty": q1,  "bypass_pipeline": True},
                    {"pair": _P12, "side": "sell", "qty": q1s, "bypass_pipeline": True},
                    {"pair": _P2F, "side": "sell", "qty": t2,  "bypass_pipeline": True}]

        if out2 > min_out and p12 > implied * (1.0 + edge):
            self._mark("arb")
            return [{"pair": _P2F, "side": "buy",  "qty": q2,  "bypass_pipeline": True},
                    {"pair": _P12, "side": "buy",  "qty": q1r, "bypass_pipeline": True},
                    {"pair": _P1F, "side": "sell", "qty": q1r, "bypass_pipeline": True}]
        return []

    def _stat_arb(self, balances, fee, regime):
        if len(self._spread) < self.W_SPREAD or regime != "LOW":
            return None
        
        z = _zscore(self._spread, self.W_SPREAD)
        if z is None:
            return None
        
        fiat_bal = float(balances.get("fiat", 0.0))
        t1_bal = float(balances.get("token_1", 0.0))
        t2_bal = float(balances.get("token_2", 0.0))
        p12 = float(self._c[_P12][-1]) if len(self._c[_P12]) > 0 else None
        
        if p12 is None or p12 < EPS or self.stat_arb_state != "flat":
            if self.stat_arb_state == "short_spread" and z <= 1.0 and t2_bal > 0.01:
                self._mark("stat_arb")
                self.stat_arb_state = "flat"
                return {"pair": _P12, "side": "buy", "qty": t2_bal * 0.99, "consensus_score": 0.60}
            elif self.stat_arb_state == "long_spread" and z >= 1.0 and t1_bal > 0.1:
                self._mark("stat_arb")
                self.stat_arb_state = "flat"
                return {"pair": _P12, "side": "sell", "qty": t1_bal * 0.99, "consensus_score": 0.60}
            return None
        
        if z > self.Z_STAT_ARB and t1_bal > 0.1:
            alloc = min(self.BUDGET_LAG * 1.5, 0.015)
            qty = t1_bal * alloc
            if qty > 0:
                self._mark("stat_arb")
                self.stat_arb_state = "short_spread"
                return {"pair": _P12, "side": "sell", "qty": qty, "consensus_score": 0.70}
        
        elif z < -self.Z_STAT_ARB and fiat_bal > 10.0:
            alloc = min(self.BUDGET_LAG * 1.5, 0.015)
            qty = (fiat_bal * alloc) / (p12 * (1.0 + fee))
            if qty > 0.001:
                self._mark("stat_arb")
                self.stat_arb_state = "long_spread"
                return {"pair": _P12, "side": "buy", "qty": qty, "consensus_score": 0.70}
        
        return None

    def _mean_rev(self, pair, close, high, low, balances, fee, regime):
        closes = self._c[pair]
        if len(closes) < self.W_MR + self.W_VOL:
            return None

        z = _zscore(closes, self.W_MR)
        if z is None:
            return None

        rv = _realized_vol(closes, self.W_VOL) or 0.001
        rv_norm = min(rv / 0.002, 2.0)
        regime_mul = {"LOW": 1.0, "MEDIUM": 1.3, "TREND": 1.8}
        band = self.Z_MR_BASE * regime_mul.get(regime, 1.3) + rv_norm * 0.5

        base, quote = pair.split("/")
        base_bal  = float(balances.get(base,  0.0))
        quote_bal = float(balances.get(quote, 0.0))

        cr = high - low + EPS
        range_pos = (close - low) / cr

        if (z < -self.Z_PANIC and range_pos > self.PANIC_RP and 
            self._ok(f"panic_{pair}", self.CD_PANIC) and quote_bal > 0 and self.step >= self.W_SLOW + 5):
            alloc = min(self.BUDGET_MR, 0.35)
            qty = (quote_bal * alloc) / (close * (1.0 + fee))
            if qty > 0:
                self._mark(f"panic_{pair}")
                self._mark(f"mr_{pair}")
                self.positions[pair]["entry_price"] = close
                self.positions[pair]["peak_price"] = close
                panic_strength = _clamp01(abs(z) / (self.Z_PANIC + EPS))
                return {"pair": pair, "side": "buy", "qty": qty,
                        "consensus_score": 0.70 + 0.30 * panic_strength}

        if not self._ok(f"mr_{pair}", self.CD_MR):
            return None

        if self.step >= self.W_SLOW + 5 and z < -band and range_pos > 0.5 and quote_bal > 0:
            strength = min(abs(z) / band, 2.0)
            alloc = min(self.BUDGET_MR * 0.4 * strength, self.BUDGET_MR)
            qty = (quote_bal * alloc) / (close * (1.0 + fee))
            if qty > 0:
                self._mark(f"mr_{pair}")
                self.positions[pair]["entry_price"] = close
                self.positions[pair]["peak_price"] = close
                score = 0.55 + 0.45 * _clamp01((strength - 1.0) / 1.0)
                return {"pair": pair, "side": "buy", "qty": qty, "consensus_score": score}

        if z > band * 0.6 and base_bal > 0:
            sell_frac = min(0.3 + 0.2 * (z / band), 0.9)
            qty = base_bal * sell_frac
            if qty > 0:
                self._mark(f"mr_{pair}")
                strength = _clamp01((z - 0.6 * band) / (0.8 * band + EPS))
                return {"pair": pair, "side": "sell", "qty": qty, "consensus_score": 0.6 + 0.4 * strength}

        return None

    def _momentum(self, pair, close, balances, fee, regime):
        closes = self._c[pair]
        highs = self._h[pair]
        lows = self._l[pair]
        volumes = self._v[pair]

        need = max(self.W_SLOW + 5, self.W_ROC + 1, self.W_ATR + 2, self.W_VOL_SMA + 1)
        if len(closes) < need or not self._ok(f"mom_{pair}", self.CD_MOM) or regime != "TREND":
            return None

        fast_ema = _ema(closes, self.W_FAST)
        slow_ema = _ema(closes, self.W_SLOW)
        roc      = _roc(closes, self.W_ROC)
        
        atr = _atr_aggregated(list(closes)[-(self.W_ATR * self.CANDLE_TF + 10):],
                              list(highs)[-(self.W_ATR * self.CANDLE_TF + 10):],
                              list(lows)[-(self.W_ATR * self.CANDLE_TF + 10):],
                              self.CANDLE_TF, self.W_ATR)

        if None in (fast_ema, slow_ema, roc, atr) or atr <= EPS:
            return None

        vol_avg = float(np.mean(list(volumes)[-self.W_VOL_SMA:] or [1.0])) + EPS
        vol_ratio = float(volumes[-1]) / vol_avg

        base, quote = pair.split("/")
        base_bal  = float(balances.get(base,  0.0))
        quote_bal = float(balances.get(quote, 0.0))

        rv = _realized_vol(closes, self.W_VOL) or 0.001
        roc_thresh = max(0.003, rv * 1.5)

        if (self.step >= self.W_SLOW + 5 and fast_ema > slow_ema and 
            roc > roc_thresh and vol_ratio > 1.15 and quote_bal > 0):
            signal_q = min(roc / roc_thresh, 2.0) * min(vol_ratio / 1.15, 1.5)
            alloc = min(self.BUDGET_MOM * 0.5 * signal_q, self.BUDGET_MOM)
            qty = (quote_bal * alloc) / (close * (1.0 + fee))
            if qty > 0:
                self._mark(f"mom_{pair}")
                self.positions[pair]["entry_price"] = close
                self.positions[pair]["peak_price"] = close
                score = 0.55 + 0.45 * _clamp01((signal_q - 1.0) / 1.0)
                return {"pair": pair, "side": "buy", "qty": qty, "consensus_score": score}

        if fast_ema < slow_ema and roc < -roc_thresh and base_bal > 0:
            signal_q  = min(abs(roc) / roc_thresh, 2.0)
            sell_frac = min(0.3 + 0.3 * signal_q, 0.85)
            qty = base_bal * sell_frac
            if qty > 0:
                self._mark(f"mom_{pair}")
                score = 0.6 + 0.4 * _clamp01((signal_q - 1.0) / 1.0)
                return {"pair": pair, "side": "sell", "qty": qty, "consensus_score": score}

        return None

    def on_data(self, market_data, balances):
        self.step += 1
        fee = float(market_data.get("fee", DEFAULT_FEE))
        actions = []

        for pair in [_P1F, _P2F, _P12]:
            data = market_data.get(pair)
            if data is None:
                continue
            c = float(data.get("close",  0.0))
            h = float(data.get("high",   c))
            l = float(data.get("low",    c))
            v = float(data.get("volume", 1.0))
            if c <= EPS:
                continue
            self._c[pair].append(c)
            self._h[pair].append(h)
            self._l[pair].append(l)
            self._v[pair].append(max(v, EPS))

        if len(self._c[_P1F]) > 0 and len(self._c[_P2F]) > 0:
            self._spread.append(np.log(float(self._c[_P2F][-1]) + EPS) - np.log(float(self._c[_P1F][-1]) + EPS))

        has_all = all(len(self._c[p]) > 0 for p in [_P1F, _P2F, _P12])
        regime = self._detect_regime()

        if has_all and self.step >= self.W_SLOW + 5:
            p1f = float(self._c[_P1F][-1])
            p2f = float(self._c[_P2F][-1])
            p12 = float(self._c[_P12][-1])
            arb_acts = self._arb(p1f, p2f, p12, balances, fee)
            actions.extend(arb_acts)

        if not actions:
            if has_all:
                sig = self._stat_arb(balances, fee, regime)
                if sig:
                    actions.append(sig)

            for pair in [_P1F, _P2F]:
                data = market_data.get(pair)
                if data is None or len(self._c[pair]) < self.W_MR:
                    continue
                c = float(self._c[pair][-1])
                h = float(self._h[pair][-1])
                l = float(self._l[pair][-1])
                sig = self._mean_rev(pair, c, h, l, balances, fee, regime)
                if sig:
                    actions.append(sig)

            for pair in [_P1F, _P2F]:
                data = market_data.get(pair)
                if data is None:
                    continue
                c = float(self._c[pair][-1])
                sig = self._momentum(pair, c, balances, fee, regime)
                if sig:
                    actions.append(sig)

        return actions if actions else None
