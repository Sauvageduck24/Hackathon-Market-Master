"""
Estrategia Unificada — Optimizada para Score = 0.7×Sharpe - 0.2×|MaxDD| - 0.1×(TV/1e6)

Combina los mejores módulos de las estrategias anteriores en una sola clase:

  MÓDULO 1 — Arbitraje Triangular (risk_free_bulletproof)
    Market-neutral por diseño. No depende del período ni de la tendencia.
    Genera Sharpe alto con turnover bajo usando safety factors.

  MÓDULO 2 — Lead-Lag Cross-Asset (estrategia_multi_activo)
    Usa el ROC de token_1 (BTC) como señal anticipada para token_2 (ETH).
    El spread log(ETH/BTC) revierte a la media → mean-reversion inter-asset.

  MÓDULO 3 — Mean-Reversion Intra-Asset (estrategia_bollinger + sniper)
    Z-score dinámico con banda adaptativa según volatilidad reciente.
    Señal de pánico extremo (z < -umbral + rechazo de vela) para entradas de alta convicción.

  MÓDULO 4 — Momentum Direccional (estrategia_momentum + scalping)
    EMA crossover + ROC + volumen. ATR para sizing y stop/take dinámicos.
    Solo actúa cuando hay confirmación multi-indicador.

Principios anti-overfitting:
  - Sin total_steps ni suposiciones sobre dirección del mercado.
  - Thresholds adaptativos: la banda de z-score escala con volatilidad realizada.
  - Cooldowns en ticks, no en tiempo absoluto.
  - Capital allocation proporcional a la calidad de la señal.
  - Módulos independientes con presupuesto de capital separado.
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
# Helpers estadísticos
# ──────────────────────────────────────────────────────────────────────────────

def _sma(arr, w):
    if len(arr) < w:
        return None
    recent = np.asarray(list(arr)[-w:], dtype=float)
    return float(np.mean(recent))

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
# Estrategia principal
# ──────────────────────────────────────────────────────────────────────────────

class Strategy:
    """Estrategia unificada multi-módulo."""

    FIXED_PARAMS = {
        # Ventanas fijas
        "w_fast": 8,
        "w_slow": 21,
        "w_roc": 10,
        "w_atr": 14,
        "w_vol_sma": 20,
        "max_hist": 200,
        # Cooldowns fijos
        "cd_arb": 1,
        "cd_lag": 10,
        "cd_mr": 8,
        "cd_mom": 5,
        "cd_panic": 20,
        # Presupuestos fijos
        "budget_arb": 0.20,
        "budget_lag": 0.15,
        "budget_mr": 0.25,
        "budget_mom": 0.20,
        # Otros fijos
        "take_profit_pct": 0.0,
        "emergency_sell_frac": 0.0,
    }

    def __init__(self):
        self.params = {
            # Ventanas tunables
            "w_spread": 60,
            "w_mr": 40,
            "w_vol": 30,
            # Thresholds tunables
            "z_mr_base": 1.8,
            "z_panic": 3.0,
            "panic_rp": 0.70,
            "roc_div": 0.006,
            "spread_z": 1.8,
            "arb_edge_mul": 3.0,
        }
        self._sync_params()
        self._reset_state()

    def _sync_params(self):
        # Los parametros fijos se aplican siempre y no forman parte de HPO.
        p = dict(self.FIXED_PARAMS)
        p.update(self.params)

        # Presupuestos
        self.BUDGET_ARB = min(max(float(p["budget_arb"]), 0.0), 1.0)
        self.BUDGET_LAG = min(max(float(p["budget_lag"]), 0.0), 1.0)
        self.BUDGET_MR = min(max(float(p["budget_mr"]), 0.0), 1.0)
        self.BUDGET_MOM = min(max(float(p["budget_mom"]), 0.0), 1.0)

        # Ventanas
        self.W_SPREAD = max(5, int(p["w_spread"]))
        self.W_MR = max(5, int(p["w_mr"]))
        self.W_VOL = max(5, int(p["w_vol"]))
        self.W_FAST = max(2, int(p["w_fast"]))
        self.W_SLOW = max(self.W_FAST + 1, int(p["w_slow"]))
        self.W_ROC = max(2, int(p["w_roc"]))
        self.W_ATR = max(2, int(p["w_atr"]))
        self.W_VOL_SMA = max(2, int(p["w_vol_sma"]))

        min_hist = max(self.W_SLOW + 10, self.W_SPREAD + 5, self.W_MR + self.W_VOL + 5)
        self.MAX_HIST = max(min_hist, int(p["max_hist"]))

        # Thresholds
        self.Z_MR_BASE = max(0.1, float(p["z_mr_base"]))
        self.Z_PANIC = max(0.1, float(p["z_panic"]))
        self.PANIC_RP = min(max(float(p["panic_rp"]), 0.0), 1.0)
        self.ROC_DIV = max(0.0, float(p["roc_div"]))
        self.SPREAD_Z = max(0.1, float(p["spread_z"]))
        self.ARB_EDGE_MUL = max(1.0, float(p["arb_edge_mul"]))

        # Cooldowns
        self.CD_ARB = max(1, int(p["cd_arb"]))
        self.CD_LAG = max(1, int(p["cd_lag"]))
        self.CD_MR = max(1, int(p["cd_mr"]))
        self.CD_MOM = max(1, int(p["cd_mom"]))
        self.CD_PANIC = max(1, int(p["cd_panic"]))

        # Guardar normalizados para plantilla de HPO
        # Normalizamos solo los tunables expuestos en self.params.
        self.params["w_spread"] = self.W_SPREAD
        self.params["w_mr"] = self.W_MR
        self.params["w_vol"] = self.W_VOL
        self.params["z_mr_base"] = self.Z_MR_BASE
        self.params["z_panic"] = self.Z_PANIC
        self.params["panic_rp"] = self.PANIC_RP
        self.params["roc_div"] = self.ROC_DIV
        self.params["spread_z"] = self.SPREAD_Z
        self.params["arb_edge_mul"] = self.ARB_EDGE_MUL

    def _reset_state(self):
        mh = self.MAX_HIST
        self._c = {p: deque(maxlen=mh) for p in [_P1F, _P2F, _P12]}
        self._h = {p: deque(maxlen=mh) for p in [_P1F, _P2F, _P12]}
        self._l = {p: deque(maxlen=mh) for p in [_P1F, _P2F, _P12]}
        self._v = {p: deque(maxlen=mh) for p in [_P1F, _P2F, _P12]}
        self._spread = deque(maxlen=mh)  # log(p2f) - log(p1f)
        self._last = defaultdict(lambda: -10_000)
        self.step = 0

    def set_params(self, params):
        if not params:
            return
        tunables = {k: v for k, v in params.items() if k not in self.FIXED_PARAMS}
        if tunables:
            self.params.update(tunables)
        self._sync_params()
        self._reset_state()

    def reset(self):
        """Reset rolling buffers and counters between backtest runs."""
        self._reset_state()

    # ── Cooldown ────────────────────────────────────────────────────────────

    def _ok(self, key, cd):
        return (self.step - self._last[key]) >= cd

    def _mark(self, key):
        self._last[key] = self.step

    # ────────────────────────────────────────────────────────────────────────
    # MÓDULO 1 — Arbitraje Triangular
    # ────────────────────────────────────────────────────────────────────────

    def _arb(self, p1f, p2f, p12, balances, fee):
        """
        Compara precio implícito token_1/token_2 = p1f/p2f con el real (p12).
        Si hay discrepancia > edge → ejecutar triángulo completo.
        Safety factor 0.99999 en cada leg para evitar rechazos por decimales.
        """
        if not self._ok("arb", self.CD_ARB):
            return []

        edge = self.ARB_EDGE_MUL * fee
        implied = p1f / (p2f + EPS)
        fiat_bal = float(balances.get("fiat", 0.0))
        if fiat_bal < 10.0:
            return []

        invest = fiat_bal * self.BUDGET_ARB
        SF = 0.99999  # safety factor

        # Camino 1: fiat → T1 → T2 → fiat
        q1  = invest / (p1f * (1.0 + fee))
        q1s = q1 * SF
        t2  = q1s * p12 * (1.0 - fee) * SF
        out1 = t2 * p2f * (1.0 - fee)

        # Camino 2: fiat → T2 → T1 → fiat
        q2  = invest / (p2f * (1.0 + fee))
        q2s = q2 * SF
        q1r = q2s / (p12 * (1.0 + fee)) * SF
        out2 = q1r * p1f * (1.0 - fee)

        min_out = invest * 1.0000001

        if out1 > min_out and p12 < implied * (1.0 - edge):
            self._mark("arb")
            return [
                {"pair": _P1F, "side": "buy",  "qty": q1,  "bypass_pipeline": True},
                {"pair": _P12, "side": "sell", "qty": q1s, "bypass_pipeline": True},
                {"pair": _P2F, "side": "sell", "qty": t2,  "bypass_pipeline": True},
            ]

        if out2 > min_out and p12 > implied * (1.0 + edge):
            self._mark("arb")
            return [
                {"pair": _P2F, "side": "buy",  "qty": q2,  "bypass_pipeline": True},
                {"pair": _P12, "side": "buy",  "qty": q1r, "bypass_pipeline": True},
                {"pair": _P1F, "side": "sell", "qty": q1r, "bypass_pipeline": True},
            ]

        return []

    # ────────────────────────────────────────────────────────────────────────
    # MÓDULO 2 — Lead-Lag + Spread Mean-Reversion
    # ────────────────────────────────────────────────────────────────────────

    def _lead_lag(self, balances, fee):
        """
        ROC(token_1) diverge de ROC(token_2) → anticipar movimiento del rezagado.
        Threshold adaptativo: escala con volatilidad realizada de cada activo.
        """
        if not self._ok("lag", self.CD_LAG):
            return None
        if len(self._c[_P1F]) < self.W_ROC + 5:
            return None

        roc1 = _roc(self._c[_P1F], self.W_ROC)
        roc2 = _roc(self._c[_P2F], self.W_ROC)
        if roc1 is None or roc2 is None:
            return None

        # Umbral adaptativo: más exigente si la volatilidad es alta
        vol2 = _realized_vol(self._c[_P2F], self.W_VOL) or 0.001
        threshold = max(self.ROC_DIV, vol2 * 0.5)

        divergence = roc1 - roc2
        fiat_bal   = float(balances.get("fiat",    0.0))
        t2_bal     = float(balances.get("token_2", 0.0))
        p2f = float(self._c[_P2F][-1])

        if self.step >= self.W_SLOW + 5 and divergence > threshold and fiat_bal > 0:
            qty = (fiat_bal * self.BUDGET_LAG) / (p2f * (1.0 + fee) + EPS)
            if qty > 0:
                self._mark("lag")
                strength = _clamp01((divergence - threshold) / (2.0 * threshold + EPS))
                return {
                    "pair": _P2F,
                    "side": "buy",
                    "qty": qty,
                    "consensus_score": 0.55 + 0.45 * strength,
                }

        if divergence < -threshold and t2_bal > 0:
            qty = t2_bal * self.BUDGET_LAG
            if qty > 0:
                self._mark("lag")
                strength = _clamp01((abs(divergence) - threshold) / (2.0 * threshold + EPS))
                return {
                    "pair": _P2F,
                    "side": "sell",
                    "qty": qty,
                    "consensus_score": 0.55 + 0.45 * strength,
                }

        return None

    def _spread_rev(self, balances, fee):
        """
        Z-score del spread log(ETH/fiat) - log(BTC/fiat).
        Revertir cuando se aleja demasiado de la media histórica.
        """
        if not self._ok("spread", self.CD_MR):
            return None

        z = _zscore(self._spread, self.W_SPREAD)
        if z is None:
            return None

        fiat_bal = float(balances.get("fiat",    0.0))
        t1_bal   = float(balances.get("token_1", 0.0))
        t2_bal   = float(balances.get("token_2", 0.0))
        p1f = float(self._c[_P1F][-1])
        p2f = float(self._c[_P2F][-1])

        alloc = self.BUDGET_MR * 0.5  # mitad del presupuesto MR para el spread

        # Spread alto (ETH cara vs BTC) → vender ETH
        if z > self.SPREAD_Z and t2_bal > 0:
            qty = t2_bal * alloc
            if qty > 0:
                self._mark("spread")
                strength = _clamp01((z - self.SPREAD_Z) / (self.SPREAD_Z + EPS))
                return {
                    "pair": _P2F,
                    "side": "sell",
                    "qty": qty,
                    "consensus_score": 0.55 + 0.45 * strength,
                }

        # Spread bajo (ETH barata vs BTC) → comprar ETH
        if self.step >= self.W_SLOW + 5 and z < -self.SPREAD_Z and fiat_bal > 0:
            qty = (fiat_bal * alloc) / (p2f * (1.0 + fee) + EPS)
            if qty > 0:
                self._mark("spread")
                strength = _clamp01((abs(z) - self.SPREAD_Z) / (self.SPREAD_Z + EPS))
                return {
                    "pair": _P2F,
                    "side": "buy",
                    "qty": qty,
                    "consensus_score": 0.55 + 0.45 * strength,
                }

        return None

    # ────────────────────────────────────────────────────────────────────────
    # MÓDULO 3 — Mean-Reversion Intra-Asset + Pánico (Sniper)
    # ────────────────────────────────────────────────────────────────────────

    def _mean_rev(self, pair, close, high, low, balances, fee):
        """
        Z-score intra-asset con banda adaptativa.
        La banda se amplía cuando la volatilidad realizada es alta
        (evita comprar en tendencias bajistas fuertes).
        Señal de pánico: z muy negativo + rechazo intravela (sniper).
        """
        closes = self._c[pair]
        if len(closes) < self.W_MR + self.W_VOL:
            return None

        z = _zscore(closes, self.W_MR)
        if z is None:
            return None

        # Banda adaptativa según volatilidad realizada
        rv = _realized_vol(closes, self.W_VOL) or 0.001
        rv_norm = min(rv / 0.002, 2.0)            # normalizar: 0.002 = vol "normal" en 1min
        band = self.Z_MR_BASE + rv_norm * 0.5     # se amplía en mercados volátiles

        base, quote = pair.split("/")
        base_bal  = float(balances.get(base,  0.0))
        quote_bal = float(balances.get(quote, 0.0))

        cr = high - low + EPS
        range_pos = (close - low) / cr   # posición del cierre en la vela

        panic_ready = (
            z < -self.Z_PANIC
            and range_pos > self.PANIC_RP
            and self._ok(f"panic_{pair}", self.CD_PANIC)
            and quote_bal > 0
        )
        if panic_ready and self.step >= self.W_SLOW + 5:
            alloc = min(self.BUDGET_MR, 0.35)
            qty = (quote_bal * alloc) / (close * (1.0 + fee))
            if qty > 0:
                self._mark(f"panic_{pair}")
                self._mark(f"mr_{pair}")
                panic_strength = _clamp01(abs(z) / (self.Z_PANIC + EPS))
                return {
                    "pair": pair,
                    "side": "buy",
                    "qty": qty,
                    "consensus_score": 0.65 + 0.35 * panic_strength,
                }

        # ── Mean-reversion normal ────────────────────────────────────────
        if not self._ok(f"mr_{pair}", self.CD_MR):
            return None

        # Compra: z muy negativo + vela con rechazo alcista (range_pos > 0.5)
        if self.step >= self.W_SLOW + 5 and z < -band and range_pos > 0.5 and quote_bal > 0:
            strength = min(abs(z) / band, 2.0)
            alloc = min(self.BUDGET_MR * 0.4 * strength, self.BUDGET_MR)
            qty = (quote_bal * alloc) / (close * (1.0 + fee))
            if qty > 0:
                self._mark(f"mr_{pair}")
                score = 0.5 + 0.45 * _clamp01((strength - 1.0) / 1.0)
                return {"pair": pair, "side": "buy", "qty": qty, "consensus_score": score}

        # Venta: z positivo → precio ha revertido a la media o la ha superado
        if z > band * 0.6 and base_bal > 0:
            sell_frac = min(0.3 + 0.2 * (z / band), 0.9)
            qty = base_bal * sell_frac
            if qty > 0:
                self._mark(f"mr_{pair}")
                strength = _clamp01((z - 0.6 * band) / (0.8 * band + EPS))
                score = 0.6 + 0.4 * strength
                return {"pair": pair, "side": "sell", "qty": qty, "consensus_score": score}

        return None

    # ────────────────────────────────────────────────────────────────────────
    # MÓDULO 4 — Momentum Direccional con ATR
    # ────────────────────────────────────────────────────────────────────────

    def _momentum(self, pair, close, balances, fee):
        """
        EMA crossover + ROC confirman tendencia.
        ATR dota de sizing dinámico y gestión de riesgo.
        Volumen filtra señales falsas.
        """
        closes = self._c[pair]
        highs = self._h[pair]
        lows = self._l[pair]
        volumes = self._v[pair]

        need = max(self.W_SLOW + 5, self.W_ROC + 1, self.W_ATR + 2, self.W_VOL_SMA + 1)
        if len(closes) < need or not self._ok(f"mom_{pair}", self.CD_MOM):
            return None

        fast_ema = _ema(closes, self.W_FAST)
        slow_ema = _ema(closes, self.W_SLOW)
        roc      = _roc(closes, self.W_ROC)
        need_w = max(self.W_ATR + 2, self.W_VOL_SMA + 1)
        closes_arr = np.asarray(list(closes)[-need_w:], dtype=float)
        highs_arr = np.asarray(list(highs)[-need_w:], dtype=float)
        lows_arr = np.asarray(list(lows)[-need_w:], dtype=float)
        volumes_arr = np.asarray(list(volumes)[-need_w:], dtype=float)

        atr = _atr(highs_arr, lows_arr, closes_arr, self.W_ATR)

        if None in (fast_ema, slow_ema, roc, atr) or atr <= EPS:
            return None

        vol_avg = float(np.mean(volumes_arr[-self.W_VOL_SMA:])) + EPS
        vol_ratio = float(volumes_arr[-1]) / vol_avg

        base, quote = pair.split("/")
        base_bal  = float(balances.get(base,  0.0))
        quote_bal = float(balances.get(quote, 0.0))

        # Umbral ROC adaptativo: más exigente en mercados tranquilos
        rv = _realized_vol(closes, self.W_VOL) or 0.001
        roc_thresh = max(0.003, rv * 1.5)

        # ── BUY: EMA alcista + ROC positivo + volumen elevado ────────────
        if (
            self.step >= self.W_SLOW + 5
            and fast_ema > slow_ema
            and roc > roc_thresh
            and vol_ratio > 1.15
            and quote_bal > 0
        ):
            # Sizing proporcional a calidad de señal (momentum + volumen)
            signal_q = min(roc / roc_thresh, 2.0) * min(vol_ratio / 1.15, 1.5)
            alloc = min(self.BUDGET_MOM * 0.5 * signal_q, self.BUDGET_MOM)
            qty = (quote_bal * alloc) / (close * (1.0 + fee))
            if qty > 0:
                self._mark(f"mom_{pair}")
                score = 0.55 + 0.45 * _clamp01((signal_q - 1.0) / 1.0)
                return {"pair": pair, "side": "buy", "qty": qty, "consensus_score": score}

        # ── SELL: EMA bajista + ROC negativo ────────────────────────────
        if fast_ema < slow_ema and roc < -roc_thresh and base_bal > 0:
            signal_q  = min(abs(roc) / roc_thresh, 2.0)
            sell_frac = min(0.3 + 0.3 * signal_q, 0.85)
            qty = base_bal * sell_frac
            if qty > 0:
                self._mark(f"mom_{pair}")
                score = 0.6 + 0.4 * _clamp01((signal_q - 1.0) / 1.0)
                return {"pair": pair, "side": "sell", "qty": qty, "consensus_score": score}

        return None

    # ────────────────────────────────────────────────────────────────────────
    # on_data — punto de entrada principal
    # ────────────────────────────────────────────────────────────────────────

    def on_data(self, market_data, balances):
        self.step += 1
        fee = float(market_data.get("fee", DEFAULT_FEE))
        actions = []

        # Actualizar historiales
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

        # Spread log(ETH/fiat) - log(BTC/fiat)
        if len(self._c[_P1F]) > 0 and len(self._c[_P2F]) > 0:
            self._spread.append(
                np.log(float(self._c[_P2F][-1]) + EPS)
                - np.log(float(self._c[_P1F][-1]) + EPS)
            )

        # --- NUEVO: FASE DE LIMPIEZA INICIAL ---
        # Cleanup one-shot: liquidar inventario heredado en el primer tick
        # para evitar residuos que bloqueen compras en el coordinador.
        if self.step == 1:
            limpieza = []
            if float(balances.get("token_1", 0.0)) > EPS:
                qty = float(balances.get("token_1", 0.0))
                if qty > EPS:
                    limpieza.append(
                        {
                            "pair": _P1F,
                            "side": "sell",
                            "qty": qty,
                            "bypass_pipeline": True,
                            "reason": "cleanup_liquidation",
                        }
                    )
            if float(balances.get("token_2", 0.0)) > EPS:
                qty = float(balances.get("token_2", 0.0))
                if qty > EPS:
                    limpieza.append(
                        {
                            "pair": _P2F,
                            "side": "sell",
                            "qty": qty,
                            "bypass_pipeline": True,
                            "reason": "cleanup_liquidation",
                        }
                    )

            if limpieza:
                return limpieza
        # ---------------------------------------

        # Requerir los tres pares para señales cross-asset
        has_all = all(
            len(self._c[p]) > 0 for p in [_P1F, _P2F, _P12]
        )

        # ── MÓDULO 1: Arbitraje ─────────────────────────────────────────
        if has_all and self.step >= self.W_SLOW + 5:
            p1f = float(self._c[_P1F][-1])
            p2f = float(self._c[_P2F][-1])
            p12 = float(self._c[_P12][-1])
            arb_acts = self._arb(p1f, p2f, p12, balances, fee)
            actions.extend(arb_acts)

        # Si el arbitraje ya generó señales, skipeamos el resto para no
        # interferir con el balance en el mismo tick
        if not actions:

            # ── MÓDULO 2: Lead-lag + Spread ─────────────────────────────
            if has_all:
                sig = self._lead_lag(balances, fee)
                if sig:
                    actions.append(sig)

                sig = self._spread_rev(balances, fee)
                if sig:
                    actions.append(sig)

            # ── MÓDULO 3: Mean-reversion intra-asset ────────────────────
            # Solo en pares fiat (no en token_1/token_2 para evitar exposición doble)
            for pair in [_P1F, _P2F]:
                data = market_data.get(pair)
                if data is None or len(self._c[pair]) < self.W_MR:
                    continue
                c = float(self._c[pair][-1])
                h = float(self._h[pair][-1])
                l = float(self._l[pair][-1])
                sig = self._mean_rev(pair, c, h, l, balances, fee)
                if sig:
                    actions.append(sig)

            # ── MÓDULO 4: Momentum ──────────────────────────────────────
            for pair in [_P1F, _P2F]:
                data = market_data.get(pair)
                if data is None:
                    continue
                c = float(self._c[pair][-1])
                sig = self._momentum(pair, c, balances, fee)
                if sig:
                    actions.append(sig)

        return actions if actions else None