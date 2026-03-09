"""Position coordinator — rules-based filter that sits between the blender
and the risk manager.

Responsibilities:
  - Track whether a position is currently open for each pair (inferred from
    balances: base_balance > threshold → position is open).
  - Enforce a cooldown: prevent re-entering the same pair too soon after the
    last trade signal was forwarded.
  - Alignment check: only allow *buy* when flat, only allow *sell* when
    holding.  Exception: strong consensus (score >= 0.8) may add to an
    existing long.
  - Force-exit: if a position has been held longer than *max_hold_steps*,
    inject a sell signal so the risk manager can close it.
"""

EPS = 1e-9


class PositionCoordinator:
    """Stateful filter — tracks per-pair position age and cooldowns."""

    def __init__(self, cooldown_steps=5, max_hold_steps=120):
        self.cooldown_steps = cooldown_steps
        self.max_hold_steps = max_hold_steps

        self._last_signal_step = {}      # pair → step when last signal passed
        self._position_open_step = {}    # pair → step when position was opened
        self._step = 0

    # ------------------------------------------------------------------
    def filter(self, signals, balances):
        """Return the subset of *signals* that pass position rules.

        Also injects force-sell signals for aged-out positions.

        Args:
            signals: list of {"pair", "side", "consensus_score", "n_votes"}.
            balances: current balances dict.

        Returns:
            Filtered (possibly augmented) list of signal dicts.
        """

        self._step += 1
        accepted = []

        # Build open_pairs directly from balances — NOT from incoming signals.
        # This is the only source of truth: if we hold base asset, position is open.
        open_pairs = set()
        for sig in signals:
            pair = sig["pair"]
            base, _ = pair.split("/")
            # Si el balance del activo base es mayor que EPS, consideramos
            # que hay una posición abierta para ese par.
            if float(balances.get(base, 0.0)) > EPS:
                open_pairs.add(pair)
                # Si acabamos de detectar la posición, registrar el paso
                # en el que se abrió para poder aplicar reglas de tiempo.
                if pair not in self._position_open_step:
                    self._position_open_step[pair] = self._step

        # Also check pairs NOT in current signals — positions may exist without
        # a signal this tick (e.g. held from previous steps).
        for pair, opened in list(self._position_open_step.items()):
            base, _ = pair.split("/")
            # Comprueba balances actuales para pares que no están en `signals`.
            # Si el balance indica que la posición ya no existe, limpiar
            # el estado interno para evitar fugas de memoria.
            if float(balances.get(base, 0.0)) > EPS:
                open_pairs.add(pair)
            else:
                # La posición fue cerrada externamente — eliminar seguimiento.
                del self._position_open_step[pair]

        print("BALANCES:", balances)
        print("OPEN_PAIRS:", open_pairs)
        print("POSITION_OPEN_STEP:", self._position_open_step)

        for sig in signals:
            pair = sig["pair"]
            side = sig["side"]
            base, _ = pair.split("/")
            has_pos = pair in open_pairs

            # Cooldown
            last = self._last_signal_step.get(pair, -10_000)
            if (self._step - last) < self.cooldown_steps:
                continue

            if side == "buy":
                # Sólo permitir `buy` si no tenemos posición abierta. Si ya
                # estamos dentro, sólo permitir añadir más si la señal es
                # muy fuerte (consenso >= 0.8).
                if has_pos and sig.get("consensus_score", 0) < 0.8:
                    continue
            elif side == "sell":
                # No permitir `sell` si no hay posición (nada que vender).
                if not has_pos:
                    continue

            # Registro rápido para depuración en desarrollo — muestra la
            # señal y si actualmente hay posición para ese par.
            print(sig, has_pos)

            accepted.append(sig)

        # Record accepted signals for cooldown tracking
        for sig in accepted:
            self._last_signal_step[sig["pair"]] = self._step
            base, _ = sig["pair"].split("/")
            # Actualizar estado por señales aceptadas:
            # - Si aceptamos un `buy` y no había posición, registrar apertura.
            # - Si aceptamos un `sell`, eliminar el registro de posición.
            if sig["side"] == "buy" and sig["pair"] not in open_pairs:
                self._position_open_step[sig["pair"]] = self._step
            elif sig["side"] == "sell":
                self._position_open_step.pop(sig["pair"], None)

        # Force-exit aged positions
        for pair, opened in list(self._position_open_step.items()):
            if (self._step - opened) <= self.max_hold_steps:
                continue
            already = any(s["pair"] == pair and s["side"] == "sell"
                          for s in accepted)
            if already:
                continue
            base, _ = pair.split("/")
            # Si la posición excede `max_hold_steps`, inyectar una señal de
            # `sell` de fuerza media para que el gestor de riesgo la cierre.
            if float(balances.get(base, 0.0)) > EPS:
                accepted.append({
                    "pair": pair,
                    "side": "sell",
                    # Score genérico para indicar que es una señal forzada.
                    "consensus_score": 0.5,
                    "n_votes": 0,
                })
                # Actualizar cooldown y limpiar estado de posición.
                self._last_signal_step[pair] = self._step
                del self._position_open_step[pair]

        return accepted