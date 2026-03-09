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

        # Refresh position tracking from balances
        open_pairs = set()
        for sig in signals:
            pair = sig["pair"]
            base, _ = pair.split("/")
            if float(balances.get(base, 0.0)) > EPS:
                open_pairs.add(pair)
                if pair not in self._position_open_step:
                    self._position_open_step[pair] = self._step

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
                if has_pos and sig.get("consensus_score", 0) < 0.8:
                    continue  # already holding, consensus not strong enough
            elif side == "sell":
                if not has_pos:
                    continue  # nothing to sell

            accepted.append(sig)

        # Record accepted signals for cooldown tracking
        for sig in accepted:
            self._last_signal_step[sig["pair"]] = self._step
            base, _ = sig["pair"].split("/")
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
            if float(balances.get(base, 0.0)) > EPS:
                accepted.append({
                    "pair": pair,
                    "side": "sell",
                    "consensus_score": 0.5,
                    "n_votes": 0,
                })
                self._last_signal_step[pair] = self._step
                del self._position_open_step[pair]

        return accepted
