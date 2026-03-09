"""Signal blender — combines raw signals from N strategies into one consensus
signal per pair.

Blending rules (no ML, no learned weights):
  - Equal-weight voting: each strategy casting a buy or sell for a pair counts
    as one vote.  The side with more votes wins.  Ties are skipped.
  - Consensus score = winning_votes / total_strategies  (0..1).
  - Coherence filter: if two correlated pairs produce contradictory consensus
    directions (e.g. buy token_1/fiat, sell token_2/fiat), both are dropped to
    avoid acting on noise.
"""

from collections import defaultdict

# Groups of pairs that are positively correlated.
# Contradictory consensus signals within a group are suppressed.
CORRELATED_GROUPS = [
    {"token_1/fiat", "token_2/fiat"},
]


class SignalBlender:
    """Stateless blender — no learned parameters, pure voting."""

    def combine(self, all_signals, market_data):
        """Blend signals from N strategies into consensus signals.

        Args:
            all_signals: {strategy_name: [{"pair", "side", ...}, ...]}
            market_data: current market snapshot (used only for pair list).

        Returns:
            List of {"pair", "side", "consensus_score", "n_votes"}.
        """
        n_strategies = len(all_signals)
        if n_strategies == 0:
            return []

        # Tally votes: pair → {"buy": int, "sell": int}
        votes = defaultdict(lambda: {"buy": 0, "sell": 0})

        for _name, signals in all_signals.items():
            seen_pairs = set()
            for sig in signals:
                pair = sig["pair"]
                side = sig["side"]
                if pair in seen_pairs:
                    continue  # one vote per pair per strategy
                seen_pairs.add(pair)
                votes[pair][side] += 1

        # Resolve consensus per pair
        consensus = []
        for pair, tally in votes.items():
            buy_v = tally["buy"]
            sell_v = tally["sell"]
            if buy_v == 0 and sell_v == 0:
                continue
            if buy_v == sell_v:
                continue  # tie → abstain

            if buy_v > sell_v:
                side, n = "buy", buy_v
            else:
                side, n = "sell", sell_v

            consensus.append({
                "pair": pair,
                "side": side,
                "consensus_score": n / n_strategies,
                "n_votes": n,
            })

        return self._coherence_filter(consensus)

    # ------------------------------------------------------------------
    @staticmethod
    def _coherence_filter(signals):
        """Drop signals from correlated pairs that disagree on direction."""
        sig_map = {s["pair"]: s for s in signals}
        drop = set()

        for group in CORRELATED_GROUPS:
            active = [p for p in group if p in sig_map]
            if len(active) < 2:
                continue
            sides = {sig_map[p]["side"] for p in active}
            if len(sides) > 1:          # contradiction
                drop.update(active)

        return [s for s in signals if s["pair"] not in drop]
