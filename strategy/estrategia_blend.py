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
        # Número total de estrategias que aportan señales.
        # Se utiliza para normalizar la `consensus_score`.
        n_strategies = len(all_signals)
        if n_strategies == 0:
            return []

        # Tally votes: pair → {"buy": int, "sell": int}
        # Diccionario que acumula los votos por par. Cada par tendrá
        # un contador independiente para "buy" y "sell".
        votes = defaultdict(lambda: {"buy": 0, "sell": 0})

        for _name, signals in all_signals.items():
            # Cada estrategia sólo puede emitir un voto por par. `seen_pairs`
            # evita contar múltiples señales del mismo par por una estrategia.
            seen_pairs = set()
            for sig in signals:
                pair = sig["pair"]
                side = sig["side"]  # se espera 'buy' o 'sell'
                if pair in seen_pairs:
                    continue  # una estrategia ya votó este par
                seen_pairs.add(pair)
                # Incrementa el contador correspondiente (buy/sell)
                votes[pair][side] += 1

        # Resolve consensus per pair
        consensus = []
        for pair, tally in votes.items():
            buy_v = tally["buy"]
            sell_v = tally["sell"]
            if buy_v == 0 and sell_v == 0:
                continue
            if buy_v == sell_v:
                # Empate: ninguna dirección domina, se abstiene
                continue

            if buy_v > sell_v:
                side, n = "buy", buy_v
            else:
                side, n = "sell", sell_v

            consensus.append({
                "pair": pair,
                "side": side,
                # `consensus_score` es la fracción de estrategias que
                # apoyaron la decisión ganadora (0..1). `n_votes` es
                # el número absoluto de votos favorables.
                "consensus_score": n / n_strategies,
                "n_votes": n,
            })

        return self._coherence_filter(consensus)

    # ------------------------------------------------------------------
    @staticmethod
    def _coherence_filter(signals):
        """Filtro de coherencia.

        Algunas parejas de mercado están correlacionadas entre sí. Si dentro
        de un grupo correlacionado hay señales activas que apuntan en
        direcciones opuestas (por ejemplo, una estrategia sugiere "buy" en
        `token_1/fiat` y otra sugiere "sell" en `token_2/fiat`), se considera
        una contradicción y se descartan ambas señales para evitar operar
        sobre ruido o arbitraje no deseado.

        Args:
            signals: lista de dicts con claves `pair` y `side`.

        Returns:
            Lista filtrada de señales donde las contradicciones dentro de los
            grupos correlacionados han sido eliminadas.
        """
        # Mapa rápido de par → señal para lookup eficiente
        sig_map = {s["pair"]: s for s in signals}
        drop = set()

        # Recorre cada grupo de pares correlacionados
        for group in CORRELATED_GROUPS:
            # `active` son los pares del grupo que tienen señal en `sig_map`
            active = [p for p in group if p in sig_map]
            if len(active) < 2:
                # Si hay menos de 2 pares activos no hay posibilidad de
                # contradicción dentro del grupo.
                continue
            # Conjunto de direcciones activas en el grupo (ej: {'buy', 'sell'})
            sides = {sig_map[p]["side"] for p in active}
            if len(sides) > 1:
                # Hay contradicción → marcar todos los pares activos para
                # que sean descartados.
                drop.update(active)

        # Devolver señales cuyo par no esté en la lista de descartes
        return [s for s in signals if s["pair"] not in drop]
