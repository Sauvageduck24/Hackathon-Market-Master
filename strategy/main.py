"""Strategy entry point required by the exchange engine."""

from .estrategia_bollinger import Strategy

strategy = Strategy()

def on_data(market_data, balances):
    """API required by the exchange engine."""
    return strategy.on_data(market_data, balances)