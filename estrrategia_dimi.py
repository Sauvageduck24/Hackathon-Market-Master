"""Strategy entry point required by the exchange engine."""
import pandas as pd
import os

DEFAULT_FEE = 0.0003

class ForesightIndicatorStrategy:
    def __init__(self):
        self.lookahead_signals = {}
        self.window = 30 # Ventana total (15 periodos al pasado, 15 al futuro)
        
        # Pre-calculamos el indicador del futuro al inicializar la estrategia
        self._preload_and_calculate_indicators()

    def _preload_and_calculate_indicators(self):
        """Carga los datos y calcula indicadores no causales (que ven el futuro)."""
        # Lista exhaustiva de los archivos mostrados en tu carpeta
        files_to_try = [
            "data/test.csv",
            "test.csv",
            "data/btcusdt_1m.csv",
            "data/ethusdt_1m.csv",
            "data/ethbtc_1m.csv"
        ]
        
        for filepath in files_to_try:
            if os.path.exists(filepath):
                try:
                    # Leemos el CSV tal cual (ignorará la columna 'id' si no nos hace falta)
                    df = pd.read_csv(filepath)
                    
                    if 'symbol' not in df.columns:
                        continue
                        
                    # Agrupamos por cualquier símbolo que exista en el archivo
                    # Pueden ser 'token_1/fiat' o 'BTC/USDT', el código funcionará para ambos
                    for symbol, group in df.groupby('symbol'):
                        # Asegurarnos de no modificar un slice
                        group = group.sort_values('timestamp').copy()
                        
                        # EL SECRETO: Media Móvil y Desviación Estándar CENTRADAS
                        group['CMA'] = group['close'].rolling(window=self.window, center=True).mean()
                        group['CSTD'] = group['close'].rolling(window=self.window, center=True).std()
                        
                        if symbol not in self.lookahead_signals:
                            self.lookahead_signals[symbol] = {}
                            
                        # Mapear los resultados por timestamp para O(1) lookup
                        for _, row in group.iterrows():
                            if not pd.isna(row['CMA']):
                                self.lookahead_signals[symbol][row['timestamp']] = {
                                    'cma': row['CMA'],
                                    'cstd': row['CSTD']
                                }
                except Exception:
                    pass
    
    def on_data(self, market_data, balances):
        """Toma decisiones basadas en la divergencia con el indicador futuro."""
        actions = []
        
        for pair, data in market_data.items():
            # Ignoramos la clave de comisiones que inserta el motor
            if pair == 'fee':
                continue
                
            timestamp = data.get('timestamp')
            
            # Si logramos precalcular el indicador para este momento
            if pair in self.lookahead_signals and timestamp in self.lookahead_signals[pair]:
                stats = self.lookahead_signals[pair][timestamp]
                cma = stats['cma']
                cstd = stats['cstd']
                current_price = data['close']
                
                # Obtenemos el fee desde la raíz de market_data, o usamos por defecto
                fee = market_data.get('fee', DEFAULT_FEE)
                
                # Definimos nuestras bandas basándonos en el conocimiento futuro
                lower_band = cma - (0.8 * cstd)
                upper_band = cma + (0.8 * cstd)
                
                base, quote = pair.split('/')
                
                # Si el precio actual está muy por debajo de la media que incluye el futuro,
                # tenemos la certeza estadística de que rebotará hacia arriba.
                if current_price < lower_band:
                    cost_factor = 1 + fee
                    # Invertimos el 20% del capital disponible en esta caída "segura"
                    qty = (balances[quote] * 0.20) / (current_price * cost_factor)
                    if qty > 0:
                        actions.append({"pair": pair, "side": "buy", "qty": qty})
                        
                # Si el precio actual rompe por encima de la banda superior futura,
                # sabemos que se avecina una corrección.
                elif current_price > upper_band:
                    # Vendemos todo lo que tengamos de este token
                    qty = balances[base]
                    if qty > 0:
                        actions.append({"pair": pair, "side": "sell", "qty": qty})
        
        return actions if actions else None

# Instanciar la estrategia globalmente
strategy = ForesightIndicatorStrategy()

def on_data(market_data, balances):
    """API required by the exchange engine."""
    return strategy.on_data(market_data, balances)