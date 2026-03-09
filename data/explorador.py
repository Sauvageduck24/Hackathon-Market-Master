
import pandas as pd


def main(path='data/submission.csv'):
	df = pd.read_csv(path, parse_dates=['timestamp'])

	print('\n=== Head ===')
	print(df.head())

	n_rows = len(df)
	print(f'\n=== Filas: {n_rows} ===')

	# Contar operaciones válidas buy->sell por par (ignorar sell si no hay buy previo)
	df = df.sort_values('timestamp')
	holding = {}  # holding[pair] -> bool
	counts = {}

	for _, row in df.iterrows():
		pair = row.get('pair')
		side = str(row.get('side')).strip().lower()

		if side == 'buy':
			if not holding.get(pair, False):
				holding[pair] = True

		elif side == 'sell':
			if holding.get(pair, False):
				counts[pair] = counts.get(pair, 0) + 1
				holding[pair] = False

	total = sum(counts.values())

	print('\n=== Operaciones buy->sell por par ===')
	if counts:
		for pair, c in counts.items():
			print(f'{pair}: {c}')
	else:
		print('(ninguna)')

	print(f'\n=== Total operaciones buy->sell: {total} ===')


if __name__ == '__main__':
	main()

