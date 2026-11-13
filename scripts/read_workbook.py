"""Read-only source extraction with the bundled spreadsheet Python runtime."""
from pathlib import Path
import pandas as pd
root = Path(__file__).resolve().parents[1]
source = root / 'data/raw/avito_clean_final_full.xlsx'
frame = pd.read_excel(source, engine='openpyxl')
frame.to_csv(root / 'data/raw/listings.csv', index=False)
print('shape', frame.shape)
print(frame.dtypes.to_string())
print(frame[['price','Общая площадь','description','Ремонт','Мебель']].head(3).to_json(orient='records', force_ascii=False))
