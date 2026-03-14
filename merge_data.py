import pandas as pd

processed = pd.read_json('processed_data.json')
transformed = pd.read_json('transformed_data.json')

# Add positional index to transformed (matches _orig_idx in processed since both come from data.json)
transformed['_orig_idx'] = range(len(transformed))

# Columns from processed to overlay onto transformed (everything except the join key)
PROCESSED_COLS = [c for c in processed.columns if c != '_orig_idx']

merged = transformed.merge(
    processed[['_orig_idx'] + PROCESSED_COLS],
    on='_orig_idx',
    how='left',
    suffixes=('_orig', '_proc'),
)

# For overlapping columns, prefer the processed (enriched) value when available
for col in PROCESSED_COLS:
    orig_col = f'{col}_orig'
    proc_col = f'{col}_proc'
    if orig_col in merged.columns:
        merged[col] = merged[proc_col].combine_first(merged[orig_col])
        merged.drop(columns=[orig_col, proc_col], inplace=True)

merged.drop(columns=['_orig_idx'], inplace=True)

merged.to_json('final_processed_data.json', orient='records', indent=2)

print(f"Final dataset: {len(merged)} companies, {len(merged.columns)} columns")
print()
print("Null counts:")
for col in merged.columns:
    n = merged[col].isnull().sum()
    if n > 0:
        print(f"  {col}: {n}")
