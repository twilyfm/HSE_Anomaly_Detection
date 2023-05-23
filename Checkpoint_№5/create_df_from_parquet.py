import pandas as pd
import os

path="parquet.csv"
if os.path.exists(path):
    os.remove(path)
N = 100 # rows
test_file = pd.read_parquet("merge_filled_without_drop.parquet", engine='pyarrow')
test_file = test_file.sample(n = N)
test_file.to_csv(path, index=False)