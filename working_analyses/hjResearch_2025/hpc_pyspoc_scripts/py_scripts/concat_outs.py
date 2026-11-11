#!/usr/bin/env python
"""combine the per-dataset parquet files of each config into one parquet file per config."""
import argparse
import os
import time

import pyarrow.dataset as ds
import pyarrow.parquet as pq

# $EPHEMERAL/nov2026_pyspoc_results/results/concat_per_config

p = argparse.ArgumentParser()
p.add_argument("--data_base", required=True)
p.add_argument("--data_name", required=True)
p.add_argument("--out_dir", required=True)
args = p.parse_args()

data_out_dir = os.path.join(args.out_dir,args.data_name)
os.makedirs(data_out_dir, exist_ok=True)

data_path = os.path.join(args.data_base,args.data_name)

configs = os.listdir(data_path)
for cfg in configs:
    t0 = time.time()
    base = os.path.join(data_path, cfg)
    files = sorted(e.path for e in os.scandir(base) if e.name.endswith(".parquet"))
    if not files:
        print(f"{cfg}: no parquet files in {base}", flush=True)
        continue

    table = ds.dataset(files, format="parquet").to_table()
    pq.write_table(table, os.path.join(data_out_dir, f"{cfg}.parquet"))
    print(f"{cfg}: {len(files)} files, {table.num_rows} rows, {time.time() - t0:.0f}s", flush=True)
    del table