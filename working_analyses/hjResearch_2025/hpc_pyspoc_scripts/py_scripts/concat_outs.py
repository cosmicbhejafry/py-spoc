#!/usr/bin/env python
"""combine the per-dataset parquet files of each config into one parquet file per config."""
# TODO: FOR FUTURE- SAVING AND USING IN 'LONG' FORMAT IS BETTER

import argparse
import os
import time

import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pyarrow as pa

p = argparse.ArgumentParser()
p.add_argument("--data_base", required=True)
p.add_argument("--data_name", required=True)
p.add_argument("--out_dir", required=True)
p.add_argument("--batch_len", default=100)
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

    # schema = pq.read_schema(files[0]) # template from one file
    # table = ds.dataset(files, format="parquet",schema=schema).to_table().combine_chunks()    
    # pq.write_table(table, os.path.join(data_out_dir, f"{cfg}.parquet"))
    
    
    blocks = []
    batch_num = int(args.batch_len)
    for k in range(0, len(files), batch_num):
        
        t = ds.dataset(files[k:k + batch_num], format="parquet").to_table()
        print('to table done')
        blocks.append(t.combine_chunks())
        print('combine chunk done')
        del t
        print(f"{cfg}: {len(blocks)} done",flush=True)
        
    table = pa.concat_tables(blocks).combine_chunks()
        
    print(f"{cfg}: {len(files)} files, {table.num_rows} rows, {time.time() - t0:.0f}s", flush=True)
    del table