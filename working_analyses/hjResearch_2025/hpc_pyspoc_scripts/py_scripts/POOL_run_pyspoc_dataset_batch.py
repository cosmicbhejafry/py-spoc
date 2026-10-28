"""
Compute pyspoc statistics for one chunk of datasets from a manifest.
Create a sub-process per dataset, to catch cases where the computation goes out of memory
without killing the entire batch job

Usage (from hpc_pipeline/):
    python py_scripts/run.py --manifest metadata_files/synthetic.csv \
        --config configs/stats.yaml --index 0 --n-chunks 50

Array index i processes manifest rows i, i+n_chunks, i+2*n_chunks, ...
Per dataset it writes, to results/per_dataset/<manifest>/<config>/:
    <safe_data_id>.parquet      results (skipped on rerun if it exists)
    <safe_data_id>.errors.json  only if there were errors or a crash
"""
import argparse
import json
import logging
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from pyspoc import Calculator, Config

import os, shutil

from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
# from concurrent.futures import TimeoutError as FutureTimeout

def compute_one(args_dict):
    # path, name, config, config_name
    path = args_dict['path']
    name = args_dict['name']
    config = args_dict['config']
    config_name = args_dict['config_name']
    
    cfg = Config.from_yaml_file(config_name, config)
    calc = Calculator(dataset=np.load(path), name=name)
    calc.compute(cfg)
    errors = {k: repr(e) for k, e in calc.get_errors().items() if e}
    return calc.get_results(), errors
    
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--config_name", default="stats")    # first arg of Config.from_yaml
    p.add_argument("--index", required=True, type=int)
    p.add_argument("--n_chunks", required=True, type=int)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--manifest_data_id", default="safe_data_id",type=str)
    p.add_argument("--manifest_path", default="path",type=str)
    # MANIFEST NEEDS TO HAVE A SAFE_DATA_ID AND A PATH COL

    args = p.parse_args()

    manifest = ROOT / args.manifest
    config = ROOT / args.config
    out_dir = ROOT / "results" / "per_dataset" / manifest.stem / config.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    chunk = pd.read_csv(manifest).iloc[args.index :: args.n_chunks]
    log.info(f"chunk {args.index}/{args.n_chunks}: {len(chunk)} datasets -> {out_dir}")
    
    # COPY CONFIG TO THE HEAD OF THE OUT_DIR EACH TIME, TO ENSURE THAT SAME CONFIG GENERATES EACH OUT
    cfg_copy = out_dir / "config_used.yml"
    if not cfg_copy.exists():
        tmp = out_dir / f".config_used.{os.getpid()}.tmp" # unique per job
        shutil.copy(config, tmp)
        os.replace(tmp, cfg_copy) # atomic
    elif cfg_copy.read_text() != config.read_text():
        log.warning(f"{config} differs from the config used for existing results in {out_dir}")
    
    # ITERATE THRU DATASET BATCHES    
    for _, row in chunk.iterrows():
        v = row[args.manifest_data_id]
        out = out_dir / f"{v}.parquet"
        if out.exists() and not args.overwrite:
            log.info(f"skip {v}")
            continue

        t0 = time.time()
        errors = {}

        try:
            in_args = {}
            in_args['path'] = str(row[args.manifest_path])
            in_args['name'] = row[args.manifest_data_id]
            in_args['config'] = config
            in_args['config_name'] = args.config_name
                        
            with ProcessPoolExecutor(max_workers=1) as ex:
                results, errors = ex.submit(compute_one, in_args).result()                
                if results is not None:
                    results.to_parquet(out)
                status = "ok"
        except BrokenProcessPool:
            errors, status = {"crash": "child killed (likely out of memory)"}, "failed"
        except Exception:
            errors = {"crash": traceback.format_exc()}
            status = "failed"

        if errors:
            (out_dir / f"{v}.errors.json").write_text(json.dumps(errors, indent=2))
        log.info(f"{status} {v} ({time.time() - t0:.1f}s, {len(errors)} errors)")

if __name__ == "__main__":
    main()
    
# COULD ADD A TIMEOUT TO THIS PER-DATASET IF NEEDED
# except FutureTimeout:
#     errors, status = {"crash": "timed out"}, "failed"
