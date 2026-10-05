import os, json
import numpy as np
import pandas as pd
import shutil, glob

def _js(o):
    return o.tolist() if hasattr(o, "tolist") else str(o)

def save_dataset(X, generator, params, seed, root, category=None,extra_arrays=None):
    """
    Save one dataset as {root}/{generator}/params{k}/seed{seed}_N{n}_P{p}.npy
    and append a row to {root}/manifest.csv.

    params       : dict of generator settings (excluding seed). Identical params
                   -> same cfg index, at every size.
    """
    X = np.asarray(X)
    n, p = X.shape

    # cfg index: append-only registry per generator, shared across all sizes
    os.makedirs(root, exist_ok=True)
    
    # make_configs_json = False
    # if make_configs_json is True:
    reg_path = os.path.join(root, "configs.json")
    registry = json.load(open(reg_path)) if os.path.exists(reg_path) else {}
    cfgs = registry.setdefault(generator, [])
    canon = json.loads(json.dumps(params, sort_keys=True, default=_js))
    if canon not in cfgs:
        cfgs.append(canon)
    cfg = f"params{cfgs.index(canon)}"
    json.dump(registry, open(reg_path, "w"), indent=2)

    # folder: N_P / generator / cfg
    rel_dir = os.path.join(generator, cfg)
    folder = os.path.join(root, rel_dir)
    os.makedirs(folder, exist_ok=True)

    # params.json written once per config folder
    pj = os.path.join(folder, "params.json")
    if not os.path.exists(pj):
        json.dump({"generator": generator, "category": category, **canon},
                  open(pj, "w"), indent=2)

    # data (refuse to overwrite)
    fname = f"seed{seed}_N{n}_P{p}.npy"
    path = os.path.join(folder, fname)
    overwrite = True # SWITCH THIS TO TURN OFF OVERRIDE
    if overwrite == False:
        if os.path.exists(path):
            raise FileExistsError(f"{path} already exists")
    np.save(path, X)

    extra_files = []
    if extra_arrays:
        pdir = os.path.join(folder, "parameter_info")
        os.makedirs(pdir, exist_ok=True)
        for k, v in extra_arrays.items():
            fy = f"{k}_seed{seed}.npy"
            np.save(os.path.join(pdir, fy), v)
            extra_files.append(fy)

    # manifest row
    dataset_id = f"N{n}_P{p}/{generator}/{cfg}/seed{seed}"
    row = {
        "dataset_id": dataset_id,
        "generator": generator,
        "category": category,
        "n_rows": n,
        "n_cols": p,
        "params_id": cfg,
        "seed": seed,
        "extra_files": json.dumps(extra_files) if extra_files else ""        
        # "params": json.dumps(canon, default=_js)
        }
    man_path = os.path.join(root, "manifest.csv")
    if os.path.exists(man_path):
        df = pd.read_csv(man_path)
        if dataset_id in set(df.dataset_id):
            raise ValueError(f"{dataset_id} already in manifest")
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(man_path, index=False)

    return dataset_id

def delete_dataset(dataset_folder_name, root):
    root = str(root)

    # 1. delete the generator folder under every N_P folder
    for d in glob.glob(os.path.join(root,dataset_folder_name)):
        shutil.rmtree(d)

    # 2. drop its manifest rows
    man_path = os.path.join(root, "manifest.csv")
    if os.path.exists(man_path):
        df = pd.read_csv(man_path)
        df[df.generator != dataset_folder_name].to_csv(man_path, index=False)

    # 3. drop its configs
    reg_path = os.path.join(root, "configs.json")
    if os.path.exists(reg_path):
        reg = json.load(open(reg_path))
        reg.pop(dataset_folder_name, None)
        json.dump(reg, open(reg_path, "w"), indent=2)