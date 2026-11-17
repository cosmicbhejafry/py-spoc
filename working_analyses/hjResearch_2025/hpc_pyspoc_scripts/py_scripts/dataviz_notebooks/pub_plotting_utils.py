"""plotting utilities."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd
import seaborn as sns

import math

from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import re

def plot_clustermap(X, row_Z, col_Z, row_colors=None, col_colors=None, title=None,
                    save_path=None, vlim=2, figsize=(16, 12), cbar_label="z-score"):
    """Clustered heatmap with precomputed linkages, colour strips and a colour bar at the bottom.

    X: DataFrame (rows × columns). row_Z, col_Z: scipy linkage matrices.
    row_colors, col_colors: Series/DataFrame of colours, indexed like X.
    Returns the seaborn ClusterGrid.
    """
    g = sns.clustermap(
        X, row_linkage=row_Z, col_linkage=col_Z,
        row_colors=row_colors, col_colors=col_colors,
        cmap="vlag", center=0, vmin=-vlim, vmax=vlim,
        xticklabels=False, yticklabels=False,
        figsize=figsize, dendrogram_ratio=(0.15, 0.15), colors_ratio=0.02,
        cbar_kws={"orientation": "horizontal", "label": cbar_label},
        rasterized=True,
    )
    g.gs.update(bottom=0.12)
    pos = g.ax_heatmap.get_position()
    g.ax_cbar.set_position([pos.x0, 0.04, pos.width, 0.015])
    if title:
        g.figure.suptitle(title, y=1.05)
    if save_path:
        g.savefig(save_path, dpi=300, bbox_inches="tight")
    return g
    
def extract_pyspoc_class_names(s, sep=".",add_scheme_str=True):
    config, rest = s.split("__", 1)
    toks = re.split(r"\.|__", rest)
    
    join_lst = []
    for i,t in enumerate(toks):
        if t[:1].isupper():
            
            if add_scheme_str:
                scheme_t = toks[i+1]
                t_app = t+'__'+scheme_t # could add the string right after class name
            else:
                t_app = t
            
            join_lst.append(t_app)

    if len(join_lst)==0:
        return s
        
    return sep.join(join_lst)

def get_nn_label_agreement(pca_knn,df_label_maps,permut_mode=False,n_permut=100,permut_seedval=23829):
    
    vc_1_prop = {}

    ids = pca_knn["dataset"]
    pos = {d: i for i, d in enumerate(ids)}
    idx = np.array([[pos[n] for n in nbs] for nbs in pca_knn["neighbors"]])
        
    check_same = lambda lab : (lab[idx] == lab[:, None]).all(axis=1)
    
    if permut_mode == False:    
        for sv, lab in df_label_maps.items():        
            lab = np.asarray(lab)
            all_same = check_same(lab) # boolean arr, pos True if every neighbor has the dataset's own label        
            vc_1_prop[sv] = all_same.mean()
            
        return vc_1_prop

    else:
        seedval=permut_seedval
        rng = np.random.default_rng(seedval)
        
        for sv, lab in df_label_maps.items():  
            all_s_lst = []
            for j in range(n_permut):
                lab_perm = rng.permutation(lab)
                all_same = check_same(lab_perm)
                all_s_lst.append(all_same.mean())
                
            vc_1_prop[sv] = np.array(all_s_lst).mean()
            
        return vc_1_prop    

def get_pca_knn_one(k,var_exp,df_module_cluster,df_label_maps):

    X = StandardScaler().fit_transform(df_module_cluster)

    pca = PCA(n_components=var_exp)
    Z = pca.fit_transform(X)
    n_comps = pca.n_components_
    # print(f'{n_comps} pca components')
    
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Z)
    dist, idx = nn.kneighbors(Z)
    idx, dist = idx[:, 1:], dist[:, 1:]

    ids = df_module_cluster.index.to_numpy()


    df_dict = {
        "dataset": ids,
        "neighbors": [list(ids[i]) for i in idx],
        "mean_dist": dist.mean(axis=1)
    }
    
    for kv in df_label_maps:
        df_dict[f"nb_{kv}_set"] = [set(df_label_maps[kv][i]) for i in idx]
    
    pca_knn = pd.DataFrame(df_dict)

    return pca_knn,n_comps

def plot_2d_embedding_panels(E, labels, per_panel=3, ncols=3, method="tSNE", order=None,
                          save_path=None, panel_size=(6, 4.5), s=12):
    """Small-multiples 2D embedding for many labels: each panel highlights `per_panel`
    labels in colour over all points in grey. Returns fig, axes.

    E: (n_samples, >=2) embedding coordinates. labels: array-like of group labels.
    order: label order across panels (default: sorted).
    """
    labels = pd.Series(labels).to_numpy()
    cats = list(order) if order is not None else sorted(pd.unique(labels))
    chunks = [cats[i:i + per_panel] for i in range(0, len(cats), per_panel)]
    nrows = math.ceil(len(chunks) / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_size[0] * ncols, panel_size[1] * nrows),
                             sharex=True, sharey=True, squeeze=False)
    for ax, chunk in zip(axes.flat, chunks):
        ax.scatter(E[:, 0], E[:, 1], c="lightgrey", s=s / 2, linewidth=0)       # background
        for c, col in zip(chunk, sns.color_palette("tab10", len(chunk))):
            m = labels == c
            ax.scatter(E[m, 0], E[m, 1], color=col, s=s, alpha=0.8, linewidth=0, label=c)
        ax.legend(fontsize=7, markerscale=1.5, loc="best", frameon=False)
        ax.set(xticks=[], yticks=[])
    for ax in axes.flat[len(chunks):]:                                         # hide unused panels
        ax.axis("off")
    for ax in axes[-1]: ax.set_xlabel(f"{method}1")
    for ax in axes[:, 0]: ax.set_ylabel(f"{method}2")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, axes
    
def plot_2d_embedding(E, labels, method="tSNE", hue_order=None, palette="tab10",
                   save_path=None, figsize=(8, 5), legend_ncol=4, s=12):
    """2D embedding scatter (t-SNE, UMAP, ...) coloured by `labels`.

    E: (n_samples, >=2) embedding coordinates. labels: array-like of group labels.
    save_path: optional path (e.g. FIG_DIR / "tSNE_coarse.png"). Returns fig, ax.
    """
    x, y = f"{method}1", f"{method}2"
    df = pd.DataFrame({x: E[:, 0], y: E[:, 1], "group": pd.Series(labels).to_numpy()})

    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(data=df, x=x, y=y, hue="group", hue_order=hue_order,
                    palette=palette, s=s, alpha=0.8, linewidth=0, ax=ax)
    ax.set(xticks=[], yticks=[])                       # embedding axes have no units
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, markerscale=1.5, ncol=legend_ncol, frameon=False,
              loc="upper center", bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, ax

def plot_pc_pairs(Z, labels, n_pcs=4, hue_order=None, palette=None, evr=None,
                  save_path=None, height=2.2, legend_ncol=2):
    """Lower-triangle pair plot of the first `n_pcs` PCs, coloured by `labels`.

    Z: (n_samples, n_components) PC scores. labels: array-like of group labels.
    evr: optional explained-variance ratios, added to the axis labels as %.
    save_path: optional path (e.g. FIG_DIR / "PCA_coarse.png"). Returns the PairGrid.
    """
    names = [f"PC{i + 1}" + (f" ({evr[i]:.1%})" if evr is not None else "") for i in range(n_pcs)]
    df = pd.DataFrame(Z[:, :n_pcs], columns=names).assign(group=pd.Series(labels).to_numpy())

    g = sns.PairGrid(df, x_vars=names[:-1], y_vars=names[1:], hue="group",
                     hue_order=hue_order, palette=palette, height=height, corner=True)
    g.map_offdiag(sns.scatterplot, s=8, alpha=0.7, linewidth=0)
    g.add_legend()
    sns.move_legend(g, "upper right", bbox_to_anchor=(0.98, 0.98), ncol=legend_ncol,
                    title=None, frameon=False, fontsize=8, markerscale=1.5)
    plt.tight_layout()
    if save_path:
        g.savefig(save_path, dpi=300, bbox_inches="tight")
    return g

def plot_explained_variance(evr, FIG_DIR,n_components=100, threshold=0.90, save_prefix=None, figsize=(4.8, 3.2)):
    """Cumulative (left axis) and per-component (right axis) explained variance; marks `threshold`.
    Returns fig, ax, k_thresh (components needed to reach threshold, or None)."""
    evr = np.asarray(evr, dtype=float)[:n_components]
    cum, k = np.cumsum(evr), np.arange(1, len(evr) + 1)

    fig, ax = plt.subplots(figsize=figsize)
    ax2 = ax.twinx()
    ax.plot(k, cum, color="C0", lw=1.8, label="cumulative")
    ax2.bar(k, evr, color="C1", width=0.9, label="per component", zorder=0)
    
    k_thresh = int(np.searchsorted(cum, threshold) + 1) if cum[-1] >= threshold else None
    ax.axhline(threshold, ls="--", lw=1, color="k")
    if k_thresh:
        ax.axvline(k_thresh, ls=":", lw=1, color="k")
        ax.annotate(f"{k_thresh} PCs → {threshold:.0%}", (k_thresh, threshold),
                    xytext=(6, -14), textcoords="offset points", fontsize=8)

    ax.set(xlabel="Principal component", ylabel="Cumulative explained variance", ylim=(0, 1.02))
    ax2.set(ylabel="Per-component explained variance", ylim=(0, evr.max() * 1.1))
    for a, c in [(ax, "C0"), (ax2, "C1")]:
        a.yaxis.set_major_formatter(PercentFormatter(1.0)); a.yaxis.label.set_color(c)
        a.spines["top"].set_visible(False)
    ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)

    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="center right")
    fig.tight_layout()
    
    if save_prefix:
        plt.savefig(FIG_DIR / f"{save_prefix}.png", dpi=400, bbox_inches="tight")

    return fig, ax, k_thresh
