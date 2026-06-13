"""
快速分析已完成方法的相关性，并生成简单融合预测
用法: python ds2_ensemble_analysis.py
"""
import numpy as np
import os
from scipy.stats import spearmanr

BASE = "/root/workspace/python/jg-race"

METHODS = {
    "LightGCN": "102_LightGCN_bipartite_0613/dataset2/dataset2_result_lgcn.csv",
    "GRU4Rec": "104_GRU4Rec_pop_0613/dataset2/dataset2_result_gru4rec.csv",
    "BERT4Rec": "103_BERT4Rec_base_0613/dataset2/dataset2_result_bert4rec.csv",
    "IDhash": "108_SASRec_IDhash_0613/dataset2/dataset2_result_idhash.csv",
    "ItemCF": "101_ItemCF_timedecay_0613/dataset2/dataset2_result_itemcf.csv",
    "ColdNeighbor": "109_SASRec_coldneighbor_0613/dataset2/dataset2_result_coldwarm.csv",
    "SASRec100way": "111_SASRec_100way_testpool_0613/dataset2/dataset2_result_100way.csv",
}

def load_scores(path, n=1000):
    """Load first n rows of score CSV"""
    rows = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            rows.append([float(x) for x in line.strip().split(',')])
    return np.array(rows, dtype=np.float32)


scores = {}
for name, rel_path in METHODS.items():
    path = os.path.join(BASE, rel_path)
    if os.path.exists(path):
        scores[name] = load_scores(path, n=1000)
        print(f"Loaded {name}: {scores[name].shape}")
    else:
        print(f"MISSING: {name}")

available = list(scores.keys())
print(f"\n=== Available methods: {available} ===\n")

# Compute rank correlation between methods (using rank of scores per row)
def get_ranks(s):
    """Convert scores to ranks [B, K], higher score = lower rank (1=best)"""
    return np.argsort(np.argsort(-s, axis=1), axis=1) + 1  # [B, K] ranks

print("=== Spearman rank correlation between methods (mean over rows) ===")
names = list(scores.keys())
n_methods = len(names)
corr_matrix = np.zeros((n_methods, n_methods))
for i in range(n_methods):
    for j in range(n_methods):
        if i == j:
            corr_matrix[i, j] = 1.0
        elif j > i:
            r_i = get_ranks(scores[names[i]])  # [B, K]
            r_j = get_ranks(scores[names[j]])  # [B, K]
            # Flatten and compute spearman
            corr = spearmanr(r_i.ravel(), r_j.ravel())[0]
            corr_matrix[i, j] = corr_matrix[j, i] = corr

# Print correlation matrix
header = " " * 14 + "  ".join([f"{n[:8]:>8}" for n in names])
print(header)
for i, name in enumerate(names):
    row = f"{name[:14]:14}" + "  ".join([f"{corr_matrix[i,j]:8.3f}" for j in range(n_methods)])
    print(row)

# Generate simple ensemble predictions
print("\n=== Simple ensemble strategies ===")

def minmax_normalize(s):
    """Normalize per row to [0, 1]"""
    mn = s.min(axis=1, keepdims=True)
    mx = s.max(axis=1, keepdims=True)
    return (s - mn) / (mx - mn + 1e-8)

def rank_normalize(s):
    """Convert to normalized rank scores [0,1] where 1=best"""
    B, K = s.shape
    ranks = np.argsort(np.argsort(-s, axis=1), axis=1)  # 0=best
    return 1.0 - ranks / (K - 1)

# Simple equal-weight ensemble of SASRec100way + LightGCN + ColdNeighbor
ensemble_key = [k for k in ["SASRec100way", "ColdNeighbor", "LightGCN"] if k in scores]
if len(ensemble_key) >= 2:
    ens_scores = np.mean([rank_normalize(scores[k]) for k in ensemble_key], axis=0)
    print(f"\nEnsemble ({'+'.join(ensemble_key)}): shape={ens_scores.shape}")

    # Save ensemble
    if "SASRec100way" in scores:
        ens_out = os.path.join(BASE, "111_SASRec_100way_testpool_0613/dataset2/dataset2_result_ensemble.csv")
        full_scores = {}
        for name, rel_path in METHODS.items():
            path = os.path.join(BASE, rel_path)
            if os.path.exists(path) and name in ensemble_key:
                full_scores[name] = load_scores(path, n=200000)

        full_ens = np.mean([rank_normalize(full_scores[k]) for k in ensemble_key if k in full_scores], axis=0)
        print(f"Full ensemble shape: {full_ens.shape}")
        with open(ens_out, "w") as f:
            for row in full_ens:
                f.write(",".join([f"{x:.8f}" for x in row]) + "\n")
        print(f"Saved ensemble to: {ens_out}")

print("\nDONE")
