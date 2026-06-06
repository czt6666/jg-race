"""Dataset2 Systematic EDA for Dynamic Graph Link Prediction"""
import pandas as pd
import numpy as np
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = "data/dataset2"

print("=" * 70)
print("LOADING DATA")
print("=" * 70)
train = pd.read_csv(f"{DATA_DIR}/train.csv")
test  = pd.read_csv(f"{DATA_DIR}/test.csv")
print(f"Train shape: {train.shape}")
print(f"Test  shape: {test.shape}")
print(f"\nTrain columns: {list(train.columns)}")
print(f"Test  columns[:5]: {list(test.columns[:5])} ... c100")

# ─────────────────────────────────────────────
# 0. Basic stats
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("0. BASIC STATISTICS")
print("=" * 70)
print(f"Train interactions: {len(train):,}")
print(f"Test  queries:      {len(test):,}")
print(f"Train split distribution:\n{train['split'].value_counts().sort_index()}")

# ─────────────────────────────────────────────
# 1. Bipartite vs Homogeneous Graph
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("1. GRAPH TYPE: BIPARTITE vs HOMOGENEOUS")
print("=" * 70)
src_nodes  = set(train['src'].unique())
dst_nodes  = set(train['dst'].unique())
overlap    = src_nodes & dst_nodes
print(f"Unique src nodes (train): {len(src_nodes):,}")
print(f"Unique dst nodes (train): {len(dst_nodes):,}")
print(f"Overlap (both src & dst): {len(overlap):,}")
print(f"Overlap ratio over src:   {len(overlap)/len(src_nodes):.4f}")
print(f"Overlap ratio over dst:   {len(overlap)/len(dst_nodes):.4f}")

# Test src nodes
test_src = set(test['src'].unique())
print(f"\nUnique src nodes (test):  {len(test_src):,}")
print(f"Test src ∩ train src:     {len(test_src & src_nodes):,}")
print(f"Test src ∩ train dst:     {len(test_src & dst_nodes):,}")

all_train_nodes = src_nodes | dst_nodes
print(f"\nAll unique node IDs (train): {len(all_train_nodes):,}")
print(f"Min node id: {min(all_train_nodes)}, Max node id: {max(all_train_nodes)}")

# Check candidate nodes in test
cand_cols = [f'c{i}' for i in range(1, 101)]
all_candidates = test[cand_cols].values.ravel()
unique_cands = set(all_candidates)
print(f"\nUnique candidate dst nodes (test): {len(unique_cands):,}")
print(f"Candidates ∈ train dst:            {len(unique_cands & dst_nodes):,} ({len(unique_cands & dst_nodes)/len(unique_cands)*100:.1f}%)")
print(f"Candidates NOT in train dst:       {len(unique_cands - dst_nodes):,} ({len(unique_cands - dst_nodes)/len(unique_cands)*100:.1f}%)")

# ─────────────────────────────────────────────
# 2. Cold Start Analysis
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("2. COLD START ANALYSIS")
print("=" * 70)

# Source cold start
cold_src = test_src - src_nodes
print(f"Test src NOT in train src: {len(cold_src):,} / {len(test_src):,} ({len(cold_src)/len(test_src)*100:.1f}%)")
cold_src_rows = test[test['src'].isin(cold_src)]
print(f"  → rows affected: {len(cold_src_rows):,} ({len(cold_src_rows)/len(test)*100:.1f}% of test queries)")

# Among cold src, what % are in train at all (as dst)?
cold_src_as_dst = cold_src & dst_nodes
if len(cold_src) > 0:
    print(f"  → cold src appearing as dst in train: {len(cold_src_as_dst):,} ({len(cold_src_as_dst)/len(cold_src)*100:.1f}%)")
else:
    print("  → No cold src nodes (all test src seen in training)")

# Split-based cold start (val vs test split)
if 'split' in train.columns:
    train_splits = sorted(train['split'].unique())
    print(f"\nTrain splits: {train_splits}")
    for sp in train_splits:
        sub = train[train['split'] == sp]
        print(f"  split={sp}: {len(sub):,} rows, src={sub['src'].nunique():,}, dst={sub['dst'].nunique():,}")

# ─────────────────────────────────────────────
# 3. Degree Distribution & Sparsity
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("3. DEGREE DISTRIBUTION & SPARSITY")
print("=" * 70)

src_counts = train['src'].value_counts()
dst_counts = train['dst'].value_counts()

print(f"\n--- Source node degree (# interactions per src) ---")
print(f"  mean={src_counts.mean():.2f}, median={src_counts.median():.0f}, "
      f"max={src_counts.max()}, min={src_counts.min()}")
print(f"  degree=1: {(src_counts==1).sum():,} ({(src_counts==1).mean()*100:.1f}%)")
print(f"  degree<=5: {(src_counts<=5).sum():,} ({(src_counts<=5).mean()*100:.1f}%)")
print(f"  degree>100: {(src_counts>100).sum():,} ({(src_counts>100).mean()*100:.1f}%)")
print(f"  top-1% src cover: {src_counts.nlargest(int(len(src_counts)*0.01)).sum()/len(train)*100:.1f}% of interactions")

print(f"\n--- Destination node degree (# interactions per dst) ---")
print(f"  mean={dst_counts.mean():.2f}, median={dst_counts.median():.0f}, "
      f"max={dst_counts.max()}, min={dst_counts.min()}")
print(f"  degree=1: {(dst_counts==1).sum():,} ({(dst_counts==1).mean()*100:.1f}%)")
print(f"  degree<=5: {(dst_counts<=5).sum():,} ({(dst_counts<=5).mean()*100:.1f}%)")
print(f"  degree>100: {(dst_counts>100).sum():,} ({(dst_counts>100).mean()*100:.1f}%)")
print(f"  top-1% dst cover: {dst_counts.nlargest(int(len(dst_counts)*0.01)).sum()/len(train)*100:.1f}% of interactions")

# Repeat edges
total_edges = len(train)
unique_edges = train.drop_duplicates(['src','dst']).shape[0]
print(f"\n--- Edge repetition ---")
print(f"  Total edges:  {total_edges:,}")
print(f"  Unique (src,dst) pairs: {unique_edges:,}")
print(f"  Repeat edge ratio: {(total_edges-unique_edges)/total_edges*100:.1f}%")

# ─────────────────────────────────────────────
# 4. Temporal Analysis
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("4. TEMPORAL ANALYSIS")
print("=" * 70)

train_ts = pd.to_datetime(train['time'], unit='s')
test_ts  = pd.to_datetime(test['time'], unit='s')

print(f"Train time range: {train_ts.min()} → {train_ts.max()}")
print(f"Test  time range: {test_ts.min()} → {test_ts.max()}")
print(f"Time gap (train_max → test_min): {(test_ts.min() - train_ts.max()).days} days")

# Timestamp granularity
train_ts_raw = train['time'].values
diffs = np.diff(np.sort(np.unique(train_ts_raw)))
diffs = diffs[diffs > 0]
print(f"\nTimestamp granularity (min non-zero diff): {diffs.min()} sec ({diffs.min()/3600:.2f} hr)")
print(f"Most common diff: {pd.Series(diffs).value_counts().index[0]} sec")

# Interaction density over time (monthly)
train['month'] = train_ts.dt.to_period('M')
monthly = train.groupby('month').size()
print(f"\nMonthly interaction counts (train):")
print(monthly.to_string())

# Time gap: last train interaction per src → test query time
train_last = train.groupby('src')['time'].max().rename('last_train_time')
test_w_last = test.merge(train_last, on='src', how='left')
test_w_last['gap_days'] = (test_w_last['time'] - test_w_last['last_train_time']) / 86400
print(f"\nTime gap (last train interaction → test query) per src:")
print(f"  mean={test_w_last['gap_days'].mean():.1f}d, "
      f"median={test_w_last['gap_days'].median():.1f}d, "
      f"max={test_w_last['gap_days'].max():.1f}d, "
      f"NaN (no history)={test_w_last['gap_days'].isna().sum():,}")

# ─────────────────────────────────────────────
# 5. Multi-neighbor & Sequence Behavior
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("5. MULTI-NEIGHBOR & SEQUENCE BEHAVIOR")
print("=" * 70)

# Multiple interactions at same timestamp for same src
same_ts = train.groupby(['src','time'])['dst'].nunique()
multi_ts = same_ts[same_ts > 1]
print(f"(src, time) groups with >1 unique dst: {len(multi_ts):,} "
      f"({len(multi_ts)/same_ts.shape[0]*100:.1f}% of unique (src,time) pairs)")
print(f"Max unique dst per (src,time): {same_ts.max()}")
print(f"Distribution of multi-dst at same time:")
print(same_ts.value_counts().sort_index().head(8).to_string())

# Sequence length distribution
seq_len = train.groupby('src').size()
print(f"\nSequence length per user:")
print(f"  mean={seq_len.mean():.1f}, median={seq_len.median():.0f}, "
      f"max={seq_len.max()}, 90th%={seq_len.quantile(0.9):.0f}, "
      f"95th%={seq_len.quantile(0.95):.0f}")

# Test: how many test queries per src?
test_per_src = test.groupby('src').size()
print(f"\nTest queries per src:")
print(f"  mean={test_per_src.mean():.1f}, median={test_per_src.median():.0f}, "
      f"max={test_per_src.max()}, min={test_per_src.min()}")
print(f"  src with >1 test query: {(test_per_src>1).sum():,} ({(test_per_src>1).mean()*100:.1f}%)")

# ─────────────────────────────────────────────
# 6. Candidate Set Distribution Analysis
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("6. CANDIDATE SET DISTRIBUTION ANALYSIS")
print("=" * 70)

# How often do candidates appear in train dst?
cand_in_train = np.isin(test[cand_cols].values, list(dst_nodes))
print(f"Fraction of candidates seen in train dst: {cand_in_train.mean():.4f}")
print(f"Queries where ALL 100 cands in train: {cand_in_train.all(axis=1).mean()*100:.1f}%")
print(f"Queries where NO  cand in train dst:  {(~cand_in_train).all(axis=1).mean()*100:.1f}%")

# For each test query, how many candidates did src interact with historically?
train_src_dst = train.groupby('src')['dst'].apply(set).to_dict()
def count_hist_cands(row):
    hist = train_src_dst.get(row['src'], set())
    cands = set(row[cand_cols].values)
    return len(hist & cands)

print("\nComputing per-query historical candidate overlap (sampling 5000)...")
sample_idx = np.random.choice(len(test), size=min(5000, len(test)), replace=False)
test_sample = test.iloc[sample_idx]
overlap_counts = test_sample.apply(count_hist_cands, axis=1)
print(f"  Candidates that src visited before (per query):")
print(f"  mean={overlap_counts.mean():.2f}, median={overlap_counts.median():.0f}, "
      f"max={overlap_counts.max()}, queries with >=1={( overlap_counts>=1).mean()*100:.1f}%")

# Global popularity of candidates
dst_pop = train['dst'].value_counts()
# For each candidate, get its global popularity rank
print(f"\nGlobal popularity of candidate nodes:")
all_cand_flat = test[cand_cols].values.ravel()
cand_pop = np.array([dst_pop.get(c, 0) for c in all_cand_flat])
print(f"  mean popularity: {cand_pop.mean():.1f}")
print(f"  % never seen in train: {(cand_pop==0).mean()*100:.1f}%")
print(f"  % seen >=10 times: {(cand_pop>=10).mean()*100:.1f}%")
print(f"  % seen >=100 times: {(cand_pop>=100).mean()*100:.1f}%")

# Column c1 analysis: is c1 the ground truth?
print(f"\n--- c1 position analysis (is c1 always the ground truth?) ---")
# If c1 is ground truth, it should be visited by src historically more than others
# Let's check: does c1 appear more in src history than c2-c100?
def c1_in_hist(row):
    hist = train_src_dst.get(row['src'], set())
    c1_hit = int(row['c1'] in hist)
    others = [int(row[f'c{i}'] in hist) for i in range(2,101)]
    return c1_hit, np.mean(others)

print("  Sampling 2000 rows to check c1 vs other candidates...")
sample2 = test.sample(min(2000, len(test)), random_state=42)
results = sample2.apply(c1_in_hist, axis=1)
c1_hits = [r[0] for r in results]
other_avg = [r[1] for r in results]
print(f"  c1 in src history: {np.mean(c1_hits)*100:.1f}%")
print(f"  avg other cand in src history: {np.mean(other_avg)*100:.2f}%")
print(f"  → Ratio: {np.mean(c1_hits)/(np.mean(other_avg)+1e-9):.1f}x")

# ─────────────────────────────────────────────
# 7. Val/Test Distribution Alignment
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("7. VAL/TEST DISTRIBUTION ALIGNMENT")
print("=" * 70)

if 'split' in train.columns:
    split_vals = sorted(train['split'].unique())
    train_only = train[train['split'] == 0] if 0 in split_vals else train
    val_data   = train[train['split'] == 1] if 1 in split_vals else pd.DataFrame()

    print(f"Splits in train: {split_vals}")

    if len(val_data) > 0:
        val_src = set(val_data['src'].unique())
        val_dst = set(val_data['dst'].unique())
        tr0_src = set(train_only['src'].unique())
        tr0_dst = set(train_only['dst'].unique())

        print(f"\nValidation set: {len(val_data):,} interactions")
        print(f"  val src NOT in split-0 src: {len(val_src-tr0_src):,} ({len(val_src-tr0_src)/len(val_src)*100:.1f}%)")
        print(f"  val dst NOT in split-0 dst: {len(val_dst-tr0_dst):,} ({len(val_dst-tr0_dst)/len(val_dst)*100:.1f}%)")

        val_ts = pd.to_datetime(val_data['time'], unit='s')
        tr0_ts = pd.to_datetime(train_only['time'], unit='s')
        print(f"  split-0 time: {tr0_ts.min()} → {tr0_ts.max()}")
        print(f"  split-1 time: {val_ts.min()} → {val_ts.max()}")
        print(f"  ⚠ Temporal overlap: {(val_ts.min() < tr0_ts.max())}")

    print(f"\n--- 100-way MRR setup validation ---")
    # Test set has 100 candidates, c1 assumed to be ground truth
    print(f"Test queries: {len(test):,}")
    print(f"Each row: src + time + 100 candidates")
    print(f"Assumed: c1 = ground truth positive")
    print(f"Task: rank c1 highest among c1..c100")
    print(f"Metric: MRR@100 (mean reciprocal rank of c1)")

# ─────────────────────────────────────────────
# 8. New Edge Analysis (test ground truth vs train)
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("8. NEW EDGE ANALYSIS (TRANSDUCTIVE vs INDUCTIVE)")
print("=" * 70)

train_edge_set = set(zip(train['src'], train['dst']))
# c1 = ground truth
c1_edges = list(zip(test['src'], test['c1']))
c1_in_train = [(s,d) in train_edge_set for s,d in c1_edges]
print(f"Ground truth (c1) edges that appeared in train: {sum(c1_in_train):,} / {len(c1_in_train):,} "
      f"({sum(c1_in_train)/len(c1_in_train)*100:.1f}%)")
print(f"Purely new (inductive) test edges: {len(c1_in_train)-sum(c1_in_train):,} "
      f"({(len(c1_in_train)-sum(c1_in_train))/len(c1_in_train)*100:.1f}%)")

# Repeat purchase rate
print(f"\nRepeat-purchase queries (src saw c1 before): {sum(c1_in_train):,} ({sum(c1_in_train)/len(c1_in_train)*100:.1f}%)")
if sum(c1_in_train) > 0:
    repeat_idx = [i for i,v in enumerate(c1_in_train) if v]
    repeat_test = test.iloc[repeat_idx]
    repeat_src_deg = repeat_test['src'].map(src_counts)
    print(f"  src degree in repeat queries: mean={repeat_src_deg.mean():.1f}, median={repeat_src_deg.median():.0f}")

# ─────────────────────────────────────────────
# 9. Quick Frequency Baseline Sanity
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("9. FREQUENCY BASELINE SANITY (UPPER BOUND ESTIMATE)")
print("=" * 70)

# For each test query, rank c1 by how often dst appeared in train globally
def freq_rank_c1(row):
    cands = [row[f'c{i}'] for i in range(1,101)]
    pops  = [dst_pop.get(c, 0) for c in cands]
    # rank of c1 (1=best)
    c1_pop = pops[0]
    rank = 1 + sum(1 for p in pops[1:] if p > c1_pop)
    return rank

print("Computing frequency baseline MRR on 5000 test samples...")
sample3 = test.sample(min(5000, len(test)), random_state=0)
ranks = sample3.apply(freq_rank_c1, axis=1)
mrr = (1/ranks).mean()
print(f"  Freq baseline MRR@100 (sample): {mrr:.4f}")
print(f"  % c1 ranked #1 by global freq:  {(ranks==1).mean()*100:.1f}%")

# Src-history recency baseline
def recency_rank_c1(row):
    src = row['src']
    cands = [row[f'c{i}'] for i in range(1,101)]
    hist = train[train['src']==src].tail(20)['dst'].values if src in train['src'].values else []
    hist_set = set(hist)
    in_hist = [int(c in hist_set) for c in cands]
    if sum(in_hist) == 0:
        return freq_rank_c1(row)
    # rank by (in_hist, freq)
    scores = [(in_hist[i], dst_pop.get(cands[i],0)) for i in range(100)]
    c1_score = scores[0]
    rank = 1 + sum(1 for s in scores[1:] if s > c1_score)
    return rank

print("\nComputing src-history recency baseline on 1000 samples...")
sample4 = test.sample(min(1000, len(test)), random_state=1)
ranks2 = sample4.apply(recency_rank_c1, axis=1)
mrr2 = (1/ranks2).mean()
print(f"  Recency+Freq baseline MRR@100 (sample): {mrr2:.4f}")
print(f"  % c1 ranked #1 by recency+freq:         {(ranks2==1).mean()*100:.1f}%")

print("\n" + "=" * 70)
print("EDA COMPLETE")
print("=" * 70)
