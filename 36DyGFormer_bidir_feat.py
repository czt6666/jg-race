import os
import os.path as osp
import sys
from datetime import datetime
from collections import Counter
from multiprocessing import Pool, cpu_count

root = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, root)
os.environ["JT_SYNC"] = "1"

import jittor as jt
from jittor import nn
import types


def _pad_sequence(sequences, batch_first=True, padding_value=0.0):
    if not sequences:
        return jt.array([])
    max_len = max(int(s.shape[0]) for s in sequences)
    padded = []
    for s in sequences:
        if int(s.shape[0]) < max_len:
            pad_len = max_len - int(s.shape[0])
            pad_shape = [pad_len] + list(s.shape[1:])
            pad = jt.full(pad_shape, padding_value, dtype=s.dtype)
            s = jt.concat([s, pad], dim=0)
        padded.append(s)
    if batch_first:
        return jt.stack(padded, dim=0)
    else:
        return jt.stack(padded, dim=1)


if not hasattr(jt.nn, "utils"):
    jt.nn.utils = types.ModuleType("utils")
if not hasattr(jt.nn.utils, "rnn"):
    jt.nn.utils.rnn = types.ModuleType("rnn")
jt.nn.utils.rnn.pad_sequence = _pad_sequence

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import average_precision_score, roc_auc_score
from jittor_geometric.data import TemporalData
from jittor_geometric.nn.models.dygformer import DyGFormer
from jittor_geometric.nn.dense.merge_predictor import MergeLayer
from jittor_geometric.dataloader.temporal_dataloader import TemporalDataLoader, get_neighbor_sampler
from jittor_geometric.evaluate.evaluators import MRR_Evaluator
import argparse

jt.flags.use_cuda = 1

IS_MPI = jt.in_mpi
RANK = jt.rank if IS_MPI else 0
WORLD_SIZE = jt.world_size if IS_MPI else 1
IS_MAIN = RANK == 0


def log(msg):
    if IS_MAIN:
        print(msg)


def sync_model_params(model):
    if not IS_MPI:
        return
    for p in model.parameters():
        p.update(p.mpi_all_reduce("mean"))


def mpi_barrier():
    if IS_MPI:
        jt.array([0], dtype="int32").mpi_all_reduce("add")


# =============================================================================
# 0. Bidirectional Node Feature Construction (multi-worker CPU)
# =============================================================================
# These module-level globals are set before Pool creation so that fork-based
# workers inherit them via copy-on-write — no serialization overhead.

_W_SRC   = None   # dict: node_id -> {'dst': np.int32[], 't': np.int64[]}
_W_DST   = None   # dict: node_id -> {'src': np.int32[], 't': np.int64[]}
_W_GS    = None   # global stats dict
_W_T_MIN = None   # int
_W_T_MAX = None   # int


def _compute_out_96(node_id, edges, gs, t_min, t_max):
    """96-dim outgoing features: statistics over edges where node is the source."""
    f = np.zeros(96, dtype=np.float32)
    if edges is None:
        return f
    dst = edges['dst']
    t   = edges['t']
    N   = len(t)
    if N == 0:
        return f

    t_f    = t.astype(np.float64)
    t_rng  = float(t_max - t_min) + 1.0

    # [0:8] Basic degree / count stats
    unique_dsts, cnts_arr = np.unique(dst, return_counts=True)
    n_uniq = len(unique_dsts)
    cnt_map = dict(zip(unique_dsts.tolist(), cnts_arr.tolist()))
    freqs   = cnts_arr.astype(np.float32)
    p       = freqs / freqs.sum()
    entropy = float(-np.sum(p * np.log2(p + 1e-10)))

    f[0] = np.log1p(N)
    f[1] = np.log1p(n_uniq)
    f[2] = n_uniq / (N + 1e-6)
    f[3] = float(np.log1p(freqs.max()))
    f[4] = float(np.log1p(freqs.mean()))
    f[5] = min(entropy / 10.0, 1.0)
    f[6] = np.log1p(N - n_uniq)
    f[7] = (N - n_uniq) / (N + 1e-6)

    # [8:24] Temporal stats (16)
    t_norm = (t_f - t_min) / t_rng
    f[8]  = float(t_norm.min())
    f[9]  = float(t_norm.max())
    f[10] = float(np.log1p(t_f.max() - t_f.min()))
    f[11] = float(t_norm.mean())
    f[12] = float(t_norm.std())
    f[13] = float(np.percentile(t_norm, 25))
    f[14] = float(np.percentile(t_norm, 50))
    f[15] = float(np.percentile(t_norm, 75))

    t_sorted = np.sort(t_f)
    if N > 1:
        iet   = np.diff(t_sorted)
        iet_m = float(iet.mean()) + 1e-6
        iet_s = float(iet.std())  + 1e-6
        f[16] = float(np.log1p(iet_m))
        f[17] = float(np.log1p(iet_s))
        f[18] = float(np.log1p(float(iet.min())))
        f[19] = float(np.log1p(float(iet.max())))
        f[20] = (iet_s - iet_m) / (iet_s + iet_m)   # burstiness in [-1, 1]
        f[21] = float(np.log1p(float(np.median(iet))))

    t_ref = float(t_f.max())
    f[22] = float(np.log1p(float(np.sum(t_f >= t_ref - 86400))))
    f[23] = float(np.log1p(float(np.sum(t_f >= t_ref - 7 * 86400))))

    # [24:32] 8-bin time histogram
    bins = np.linspace(t_min, t_max + 1.0, 9)
    hist, _ = np.histogram(t_f, bins=bins)
    f[24:32] = np.log1p(hist.astype(np.float32))

    # [32:48] Dst property stats (16)
    in_degs  = np.array([gs['in_deg'].get(int(d), 0) for d in unique_dsts], dtype=np.float32)
    out_degs = np.array([gs['out_deg'].get(int(d), 0) for d in unique_dsts], dtype=np.float32)

    if n_uniq > 0:
        f[32] = float(np.log1p(in_degs.mean()))
        f[33] = float(np.log1p(in_degs.std()))
        f[34] = float(np.log1p(in_degs.min()))
        f[35] = float(np.log1p(in_degs.max()))
        f[36] = float(np.log1p(float(np.median(in_degs))))
        f[37] = float(np.log1p(out_degs.mean()))
        f[38] = float(np.log1p(out_degs.std()))
        f[39] = float(np.log1p(out_degs.min()))
        f[40] = float(np.log1p(out_degs.max()))
        f[41] = float(np.log1p(float(np.median(out_degs))))

    fw_in  = sum(gs['in_deg'].get(int(d), 0) * cnt_map.get(int(d), 0) for d in unique_dsts) / (N + 1e-6)
    fw_out = sum(gs['out_deg'].get(int(d), 0) * cnt_map.get(int(d), 0) for d in unique_dsts) / (N + 1e-6)
    f[42] = float(np.log1p(fw_in))
    f[43] = float(np.log1p(fw_out))

    top100   = gs['top100_dsts']
    all_srcs = gs['all_srcs']
    f[44] = float(sum(1 for d in unique_dsts if int(d) in top100))   / (n_uniq + 1e-6)
    f[45] = float(sum(1 for d in unique_dsts if int(d) in all_srcs)) / (n_uniq + 1e-6)

    p75_in = gs['in_deg_p75']
    f[46] = float(sum(1 for d in unique_dsts if gs['in_deg'].get(int(d), 0) >= p75_in)) / (n_uniq + 1e-6)
    f[47] = float(np.log1p(len(gs['two_hop'].get(node_id, ()))))

    # [48:64] Horizon activity counts (16)
    horizons_out = [
        3600,        6*3600,      86400,        3*86400,
        7*86400,     14*86400,    30*86400,     90*86400,
        180*86400,   365*86400,   2*365*86400,  3*365*86400,
        5*365*86400, 7*365*86400, 10*365*86400, 15*365*86400,
    ]
    for j, h in enumerate(horizons_out):
        f[48 + j] = float(np.log1p(float(np.sum(t_f >= t_ref - h))))

    # [64:72] Trend stats (8)
    t_span = float(t_f.max() - t_f.min()) + 1.0
    sp1 = float(t_f.min()) + t_span / 3.0
    sp2 = float(t_f.max()) - t_span / 3.0
    n_first = int(np.sum(t_f <= sp1))
    n_last  = int(np.sum(t_f >= sp2))
    n_mid   = N - n_first - n_last

    f[64] = float(np.log1p(n_first))
    f[65] = float(np.log1p(max(0, n_mid)))
    f[66] = float(np.log1p(n_last))
    f[67] = float(n_last) / (float(n_first) + 1e-6)
    f[68] = float(np.log1p(N / (t_span / 86400.0 + 1e-6)))
    f[69] = float(n_last)  / (N + 1e-6)
    f[70] = float(n_first) / (N + 1e-6)
    f[71] = float(N > 1)

    # [72:88] IET histogram (16)
    if N > 1:
        iet_vals = np.diff(np.sort(t_f))
        iet_log  = np.log1p(iet_vals)
        max_iet  = float(iet_log.max()) + 1e-6
        iet_bins = np.linspace(0.0, max_iet, 17)
        iet_hist, _ = np.histogram(iet_log, bins=iet_bins)
        f[72:88] = np.log1p(iet_hist.astype(np.float32))

    # [88:96] Global / topology stats (8)
    out_self = gs['out_deg'].get(node_id, 0)
    in_self  = gs['in_deg'].get(node_id, 0)
    f[88] = float(np.log1p(out_self))
    f[89] = float(out_self) / (gs['max_out'] + 1e-6)
    f[90] = float(np.log1p(in_self))
    f[91] = float(in_self)  / (gs['max_in']  + 1e-6)
    f[92] = float(in_self)  / (out_self + 1e-6)
    f[93] = float(node_id in gs['all_srcs'])
    f[94] = float(node_id in gs['all_dsts'])
    f[95] = float(np.log1p(len(gs['two_hop'].get(node_id, ())) / (n_uniq + 1e-6)))

    return f


def _compute_in_32(node_id, edges, gs, t_min, t_max):
    """32-dim incoming features: statistics over edges where node is the destination."""
    f = np.zeros(32, dtype=np.float32)
    if edges is None:
        return f
    src = edges['src']
    t   = edges['t']
    N   = len(t)
    if N == 0:
        return f

    t_f   = t.astype(np.float64)
    t_rng = float(t_max - t_min) + 1.0

    # [0:8] Basic in-degree stats
    unique_srcs, cnts_arr = np.unique(src, return_counts=True)
    n_uniq = len(unique_srcs)
    freqs  = cnts_arr.astype(np.float32)
    p      = freqs / freqs.sum()
    entropy = float(-np.sum(p * np.log2(p + 1e-10)))

    f[0] = float(np.log1p(N))
    f[1] = float(np.log1p(n_uniq))
    f[2] = n_uniq / (N + 1e-6)
    f[3] = float(np.log1p(float(freqs.max())))
    f[4] = float(N) / (gs['max_in'] + 1e-6)
    f[5] = min(entropy / 10.0, 1.0)
    f[6] = float(np.log1p(N - n_uniq))
    f[7] = (N - n_uniq) / (N + 1e-6)

    # [8:16] Temporal stats
    t_norm = (t_f - t_min) / t_rng
    t_ref  = float(t_f.max())
    f[8]  = float(t_norm.min())
    f[9]  = float(t_norm.max())
    f[10] = float(np.log1p(t_f.max() - t_f.min()))
    f[11] = float(t_norm.mean())
    f[12] = float(t_norm.std())
    f[13] = float(np.log1p(float(np.sum(t_f >= t_ref - 86400))))
    f[14] = float(np.log1p(float(np.sum(t_f >= t_ref - 7 * 86400))))
    f[15] = float(np.log1p(float(np.sum(t_f >= t_ref - 30 * 86400))))

    # [16:24] Src property stats
    src_out_degs = np.array([gs['out_deg'].get(int(s), 0) for s in unique_srcs], dtype=np.float32)
    src_in_degs  = np.array([gs['in_deg'].get(int(s),  0) for s in unique_srcs], dtype=np.float32)
    if n_uniq > 0:
        f[16] = float(np.log1p(src_out_degs.mean()))
        f[17] = float(np.log1p(src_out_degs.std()))
        f[18] = float(np.log1p(float(src_out_degs.max())))
        f[19] = float(np.log1p(src_in_degs.mean()))
    f[20] = float(sum(1 for s in unique_srcs if int(s) in gs['all_dsts'])) / (n_uniq + 1e-6)
    f[21] = float(np.log1p(len(gs['two_hop_in'].get(node_id, ()))))
    f[22] = float(node_id in gs['all_srcs'])
    f[23] = float(node_id in gs['all_dsts'])

    # [24:32] Trend / horizon stats
    t_span = float(t_f.max() - t_f.min()) + 1.0
    sp1 = float(t_f.min()) + t_span / 3.0
    sp2 = float(t_f.max()) - t_span / 3.0
    n_first = int(np.sum(t_f <= sp1))
    n_last  = int(np.sum(t_f >= sp2))

    f[24] = float(np.log1p(max(0, n_first)))
    f[25] = float(np.log1p(max(0, n_last)))
    f[26] = float(n_last) / (float(n_first) + 1e-6)
    f[27] = float(n_last) / (N + 1e-6)

    horizons_in = [86400, 7 * 86400, 30 * 86400, 90 * 86400]
    for j, h in enumerate(horizons_in):
        f[28 + j] = float(np.log1p(float(np.sum(t_f >= t_ref - h))))

    return f


def _feature_worker(node_chunk):
    """Worker: compute 96-out + 32-in features for a list of node IDs."""
    out_d = {}
    in_d  = {}
    for nid in node_chunk:
        out_d[nid] = _compute_out_96(nid, _W_SRC.get(nid), _W_GS, _W_T_MIN, _W_T_MAX)
        in_d[nid]  = _compute_in_32(nid,  _W_DST.get(nid), _W_GS, _W_T_MIN, _W_T_MAX)
    return out_d, in_d


def build_bidir_node_features(df, node_size, n_workers=None):
    """
    Build 128-dim node features (96 outgoing + 32 incoming) in parallel.
    Uses fork-based multiprocessing.Pool so large lookup dicts are shared
    copy-on-write — no serialization per worker call.
    """
    global _W_SRC, _W_DST, _W_GS, _W_T_MIN, _W_T_MAX

    if n_workers is None:
        n_workers = min(cpu_count(), 16)

    src_arr = df['src'].values.astype(np.int32)
    dst_arr = df['dst'].values.astype(np.int32)
    t_arr   = df['time'].values.astype(np.int64)
    t_min   = int(t_arr.min())
    t_max   = int(t_arr.max())

    # Group edges by src (outgoing) and by dst (incoming)
    print(f"  Grouping {len(df)} edges by src/dst...", flush=True)
    src_by_node = {}
    dst_by_node = {}
    for s, d, t in zip(src_arr.tolist(), dst_arr.tolist(), t_arr.tolist()):
        if s not in src_by_node:
            src_by_node[s] = {'dst': [], 't': []}
        src_by_node[s]['dst'].append(d)
        src_by_node[s]['t'].append(t)
        if d not in dst_by_node:
            dst_by_node[d] = {'src': [], 't': []}
        dst_by_node[d]['src'].append(s)
        dst_by_node[d]['t'].append(t)

    for k in src_by_node:
        src_by_node[k]['dst'] = np.array(src_by_node[k]['dst'], dtype=np.int32)
        src_by_node[k]['t']   = np.array(src_by_node[k]['t'],   dtype=np.int64)
    for k in dst_by_node:
        dst_by_node[k]['src'] = np.array(dst_by_node[k]['src'], dtype=np.int32)
        dst_by_node[k]['t']   = np.array(dst_by_node[k]['t'],   dtype=np.int64)

    # Global statistics
    out_deg     = Counter(src_arr.tolist())
    in_deg      = Counter(dst_arr.tolist())
    top100_dsts = set(n for n, _ in Counter(dst_arr.tolist()).most_common(100))
    all_srcs    = set(src_arr.tolist())
    all_dsts    = set(dst_arr.tolist())
    max_out     = max(out_deg.values()) if out_deg else 1
    max_in      = max(in_deg.values())  if in_deg  else 1
    in_deg_p75  = float(np.percentile(list(in_deg.values()), 75)) if in_deg else 1.0

    # Two-hop outgoing (cap to MAX_EXPAND neighbors to avoid O(n²))
    MAX_EXPAND = 50
    print("  Computing two-hop tables...", flush=True)
    src_nbrs = {k: set(v['dst'].tolist()) for k, v in src_by_node.items()}
    two_hop = {}
    for s, nbrs in src_nbrs.items():
        hop2 = set()
        for n in list(nbrs)[:MAX_EXPAND]:
            hop2.update(src_nbrs.get(n, set()))
        hop2 -= src_nbrs.get(s, set())
        hop2.discard(s)
        two_hop[s] = hop2

    dst_nbrs = {k: set(v['src'].tolist()) for k, v in dst_by_node.items()}
    two_hop_in = {}
    for d, srcs in dst_nbrs.items():
        hop2 = set()
        for s in list(srcs)[:MAX_EXPAND]:
            hop2.update(dst_nbrs.get(s, set()))
        hop2 -= dst_nbrs.get(d, set())
        hop2.discard(d)
        two_hop_in[d] = hop2

    global_stats = {
        'out_deg':     dict(out_deg),
        'in_deg':      dict(in_deg),
        'top100_dsts': top100_dsts,
        'all_srcs':    all_srcs,
        'all_dsts':    all_dsts,
        'max_out':     max_out,
        'max_in':      max_in,
        'in_deg_p75':  in_deg_p75,
        'two_hop':     two_hop,
        'two_hop_in':  two_hop_in,
    }

    # Set module-level globals before forking — workers inherit via COW
    _W_SRC, _W_DST, _W_GS = src_by_node, dst_by_node, global_stats
    _W_T_MIN, _W_T_MAX     = t_min, t_max

    all_nodes  = list(range(node_size))
    chunk_size = max(1, (len(all_nodes) + n_workers * 8 - 1) // (n_workers * 8))
    chunks     = [all_nodes[i:i + chunk_size] for i in range(0, len(all_nodes), chunk_size)]

    print(f"  Launching {n_workers} workers for {node_size} nodes ({len(chunks)} chunks)...", flush=True)
    node_features = np.zeros((node_size, 128), dtype=np.float32)

    with Pool(n_workers) as pool:
        for out_d, in_d in tqdm(pool.imap(_feature_worker, chunks),
                                 total=len(chunks), desc="Bidir features", ncols=100):
            for nid, feat in out_d.items():
                node_features[nid, :96] = feat
            for nid, feat in in_d.items():
                node_features[nid, 96:] = feat

    nonzero = int(np.sum(np.any(node_features != 0, axis=1)))
    print(f"  Done: shape={node_features.shape}, nodes with features={nonzero}/{node_size}", flush=True)
    return node_features


# =============================================================================
# 1. Random Negative Sampler
# =============================================================================

class RandomNegativeSampler:
    def __init__(self, min_dst, max_dst, seed=42):
        self.min_dst = int(min_dst)
        self.max_dst = int(max_dst)
        self._rng    = np.random.RandomState(seed)

    def step_epoch(self):
        pass

    def sample(self, src, dst, K):
        return self._rng.randint(self.min_dst, self.max_dst + 1,
                                  size=(len(src) * K,)).astype(np.int32)


# =============================================================================
# 2. Ranking Loss (InfoNCE)
# =============================================================================

def compute_ranking_loss(pos_logit, neg_logit, temperature=1.0):
    B   = pos_logit.shape[0]
    K   = neg_logit.shape[0] // B
    pos = pos_logit.unsqueeze(1)
    neg = neg_logit.reshape(B, K)
    all_logits = jt.concat([pos, neg], dim=1) / temperature
    log_probs  = jt.nn.log_softmax(all_logits, dim=1)
    return -log_probs[:, 0].mean()


# =============================================================================
# 3. Validation — 100-way MRR
# =============================================================================

def test_val(model, loader, sampler, val_num_neg=99):
    model.eval()
    backbone, predictor = model[0], model[1]
    mrr_eval = MRR_Evaluator()

    ap_sum, auc_sum, n_batches = 0.0, 0.0, 0
    mrr_sum, mrr_count         = 0.0, 0
    loss_sum                   = 0.0

    loader_tqdm = tqdm(loader, ncols=120, desc="Validation", disable=not IS_MAIN)
    for _, batch_data in enumerate(loader_tqdm):
        src = np.array(batch_data.src).astype(np.int32)
        dst = np.array(batch_data.dst).astype(np.int32)
        t   = np.array(batch_data.t).astype(np.float32)

        src_emb_pos, dst_emb_pos = backbone.compute_src_dst_node_temporal_embeddings(
            src_node_ids=src, dst_node_ids=dst, node_interact_times=t
        )
        pos_logit = predictor(src_emb_pos, dst_emb_pos).squeeze(-1)
        pos_score = jt.sigmoid(pos_logit).numpy()

        neg_dst  = sampler.sample(src, dst, K=val_num_neg)
        src_rep  = np.repeat(src, val_num_neg)
        t_rep    = np.repeat(t,   val_num_neg)
        src_emb_neg, neg_dst_emb = backbone.compute_src_dst_node_temporal_embeddings(
            src_node_ids=src_rep, dst_node_ids=neg_dst, node_interact_times=t_rep
        )
        neg_logit = predictor(src_emb_neg, neg_dst_emb).squeeze(-1)
        neg_score = jt.sigmoid(neg_logit).numpy()

        with jt.no_grad():
            loss_sum += float(compute_ranking_loss(pos_logit, neg_logit).item())

        y_true  = np.concatenate([np.ones_like(pos_score),  np.zeros_like(neg_score)])
        y_score = np.concatenate([pos_score, neg_score])
        ap_sum  += float(average_precision_score(y_true, y_score))
        auc_sum += float(roc_auc_score(y_true, y_score))
        n_batches += 1

        batch_mrr  = mrr_eval.eval(pos_score, neg_score.reshape(len(src), val_num_neg))
        mrr_sum   += float(np.sum(batch_mrr))
        mrr_count += len(batch_mrr)

        del src_emb_pos, dst_emb_pos, pos_logit, pos_score
        del src_emb_neg, neg_dst_emb, neg_logit, neg_score
        jt.sync_all()

    if IS_MPI:
        agg = jt.array(
            [ap_sum, auc_sum, mrr_sum, float(n_batches), float(mrr_count), loss_sum],
            dtype="float32",
        ).mpi_all_reduce("add")
        ap_sum, auc_sum, mrr_sum, n_batches, mrr_count, loss_sum = agg.numpy().tolist()

    if n_batches == 0 or mrr_count == 0:
        return {"AP": 0.0, "AUC": 0.0, "MRR": 0.0, "Loss": 0.0}
    return {
        "AP":   ap_sum   / n_batches,
        "AUC":  auc_sum  / n_batches,
        "MRR":  mrr_sum  / mrr_count,
        "Loss": loss_sum / n_batches,
    }


# =============================================================================
# 4. Training — best model selected by MRR
# =============================================================================

def train(
    model, optimizer, sampler, train_loader, val_loader,
    num_epochs, save_path, dataset_name,
    early_stop_patience=10, num_neg=99, val_num_neg=99,
    ranking_temperature=1.0, select_metric="MRR",
):
    best_val        = 0.0
    patience_counter = 0

    for epoch in range(num_epochs):
        sampler.step_epoch()
        model.train()
        backbone, predictor = model[0], model[1]
        train_losses = []
        train_tqdm = tqdm(train_loader, ncols=120, desc=f"Epoch {epoch + 1}", disable=not IS_MAIN)

        for batch_data in train_tqdm:
            src = np.array(batch_data.src).astype(np.int32)
            dst = np.array(batch_data.dst).astype(np.int32)
            t   = np.array(batch_data.t).astype(np.float32)

            src_emb_pos, dst_emb_pos = backbone.compute_src_dst_node_temporal_embeddings(
                src_node_ids=src, dst_node_ids=dst, node_interact_times=t
            )
            pos_logit = predictor(src_emb_pos, dst_emb_pos)

            neg_dst  = sampler.sample(src, dst, K=num_neg)
            src_rep  = np.repeat(src, num_neg)
            t_rep    = np.repeat(t,   num_neg)
            src_emb_neg, neg_dst_emb = backbone.compute_src_dst_node_temporal_embeddings(
                src_node_ids=src_rep, dst_node_ids=neg_dst, node_interact_times=t_rep
            )
            neg_logit = predictor(src_emb_neg, neg_dst_emb)

            loss = compute_ranking_loss(pos_logit.squeeze(-1), neg_logit.squeeze(-1),
                                        temperature=ranking_temperature)
            optimizer.zero_grad()
            optimizer.step(loss)
            jt.sync_all()
            sync_model_params(model)
            jt.sync_all()

            train_losses.append(loss.item())
            train_tqdm.set_description(f"Epoch {epoch + 1}, loss: {loss.item():.4f}")

        if IS_MPI:
            agg_loss = jt.array(
                [float(np.sum(train_losses)), float(len(train_losses))], dtype="float32"
            ).mpi_all_reduce("add")
            avg_train_loss = agg_loss[0].item() / agg_loss[1].item() if agg_loss[1].item() > 0 else 0.0
        else:
            avg_train_loss = float(np.mean(train_losses)) if train_losses else 0.0

        log(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}")
        val_res = test_val(model, val_loader, sampler, val_num_neg=val_num_neg)
        log(f"Epoch {epoch + 1}, Val: {val_res}")

        current_val = val_res[select_metric]
        if current_val > best_val:
            best_val         = current_val
            patience_counter = 0
            if IS_MAIN:
                jt.save(model.state_dict(), f"{save_path}/{dataset_name}_DyGFormer_best.pkl")
                os.sync()
            mpi_barrier()
            log(f"  -> New best {select_metric}: {best_val:.6f}, model saved!")
        else:
            patience_counter += 1
            log(f"  -> No improvement for {patience_counter} epoch(s), best {select_metric}: {best_val:.6f}")

        if IS_MAIN:
            jt.save(model.state_dict(), f"{save_path}/{dataset_name}_DyGFormer.pkl")
            os.sync()
        mpi_barrier()

        if patience_counter >= early_stop_patience:
            log(f"\nEarly stopping after {epoch + 1} epochs. Best {select_metric}: {best_val:.6f}")
            break

    return best_val


# =============================================================================
# 5. Test Inference
# =============================================================================

def test_competition(model, test_src, test_time, test_candidates, tmp_dir, dataset_name, batch_size=50):
    model.eval()
    backbone, predictor = model[0], model[1]
    n        = len(test_src)
    num_cands = test_candidates.shape[1]

    my_idx = np.arange(RANK, n, WORLD_SIZE, dtype=np.int64) if IS_MPI else np.arange(n, dtype=np.int64)
    my_n   = len(my_idx)

    my_scores = np.zeros((my_n, num_cands), dtype=np.float32)
    pbar = tqdm(range((my_n + batch_size - 1) // batch_size),
                ncols=120, desc=f"Testing[r{RANK}]", position=RANK)

    with jt.no_grad():
        for batch_idx in pbar:
            start = batch_idx * batch_size
            end   = min((batch_idx + 1) * batch_size, my_n)
            idx_chunk = my_idx[start:end]
            b = len(idx_chunk)

            src_rep  = np.repeat(test_src[idx_chunk], num_cands).astype(np.int32)
            t_rep    = np.repeat(test_time[idx_chunk].astype(np.float32), num_cands)
            cand_flat = test_candidates[idx_chunk].reshape(-1).astype(np.int32)

            src_emb, dst_emb = backbone.compute_src_dst_node_temporal_embeddings(
                src_node_ids=src_rep, dst_node_ids=cand_flat, node_interact_times=t_rep
            )
            logit  = predictor(src_emb, dst_emb).squeeze(-1)
            probs  = jt.sigmoid(logit).numpy().reshape(b, num_cands).astype(np.float32)
            my_scores[start:end] = probs

    if not IS_MPI:
        full = np.zeros((n, num_cands), dtype=np.float32)
        full[my_idx] = my_scores
        return full

    tmp_idx    = f"{tmp_dir}/_tmp_{dataset_name}_rank{RANK}_idx.npy"
    tmp_scores = f"{tmp_dir}/_tmp_{dataset_name}_rank{RANK}_scores.npy"
    tmp_done   = f"{tmp_dir}/_tmp_{dataset_name}_rank{RANK}_done.npy"
    np.save(tmp_idx, my_idx)
    np.save(tmp_scores, my_scores)
    np.save(tmp_done, np.array([1], dtype=np.int8))

    def _wait(path, timeout=300):
        import time as _t
        deadline = _t.time() + timeout
        while _t.time() < deadline:
            if os.path.exists(path) and os.path.getsize(path) > 0:
                return True
            _t.sleep(1.0)
        return False

    for r in range(WORLD_SIZE):
        if not _wait(f"{tmp_dir}/_tmp_{dataset_name}_rank{r}_done.npy"):
            print(f"[r{RANK}] timeout waiting for rank {r}", flush=True)
            return None

    if not IS_MAIN:
        return None

    all_idx    = np.concatenate([np.load(f"{tmp_dir}/_tmp_{dataset_name}_rank{r}_idx.npy")
                                  for r in range(WORLD_SIZE)])
    all_scores = np.concatenate([np.load(f"{tmp_dir}/_tmp_{dataset_name}_rank{r}_scores.npy")
                                  for r in range(WORLD_SIZE)], axis=0)
    result = all_scores[np.argsort(all_idx)]
    for r in range(WORLD_SIZE):
        for suf in ["idx", "scores", "done"]:
            try:
                os.remove(f"{tmp_dir}/_tmp_{dataset_name}_rank{r}_{suf}.npy")
            except OSError:
                pass
    return result


# =============================================================================
# 6. Main
# =============================================================================

DATE_STR         = datetime.now().strftime("%m%d")
ALGO_NAME        = "DyGFormer_bidir_feat"
DEFAULT_SAVE_DIR = f"./saved_models/{ALGO_NAME}_{DATE_STR}"
DEFAULT_OUT_DIR  = f"./data/{ALGO_NAME}_{DATE_STR}"

parser = argparse.ArgumentParser()
parser.add_argument("--dataset",          type=str,   required=True)
parser.add_argument("--data_dir",         type=str,   default="./data")
parser.add_argument("--save_dir",         type=str,   default=DEFAULT_SAVE_DIR)
parser.add_argument("--output_dir",       type=str,   default=DEFAULT_OUT_DIR)
parser.add_argument("--epochs",           type=int,   default=10)
parser.add_argument("--batch_size",       type=int,   default=32)
parser.add_argument("--test_batch_size",  type=int,   default=50)
parser.add_argument("--early_stop",       type=int,   default=10)
parser.add_argument("--node_feat_dim",    type=int,   default=128)   # must equal 96+32
parser.add_argument("--edge_feat_dim",    type=int,   default=128)
parser.add_argument("--time_feat_dim",    type=int,   default=100)
parser.add_argument("--channel_dim",      type=int,   default=64)
parser.add_argument("--num_layers",       type=int,   default=2)
parser.add_argument("--num_heads",        type=int,   default=2)
parser.add_argument("--patch_size",       type=int,   default=1)
parser.add_argument("--max_seq_len",      type=int,   default=48)
parser.add_argument("--dropout",          type=float, default=0.2)
parser.add_argument("--lr",               type=float, default=1e-4)
parser.add_argument("--num_neg",          type=int,   default=99)
parser.add_argument("--val_num_neg",      type=int,   default=99)
parser.add_argument("--ranking_temp",     type=float, default=1.0)
parser.add_argument("--n_feat_workers",   type=int,   default=None,
                    help="CPU workers for feature construction (default: min(cpu_count,16))")
parser.add_argument("--eval_only",        action="store_true")
parser.add_argument("--model_path",       type=str,   default=None)
args = parser.parse_args()

save_path = osp.abspath(args.save_dir)
if IS_MAIN:
    os.makedirs(save_path,        exist_ok=True)
    os.makedirs(args.output_dir,  exist_ok=True)

effective_lr = args.lr * WORLD_SIZE if IS_MPI else args.lr

log("=" * 80)
log(f"DyGFormer — 100-way MRR, bidir node features (96 out + 32 in) — {DATE_STR}")
log(f"Dataset: {args.dataset} | batch: {args.batch_size} | lr: {effective_lr}")
log(f"num_neg(train): {args.num_neg} | val_num_neg: {args.val_num_neg} | select_by: MRR")
log("=" * 80)

df       = pd.read_csv(f"{args.data_dir}/{args.dataset}/train.csv")
test_df  = pd.read_csv(f"{args.data_dir}/{args.dataset}/test.csv")

src_np         = df["src"].values.astype(np.int32)
dst_np         = df["dst"].values.astype(np.int32)
t_np           = df["time"].values.astype(np.int32)
edge_ids_np    = np.arange(len(df), dtype=np.int32) + 1
num_edges      = len(df)

test_src        = test_df["src"].values.astype(np.int32)
test_time       = test_df["time"].values.astype(np.int32)
test_candidates = test_df.iloc[:, 2:].values.astype(np.int32)

log(f"Train+Val: {len(df)}, Test: {len(test_df)}")
is_bipartite = len(set(src_np.tolist()) & set(dst_np.tolist())) == 0
log(f"Bipartite: {is_bipartite}")

max_node  = max(int(src_np.max()), int(dst_np.max()), int(test_candidates.max()))
node_size = max_node + 1
log(f"Node size: {node_size}, Num edges: {num_edges}")

# Build 128-dim bidirectional node features (parallel CPU)
log("\nBuilding bidirectional node features (96 outgoing + 32 incoming)...")
node_raw_features = build_bidir_node_features(df, node_size, n_workers=args.n_feat_workers)
edge_raw_features = np.zeros((num_edges + 1, args.edge_feat_dim), dtype=np.float32)
log(f"Node features: {node_raw_features.shape}, edge features: {edge_raw_features.shape}\n")

sampler = RandomNegativeSampler(
    min_dst=int(dst_np.min()),
    max_dst=int(dst_np.max()),
    seed=42,
)

num_total = len(df)
num_val   = int(num_total * 0.15)
num_train = num_total - num_val

train_data = TemporalData(
    src=jt.Var(src_np[:num_train]), dst=jt.Var(dst_np[:num_train]),
    t=jt.Var(t_np[:num_train]),     edge_ids=jt.Var(edge_ids_np[:num_train]),
)
val_data = TemporalData(
    src=jt.Var(src_np[num_train:]), dst=jt.Var(dst_np[num_train:]),
    t=jt.Var(t_np[num_train:]),     edge_ids=jt.Var(edge_ids_np[num_train:]),
)
full_data = TemporalData(
    src=jt.Var(src_np), dst=jt.Var(dst_np),
    t=jt.Var(t_np),     edge_ids=jt.Var(edge_ids_np),
)

train_loader = TemporalDataLoader(train_data, batch_size=args.batch_size, neg_sampling_ratio=None)
val_loader   = TemporalDataLoader(val_data,   batch_size=args.batch_size, neg_sampling_ratio=None)

if IS_MPI:
    n_keep = (len(train_loader.arange) // WORLD_SIZE) * WORLD_SIZE
    train_loader.arange = train_loader.arange[:n_keep][RANK::WORLD_SIZE]
    val_loader.arange   = val_loader.arange[RANK::WORLD_SIZE]

full_neighbor_sampler = get_neighbor_sampler(full_data, "recent", seed=1)

backbone = DyGFormer(
    node_raw_features=node_raw_features,
    edge_raw_features=edge_raw_features,
    neighbor_sampler=full_neighbor_sampler,
    time_feat_dim=args.time_feat_dim,
    channel_embedding_dim=args.channel_dim,
    patch_size=args.patch_size,
    num_layers=args.num_layers,
    num_heads=args.num_heads,
    dropout=args.dropout,
    max_input_sequence_length=args.max_seq_len,
    bipartite=is_bipartite,
)
predictor = MergeLayer(
    input_dim1=args.node_feat_dim,
    input_dim2=args.node_feat_dim,
    hidden_dim=args.node_feat_dim,
    output_dim=1,
)
model = nn.Sequential(backbone, predictor)

if IS_MPI:
    model.mpi_param_broadcast(root=0)

optimizer = jt.nn.Adam(list(model.parameters()), lr=effective_lr)

if not args.eval_only:
    log(f"\nTraining — 100-way MRR, bidir node features, best selected by MRR...")
    train(
        model, optimizer, sampler, train_loader, val_loader,
        args.epochs, save_path, args.dataset, args.early_stop,
        num_neg=args.num_neg, val_num_neg=args.val_num_neg,
        ranking_temperature=args.ranking_temp, select_metric="MRR",
    )
    mpi_barrier()
else:
    log("\n--eval_only set, skipping training.")

log("\nGenerating predictions using best model...")
if args.model_path is not None:
    model.load_state_dict(jt.load(args.model_path))
    log(f"Loaded model from {args.model_path}")
else:
    best_path   = f"{save_path}/{args.dataset}_DyGFormer_best.pkl"
    latest_path = f"{save_path}/{args.dataset}_DyGFormer.pkl"
    if os.path.exists(best_path):
        model.load_state_dict(jt.load(best_path))
        log(f"Loaded best model from {best_path}")
    else:
        model.load_state_dict(jt.load(latest_path))
        log(f"Best not found, using latest model")

scores = test_competition(model, test_src, test_time, test_candidates,
                           save_path, args.dataset, args.test_batch_size)

if IS_MAIN:
    if scores is None:
        log("Aggregation skipped.")
        sys.exit(0)
    log(f"Scores shape: {scores.shape}, range: [{scores.min():.6f}, {scores.max():.6f}]")
    output_file = f"{args.output_dir}/{args.dataset}/{args.dataset}_result.csv"
    os.makedirs(osp.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f_out:
        for row in scores:
            f_out.write(",".join([f"{p:.8f}" for p in row]) + "\n")
    log(f"Result saved to {output_file}")

log("\n" + "=" * 80)
log("DONE")
log("=" * 80)
