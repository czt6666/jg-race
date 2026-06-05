"""
7DyGFormer - 直接优化 MRR 的版本
核心改动：BCE → InfoNCE (Sampled Softmax)
  - BCE: 正样本 vs 1负样本，独立二分类，不感知排名
  - InfoNCE: 正样本 vs K负样本，softmax 排名，梯度直接推高 1/rank
训练时用 PopNegSampler 采更难的负样本（接近测试集负样本分布）
log-popularity 节点特征
使用 split 列做训练/验证划分
"""
import os
import os.path as osp
import sys
import time
from collections import Counter

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
    return jt.stack(padded, dim=0) if batch_first else jt.stack(padded, dim=1)


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
import argparse

jt.flags.use_cuda = 1
IS_MPI = jt.in_mpi
RANK = jt.rank if IS_MPI else 0
WORLD_SIZE = jt.world_size if IS_MPI else 1
IS_MAIN = RANK == 0


def log(msg):
    if IS_MAIN:
        print(msg, flush=True)


# ============================================================
# Negative Sampler：流行度加权，避免把用户历史正样本采成负样本
# ============================================================
class PopNegSampler:
    def __init__(self, df, p_pop=0.7, seed=42):
        self.p_pop = p_pop
        self._rng = np.random.RandomState(seed)
        dst_vals = df["dst"].values
        self.min_dst = int(dst_vals.min())
        self.max_dst = int(dst_vals.max())

        # 每个 src 的历史正样本集合（避免采成负样本）
        self.src_history = {}
        for s, d in zip(df["src"].values, dst_vals):
            self.src_history.setdefault(int(s), set()).add(int(d))

        # sqrt(freq) 权重
        dst_counts = Counter(dst_vals.tolist())
        dsts = np.array(list(dst_counts.keys()), dtype=np.int32)
        freqs = np.array([dst_counts[d] for d in dsts], dtype=np.float64)
        sqrt_freqs = np.sqrt(freqs)
        self.pop_dsts = dsts
        self.pop_probs = sqrt_freqs / sqrt_freqs.sum()

    def sample(self, src, dst, K):
        """返回 [len(src)*K] 的负样本数组，每个正样本对应 K 个负样本。
        向量化：一次性采 B*K*OV 个候选，用 numpy 布尔索引过滤，消除 B 次独立 choice 调用。
        """
        B = len(src)
        n_pop = max(1, int(K * self.p_pop))
        n_uni = K - n_pop
        OV = 8  # 过采样倍率，保证过滤后足够

        # 一次性采样所有候选（替代 B 次独立 np.random.choice）
        pop_flat = self._rng.choice(
            self.pop_dsts, size=B * n_pop * OV, p=self.pop_probs, replace=True
        ).astype(np.int32).reshape(B, n_pop * OV)  # [B, n_pop*OV]

        uni_flat = self._rng.randint(
            self.min_dst, self.max_dst + 1, size=B * max(n_uni * OV, 1)
        ).astype(np.int32).reshape(B, -1)  # [B, n_uni*OV]

        # 向量化屏蔽掉与正样本 dst 完全相同的候选
        dst_col = dst.reshape(B, 1)
        pop_mask = (pop_flat != dst_col)  # [B, n_pop*OV]
        uni_mask = (uni_flat != dst_col)  # [B, n_uni*OV]

        result = np.empty(B * K, dtype=np.int32)
        for i in range(B):
            p_arr = pop_flat[i, pop_mask[i]][:n_pop]
            if n_uni > 0:
                u_arr = uni_flat[i, uni_mask[i]][:n_uni]
                row = np.concatenate([p_arr, u_arr])
            else:
                row = p_arr

            # 极少数情况：过采样后仍不足，随机补齐
            if len(row) < K:
                extra = self._rng.randint(self.min_dst, self.max_dst + 1, size=K).astype(np.int32)
                row = np.concatenate([row, extra])

            result[i * K:(i + 1) * K] = row[:K]

        return result


# ============================================================
# InfoNCE 损失
# ============================================================
def infonce_loss(pos_logit, neg_logit, K):
    """
    pos_logit: [B]   - 正样本分数
    neg_logit: [B*K] - K 个负样本分数（已按 [B, K] 顺序排列）
    返回 InfoNCE loss = -log(exp(pos) / (exp(pos) + sum(exp(neg))))
    """
    B = pos_logit.shape[0]
    neg = neg_logit.reshape(B, K)                        # [B, K]
    pos = pos_logit.unsqueeze(1)                         # [B, 1]
    logits = jt.concat([pos, neg], dim=1)                # [B, K+1]
    labels = jt.zeros(B, dtype='int32')                  # 正样本在位置 0
    return jt.nn.cross_entropy_loss(logits, labels)


# ============================================================
# 验证（100-way MRR：1正 + 99负，与比赛 test 指标一致）
# ============================================================
VAL_NEG = 99        # 每个正样本对应的负样本数，与 test 100-way 一致
VAL_NEG_CHUNK = 33  # 每次 forward 处理的负样本数（chunk=33 → 3次forward/batch，避免OOM）
MAX_VAL_SAMPLES = 5000  # 快速验证：对 val 集随机抽样，~8分钟完成

def test_val(model, loader):
    model.eval()
    backbone, predictor = model[0], model[1]
    ap_sum, auc_sum, n_batches = 0.0, 0.0, 0
    mrr_sum, mrr_count = 0.0, 0

    with jt.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(loader, ncols=120, desc="Val", disable=not IS_MAIN)):
            src = np.array(batch_data.src).astype(np.int32)
            dst = np.array(batch_data.dst).astype(np.int32)
            t   = np.array(batch_data.t).astype(np.float32)
            neg_dst_flat = np.array(batch_data.neg_dst).astype(np.int32)  # [B*VAL_NEG]
            B = len(src)
            K_val = len(neg_dst_flat) // B  # 实际负样本数

            # 正样本得分
            src_emb_p, dst_emb_p = backbone.compute_src_dst_node_temporal_embeddings(src, dst, t)
            pos_score = jt.sigmoid(predictor(src_emb_p, dst_emb_p).squeeze(-1)).numpy()  # [B]
            del src_emb_p, dst_emb_p

            # 负样本分块 forward，避免 B*99 一次性 OOM
            neg_dst_mat = neg_dst_flat.reshape(B, K_val)  # [B, K_val]
            neg_scores_chunks = []
            for c in range(0, K_val, VAL_NEG_CHUNK):
                chunk = neg_dst_mat[:, c:c + VAL_NEG_CHUNK]  # [B, chunk_size]
                cs = chunk.shape[1]
                src_rep = np.repeat(src, cs)
                t_rep   = np.repeat(t, cs)
                src_emb_n, neg_emb = backbone.compute_src_dst_node_temporal_embeddings(
                    src_rep, chunk.reshape(-1), t_rep)
                ns = jt.sigmoid(predictor(src_emb_n, neg_emb).squeeze(-1)).numpy()  # [B*cs]
                neg_scores_chunks.append(ns.reshape(B, cs))
                del src_emb_n, neg_emb

            neg_score = np.concatenate(neg_scores_chunks, axis=1)  # [B, K_val]

            # AP/AUC（用全部负样本）
            y_true  = np.concatenate([np.ones(B), np.zeros(B * K_val)])
            y_score = np.concatenate([pos_score, neg_score.reshape(-1)])
            ap_sum  += float(average_precision_score(y_true, y_score))
            auc_sum += float(roc_auc_score(y_true, y_score))
            n_batches += 1

            # 100-way MRR：pos 在 [pos, neg_1..neg_99] 中的排名
            ranks = 1 + np.sum(neg_score > pos_score.reshape(B, 1), axis=1)  # [B]
            mrr_sum   += float(np.sum(1.0 / ranks))
            mrr_count += B

            if (batch_idx + 1) % 16 == 0:
                jt.gc()
            jt.sync_all()
    jt.gc()

    if IS_MPI:
        agg = jt.array([ap_sum, auc_sum, mrr_sum, float(n_batches), float(mrr_count)], dtype="float32").mpi_all_reduce("add")
        ap_sum, auc_sum, mrr_sum, n_batches, mrr_count = agg.numpy().tolist()

    if n_batches == 0:
        return {"AP": 0.0, "AUC": 0.0, "MRR": 0.0}
    return {"AP": ap_sum / n_batches, "AUC": auc_sum / n_batches, "MRR": mrr_sum / mrr_count}


# ============================================================
# 训练（InfoNCE + PopNegSampler）
# ============================================================
def train(model, optimizer, train_loader, val_loader, neg_sampler, K,
          num_epochs, save_path, dataset_name, early_stop_patience=5):
    """
    neg_chunk: 每次 forward 最多处理的负样本对数。
    防止 B*K 对全部一次 forward 导致 OOM。
    K=19, batch_size=200 → B*K=3800，分成 neg_chunk=200 的 19 个 chunk。
    """
    best_mrr = 0.0
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        backbone, predictor = model[0], model[1]
        losses = []

        pbar = tqdm(train_loader, ncols=120, desc=f"Epoch {epoch+1}", disable=not IS_MAIN)
        for batch_idx, batch_data in enumerate(pbar):
            src = np.array(batch_data.src).astype(np.int32)
            dst = np.array(batch_data.dst).astype(np.int32)
            t = np.array(batch_data.t).astype(np.float32)
            B = len(src)

            # 采 K 个负样本
            neg_dst = neg_sampler.sample(src, dst, K)  # [B*K]

            # 正样本 forward
            src_emb_p, dst_emb_p = backbone.compute_src_dst_node_temporal_embeddings(src, dst, t)
            pos_logit = predictor(src_emb_p, dst_emb_p).squeeze(-1)  # [B]
            del src_emb_p, dst_emb_p

            # 负样本：把 B*K 对拼成一个大 batch，一次 forward
            # DyGFormer 中 src_emb 只依赖 (src, t)，和 dst 无关
            # 所以 src 重复 K 次后，src 侧的计算量也是 K 倍——这是主要开销
            # K 不能太大；K=3 约 2x 慢于 BCE，可接受
            src_rep = np.repeat(src, K)
            t_rep   = np.repeat(t, K)
            src_emb_n, neg_emb = backbone.compute_src_dst_node_temporal_embeddings(
                src_rep, neg_dst, t_rep)
            neg_logit = predictor(src_emb_n, neg_emb).squeeze(-1)    # [B*K]
            del src_emb_n, neg_emb

            loss = infonce_loss(pos_logit, neg_logit, K)
            optimizer.zero_grad()
            optimizer.step(loss)
            jt.sync_all()

            losses.append(loss.item())
            pbar.set_description(f"Epoch {epoch+1} loss={loss.item():.4f}")

            del pos_logit, neg_logit
            if (batch_idx + 1) % 32 == 0:
                jt.gc()

        log(f"Epoch {epoch+1} Train Loss: {np.mean(losses):.4f}")

        if val_loader is not None:
            val_res = test_val(model, val_loader)
            log(f"Epoch {epoch+1} Val: {val_res}")
            cur = val_res["MRR"]
            if cur > best_mrr:
                best_mrr = cur
                patience_counter = 0
                if IS_MAIN:
                    jt.save(model.state_dict(), f"{save_path}/{dataset_name}_DyGFormer_best.pkl")
                    os.sync()
                log(f"  -> New best MRR: {best_mrr:.6f}")
            else:
                patience_counter += 1
                log(f"  -> No improvement ({patience_counter}/{early_stop_patience}), best={best_mrr:.6f}")
            if IS_MAIN:
                jt.save(model.state_dict(), f"{save_path}/{dataset_name}_DyGFormer.pkl")
                os.sync()
            if patience_counter >= early_stop_patience:
                log("Early stopping.")
                break
        else:
            if IS_MAIN:
                jt.save(model.state_dict(), f"{save_path}/{dataset_name}_DyGFormer_best.pkl")
                jt.save(model.state_dict(), f"{save_path}/{dataset_name}_DyGFormer.pkl")
                os.sync()

        jt.gc()
    return best_mrr


# ============================================================
# 测试推断（与 1DyGFormer 相同）
# ============================================================
def test_competition(model, test_src, test_time, test_candidates,
                     tmp_dir, dataset_name, batch_size=5, cand_chunk_size=25):
    model.eval()
    backbone, predictor = model[0], model[1]
    n = len(test_src)
    num_cands = test_candidates.shape[1]
    my_idx = np.arange(RANK, n, WORLD_SIZE, dtype=np.int64) if IS_MPI else np.arange(n, dtype=np.int64)
    my_n = len(my_idx)
    my_scores = np.zeros((my_n, num_cands), dtype=np.float32)
    chunk_pairs = max(1, min(cand_chunk_size, batch_size * num_cands))

    pbar = tqdm(range((my_n + batch_size - 1) // batch_size), ncols=120, desc=f"Testing[r{RANK}]")
    with jt.no_grad():
        for batch_idx in pbar:
            start = batch_idx * batch_size
            end = min(start + batch_size, my_n)
            idx_chunk = my_idx[start:end]
            b = len(idx_chunk)
            total_pairs = b * num_cands
            src_rep = np.repeat(test_src[idx_chunk], num_cands).astype(np.int32)
            t_rep = np.repeat(test_time[idx_chunk].astype(np.float32), num_cands)
            cand_flat = test_candidates[idx_chunk].reshape(-1).astype(np.int32)
            probs_buf = np.empty(total_pairs, dtype=np.float32)
            for cs in range(0, total_pairs, chunk_pairs):
                ce = min(cs + chunk_pairs, total_pairs)
                se, de = backbone.compute_src_dst_node_temporal_embeddings(src_rep[cs:ce], cand_flat[cs:ce], t_rep[cs:ce])
                probs_buf[cs:ce] = jt.sigmoid(predictor(se, de).squeeze(-1)).numpy().astype(np.float32)
                del se, de
                jt.sync_all()
            my_scores[start:end] = probs_buf.reshape(b, num_cands)
            if (batch_idx + 1) % 64 == 0:
                jt.gc()
    jt.gc()

    if not IS_MPI:
        full = np.zeros((n, num_cands), dtype=np.float32)
        full[my_idx] = my_scores
        return full

    # MPI gather（与 1DyGFormer 相同逻辑）
    def _save_fsync(path, arr):
        np.save(path, arr)
        fd = os.open(path, os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)

    np.save(f"{tmp_dir}/_tmp_{dataset_name}_rank{RANK}_idx.npy", my_idx)
    np.save(f"{tmp_dir}/_tmp_{dataset_name}_rank{RANK}_scores.npy", my_scores)
    np.save(f"{tmp_dir}/_tmp_{dataset_name}_rank{RANK}_done.npy", np.array([1], dtype=np.int8))

    def _wait(path, timeout=300):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if os.path.exists(path) and os.path.getsize(path) > 0:
                return True
            time.sleep(1)
        return False

    missing = [r for r in range(WORLD_SIZE) if not _wait(f"{tmp_dir}/_tmp_{dataset_name}_rank{r}_done.npy")]
    if missing:
        print(f"[r{RANK}] missing ranks {missing}")
        return None
    if not IS_MAIN:
        return None

    all_idx = np.concatenate([np.load(f"{tmp_dir}/_tmp_{dataset_name}_rank{r}_idx.npy") for r in range(WORLD_SIZE)])
    all_sc = np.concatenate([np.load(f"{tmp_dir}/_tmp_{dataset_name}_rank{r}_scores.npy") for r in range(WORLD_SIZE)])
    result = all_sc[np.argsort(all_idx)]
    for r in range(WORLD_SIZE):
        for suf in ["_idx.npy", "_scores.npy", "_done.npy"]:
            try: os.remove(f"{tmp_dir}/_tmp_{dataset_name}_rank{r}{suf}")
            except: pass
    return result


# ============================================================
# Main
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--data_dir", type=str, default="./data")
parser.add_argument("--save_dir", type=str, default="./saved_models")
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=200)
parser.add_argument("--test_batch_size", type=int, default=5)
parser.add_argument("--cand_chunk_size", type=int, default=25)
parser.add_argument("--early_stop", type=int, default=5)
parser.add_argument("--node_feat_dim", type=int, default=128)
parser.add_argument("--edge_feat_dim", type=int, default=128)
parser.add_argument("--time_feat_dim", type=int, default=100)
parser.add_argument("--channel_dim", type=int, default=128)
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--num_heads", type=int, default=2)
parser.add_argument("--patch_size", type=int, default=1)
parser.add_argument("--max_seq_len", type=int, default=32)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--lr", type=float, default=1e-4)
# 核心参数
parser.add_argument("--num_neg", type=int, default=19,
                    help="InfoNCE 负样本数，模拟 (num_neg+1)-way 排名。默认 19=20-way。")
parser.add_argument("--p_pop", type=float, default=0.7,
                    help="负样本中流行度加权的比例（0=全随机，1=全流行度）")
parser.add_argument("--neg_chunk", type=int, default=200,
                    help="训练时每次 forward 的负样本对数上限，防止 OOM")
parser.add_argument("--use_all_train", action="store_true",
                    help="训练集用全量 train.csv（不留验证集），用于最终提交")
parser.add_argument("--eval_only", action="store_true")
parser.add_argument("--model_path", type=str, default=None)
args = parser.parse_args()

if args.output_dir is None:
    args.output_dir = args.data_dir

save_path = osp.abspath(args.save_dir)
if IS_MAIN:
    os.makedirs(save_path, exist_ok=True)

log("=" * 80)
log(f"7DyGFormer [InfoNCE] - Dataset: {args.dataset}")
log(f"  num_neg={args.num_neg}, p_pop={args.p_pop}, max_seq_len={args.max_seq_len}")
log(f"  use_all_train={args.use_all_train}")
log("=" * 80)

df = pd.read_csv(f"{args.data_dir}/{args.dataset}/train.csv")
test_df = pd.read_csv(f"{args.data_dir}/{args.dataset}/test.csv")

# 训练/验证划分
if args.use_all_train:
    train_df = df
    val_df = None
    log(f"use_all_train: {len(train_df)} 行，无验证集")
elif "split" in df.columns:
    train_df = df[df["split"] == 0].reset_index(drop=True)
    val_df   = df[df["split"] == 1].reset_index(drop=True)
    log(f"split 列: train={len(train_df)}, val={len(val_df)}")
else:
    n_val = int(len(df) * 0.15)
    train_df = df.iloc[:-n_val].reset_index(drop=True)
    val_df   = df.iloc[-n_val:].reset_index(drop=True)
    log(f"85/15 split: train={len(train_df)}, val={len(val_df)}")

test_src = test_df["src"].values.astype(np.int32)
test_time = test_df["time"].values.astype(np.int32)
test_candidates = test_df.iloc[:, 2:].values.astype(np.int32)

# 全量数据用于 neighbor sampler（测试时能看到所有历史）
src_all = df["src"].values.astype(np.int32)
dst_all = df["dst"].values.astype(np.int32)
t_all   = df["time"].values.astype(np.int32)
eid_all = np.arange(len(df), dtype=np.int32) + 1

max_node = max(int(src_all.max()), int(dst_all.max()), int(test_candidates.max()))
node_size = max_node + 1

# 节点特征：log(degree+1) 流行度特征
node_raw_features = np.zeros((node_size, args.node_feat_dim), dtype=np.float32)
deg_counter = Counter()
deg_counter.update(src_all.tolist())
deg_counter.update(dst_all.tolist())
for nid, deg in deg_counter.items():
    if nid < node_size:
        node_raw_features[nid, 0] = float(np.log1p(deg))
log(f"Pop feature: {(node_raw_features[:, 0] > 0).sum()} 节点有非零特征，"
    f"最大值={node_raw_features[:, 0].max():.2f}")

num_edges = len(df)
edge_raw_features = np.zeros((num_edges + 1, args.edge_feat_dim), dtype=np.float32)

# TemporalData
train_src = train_df["src"].values.astype(np.int32)
train_dst = train_df["dst"].values.astype(np.int32)
train_t   = train_df["time"].values.astype(np.int32)
train_eid = np.arange(len(train_df), dtype=np.int32) + 1

train_data = TemporalData(src=jt.Var(train_src), dst=jt.Var(train_dst),
                          t=jt.Var(train_t), edge_ids=jt.Var(train_eid))

# 验证集 DataLoader（100-way MRR，随机抽 MAX_VAL_SAMPLES 样本加速）
if val_df is not None:
    if len(val_df) > MAX_VAL_SAMPLES:
        val_df_eval = val_df.sample(n=MAX_VAL_SAMPLES, random_state=42).reset_index(drop=True)
        log(f"Val 抽样: {MAX_VAL_SAMPLES}/{len(val_df)} 行用于快速验证")
    else:
        val_df_eval = val_df
    val_src = val_df_eval["src"].values.astype(np.int32)
    val_dst = val_df_eval["dst"].values.astype(np.int32)
    val_t   = val_df_eval["time"].values.astype(np.int32)
    val_eid = np.arange(len(val_df_eval), dtype=np.int32) + len(train_df) + 1
    val_data = TemporalData(src=jt.Var(val_src), dst=jt.Var(val_dst),
                            t=jt.Var(val_t), edge_ids=jt.Var(val_eid))
    val_loader = TemporalDataLoader(val_data, batch_size=args.batch_size, neg_sampling_ratio=float(VAL_NEG))
else:
    val_loader = None

full_data = TemporalData(src=jt.Var(src_all), dst=jt.Var(dst_all),
                         t=jt.Var(t_all), edge_ids=jt.Var(eid_all))

# 训练 DataLoader（只用来迭代批次，负样本由 PopNegSampler 手动采）
train_loader = TemporalDataLoader(train_data, batch_size=args.batch_size, neg_sampling_ratio=0)

full_neighbor_sampler = get_neighbor_sampler(full_data, "recent", seed=1)
is_bipartite = len(set(src_all.tolist()) & set(dst_all.tolist())) == 0
log(f"Node size: {node_size}, Edges: {num_edges}, Bipartite: {is_bipartite}")

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
    input_dim1=args.node_feat_dim, input_dim2=args.node_feat_dim,
    hidden_dim=args.node_feat_dim, output_dim=1,
)
model = nn.Sequential(backbone, predictor)

if IS_MPI:
    model.mpi_param_broadcast(root=0)

if not args.eval_only:
    neg_sampler = PopNegSampler(train_df, p_pop=args.p_pop)
    optimizer = jt.nn.Adam(list(model.parameters()), lr=args.lr)
    log(f"\nInfoNCE training: K={args.num_neg} negatives/positive, "
        f"simulating {args.num_neg+1}-way ranking")
    train(model, optimizer, train_loader, val_loader, neg_sampler,
          args.num_neg, args.epochs, save_path, args.dataset, args.early_stop)
    if IS_MPI:
        jt.array([0], dtype="int32").mpi_all_reduce("add")
else:
    log("\n--eval_only: skip training.")

log("\nGenerating predictions...")
if args.model_path:
    model.load_state_dict(jt.load(args.model_path))
    log(f"Loaded: {args.model_path}")
else:
    best_path = f"{save_path}/{args.dataset}_DyGFormer_best.pkl"
    last_path = f"{save_path}/{args.dataset}_DyGFormer.pkl"
    if os.path.exists(best_path):
        model.load_state_dict(jt.load(best_path))
        log(f"Loaded best: {best_path}")
    else:
        model.load_state_dict(jt.load(last_path))
        log("Loaded latest model.")

scores = test_competition(model, test_src, test_time, test_candidates,
                          save_path, args.dataset, args.test_batch_size, args.cand_chunk_size)

if IS_MAIN:
    if scores is None:
        sys.exit(0)
    log(f"Scores: {scores.shape}, range=[{scores.min():.4f}, {scores.max():.4f}]")
    output_file = f"{args.output_dir}/{args.dataset}/{args.dataset}_result.csv"
    os.makedirs(osp.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        for row in scores:
            f.write(",".join([f"{p:.8f}" for p in row]) + "\n")
    log(f"Saved: {output_file}")

log("\n" + "=" * 80 + "\nDONE\n" + "=" * 80)
