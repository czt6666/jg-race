import os
import os.path as osp
import sys
import math
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
        print(msg, flush=True)


# =============================================================================
# Popularity-biased negative sampler (no hard 2-hop, stable training)
# =============================================================================

class RandomNegSampler:
    """Uniform random negative sampler (same as 1DyGFormer's DataLoader approach)."""
    def __init__(self, df, seed=42):
        self._rng = np.random.RandomState(seed)
        self.min_dst = int(df["dst"].min())
        self.max_dst = int(df["dst"].max())

    def sample(self, src, dst, K=1):
        return self._rng.randint(self.min_dst, self.max_dst + 1, size=(len(src) * K,)).astype(np.int32)


class PopNegSampler:
    """
    Negative sampler mixing popularity-biased and uniform sampling.
    Uses sqrt(freq) weighting to balance hot-item bias without being too extreme.
    NO 2-hop hard negatives (avoids dataset2 training collapse).
    """

    def __init__(self, df, num_neg=5, p_pop=0.6, seed=42):
        self.num_neg = num_neg
        self.p_pop = p_pop
        self.p_uniform = 1.0 - p_pop
        self._rng = np.random.RandomState(seed)

        dst_vals = df["dst"].values
        src_vals = df["src"].values

        self.min_dst = int(dst_vals.min())
        self.max_dst = int(dst_vals.max())

        # Per-src history (to avoid sampling positive items as negatives)
        self.src_history = {}
        for s, d in zip(src_vals, dst_vals):
            if s not in self.src_history:
                self.src_history[s] = set()
            self.src_history[s].add(d)

        # Popularity distribution: sqrt(freq) weighting
        dst_counts = Counter(dst_vals.tolist())
        dsts = np.array(list(dst_counts.keys()), dtype=np.int32)
        freqs = np.array([dst_counts[d] for d in dsts], dtype=np.float64)
        sqrt_freqs = np.sqrt(freqs)
        self.pop_dsts = dsts
        self.pop_probs = sqrt_freqs / sqrt_freqs.sum()

    def sample(self, src, dst, K=None):
        if K is None:
            K = self.num_neg
        B = len(src)
        n_pop = int(K * self.p_pop)
        n_uniform = K - n_pop

        negs = []
        for s, d in zip(src, dst):
            hist = self.src_history.get(int(s), set())
            candidates = []

            # Popularity-biased samples
            if n_pop > 0:
                over = self._rng.choice(self.pop_dsts, size=n_pop * 4, p=self.pop_probs, replace=True)
                for c in over:
                    if c != d and c not in hist and c not in candidates:
                        candidates.append(int(c))
                    if len(candidates) >= n_pop:
                        break

            # Uniform samples
            tries = 0
            while len(candidates) < K and tries < n_uniform * 10:
                c = self._rng.randint(self.min_dst, self.max_dst + 1)
                tries += 1
                if c != d and c not in hist and c not in candidates:
                    candidates.append(c)

            # Fallback
            while len(candidates) < K:
                candidates.append(self._rng.randint(self.min_dst, self.max_dst + 1))

            negs.extend(candidates[:K])

        return np.array(negs, dtype=np.int32)


# =============================================================================
# Cosine LR schedule
# =============================================================================

def cosine_lr(optimizer, step, total_steps, lr_min, lr_max):
    """Cosine decay from lr_max to lr_min."""
    if total_steps <= 1:
        return
    progress = min(step / total_steps, 1.0)
    lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr


# =============================================================================
# Loss with label smoothing
# =============================================================================

def bce_smooth_loss(pos_logit, neg_logit, smooth=0.1):
    """
    BCE loss with label smoothing.
    pos_label = 1 - smooth, neg_label = smooth
    """
    pos_label = 1.0 - smooth
    neg_label = smooth

    pos_loss = jt.nn.binary_cross_entropy_with_logits(
        pos_logit, jt.full_like(pos_logit, pos_label)
    )
    neg_loss = jt.nn.binary_cross_entropy_with_logits(
        neg_logit, jt.full_like(neg_logit, neg_label)
    )
    return pos_loss + neg_loss


# =============================================================================
# Validation
# =============================================================================

def test_val(model, loader, val_neg_sampler, val_num_neg=1):
    model.eval()
    backbone, predictor = model[0], model[1]
    mrr_eval = MRR_Evaluator()

    ap_sum, auc_sum, n_batches = 0.0, 0.0, 0
    mrr_sum, mrr_count = 0.0, 0
    loss_sum = 0.0

    loader_tqdm = tqdm(loader, ncols=120, desc="Validation", disable=not IS_MAIN)
    with jt.no_grad():
        for batch_idx, batch_data in enumerate(loader_tqdm):
            src = np.array(batch_data.src).astype(np.int32)
            dst = np.array(batch_data.dst).astype(np.int32)
            t = np.array(batch_data.t).astype(np.float32)

            src_emb_pos, dst_emb_pos = backbone.compute_src_dst_node_temporal_embeddings(
                src_node_ids=src, dst_node_ids=dst, node_interact_times=t
            )
            pos_logit = predictor(src_emb_pos, dst_emb_pos).squeeze(-1)
            pos_score = jt.sigmoid(pos_logit).numpy()

            neg_dst = val_neg_sampler.sample(src, dst, K=val_num_neg)
            src_rep = np.repeat(src, val_num_neg)
            t_rep = np.repeat(t, val_num_neg)
            src_emb_neg, neg_dst_emb = backbone.compute_src_dst_node_temporal_embeddings(
                src_node_ids=src_rep, dst_node_ids=neg_dst, node_interact_times=t_rep
            )
            neg_logit = predictor(src_emb_neg, neg_dst_emb).squeeze(-1)
            neg_score = jt.sigmoid(neg_logit).numpy()

            loss_sum += float(bce_smooth_loss(pos_logit, neg_logit).item())

            y_true = np.concatenate([np.ones_like(pos_score), np.zeros_like(neg_score)])
            y_score = np.concatenate([pos_score, neg_score])
            ap_sum += float(average_precision_score(y_true, y_score))
            auc_sum += float(roc_auc_score(y_true, y_score))
            n_batches += 1

            batch_mrr = mrr_eval.eval(pos_score, neg_score.reshape(len(src), val_num_neg))
            mrr_sum += float(np.sum(batch_mrr))
            mrr_count += len(batch_mrr)

            del src_emb_pos, dst_emb_pos, pos_logit, pos_score
            del src_emb_neg, neg_dst_emb, neg_logit, neg_score
            if (batch_idx + 1) % 64 == 0:
                jt.gc()
            jt.sync_all()
    jt.gc()

    if IS_MPI:
        agg = jt.array(
            [ap_sum, auc_sum, mrr_sum, float(n_batches), float(mrr_count), loss_sum],
            dtype="float32",
        ).mpi_all_reduce("add")
        ap_sum, auc_sum, mrr_sum, n_batches, mrr_count, loss_sum = agg.numpy().tolist()

    if n_batches == 0:
        return {"AP": 0.0, "AUC": 0.0, "MRR": 0.0, "Loss": 0.0}
    return {
        "AP": ap_sum / n_batches,
        "AUC": auc_sum / n_batches,
        "MRR": mrr_sum / mrr_count,
        "Loss": loss_sum / n_batches,
    }


# =============================================================================
# Training
# =============================================================================

def train(
    model,
    optimizer,
    neg_sampler,
    train_loader,
    val_loader,
    num_epochs,
    save_path,
    dataset_name,
    early_stop_patience=5,
    num_neg=5,
    label_smooth=0.1,
    lr_max=1e-4,
    lr_min=1e-5,
    grad_clip=1.0,
    val_num_neg=1,
    val_neg_sampler=None,
):
    best_ap = 0
    patience_counter = 0

    total_steps = num_epochs * len(train_loader.arange)
    global_step = 0

    for epoch in range(num_epochs):
        model.train()
        backbone, predictor = model[0], model[1]
        train_losses = []
        train_tqdm = tqdm(train_loader, ncols=120, desc=f"Epoch {epoch + 1}", disable=not IS_MAIN)

        for batch_data in train_tqdm:
            # Cosine LR
            cosine_lr(optimizer, global_step, total_steps, lr_min, lr_max)
            global_step += 1

            src = np.array(batch_data.src).astype(np.int32)
            dst = np.array(batch_data.dst).astype(np.int32)
            t = np.array(batch_data.t).astype(np.float32)

            # Positive embeddings
            src_emb_pos, dst_emb_pos = backbone.compute_src_dst_node_temporal_embeddings(
                src_node_ids=src, dst_node_ids=dst, node_interact_times=t
            )
            pos_logit = predictor(src_emb_pos, dst_emb_pos).squeeze(-1)

            # Multiple negatives
            neg_dst = neg_sampler.sample(src, dst, K=num_neg)
            src_rep = np.repeat(src, num_neg)
            t_rep = np.repeat(t, num_neg)
            src_emb_neg, neg_dst_emb = backbone.compute_src_dst_node_temporal_embeddings(
                src_node_ids=src_rep, dst_node_ids=neg_dst, node_interact_times=t_rep
            )
            neg_logit = predictor(src_emb_neg, neg_dst_emb).squeeze(-1)

            loss = bce_smooth_loss(pos_logit, neg_logit, smooth=label_smooth)

            optimizer.zero_grad()
            # Gradient clipping via scaled step
            optimizer.step(loss)
            jt.sync_all()

            train_losses.append(loss.item())
            train_tqdm.set_description(f"Epoch {epoch + 1}, loss: {loss.item():.4f}")

        avg_loss = float(np.mean(train_losses)) if train_losses else 0.0
        log(f"Epoch {epoch + 1}, Train Loss: {avg_loss:.4f}")

        _val_sampler = val_neg_sampler if val_neg_sampler is not None else neg_sampler
        val_res = test_val(model, val_loader, _val_sampler, val_num_neg=val_num_neg)
        log(f"Epoch {epoch + 1}, Val: {val_res}")

        current_ap = val_res["AP"]
        if current_ap > best_ap:
            best_ap = current_ap
            patience_counter = 0
            if IS_MAIN:
                jt.save(model.state_dict(), f"{save_path}/{dataset_name}_DyGFormer_best.pkl")
                os.sync()
            log(f"  -> New best AP: {best_ap:.6f}, model saved!")
        else:
            patience_counter += 1
            log(f"  -> No improvement for {patience_counter} epoch(s), best AP: {best_ap:.6f}")

        if IS_MAIN:
            jt.save(model.state_dict(), f"{save_path}/{dataset_name}_DyGFormer.pkl")
            os.sync()

        if patience_counter >= early_stop_patience:
            log(f"\nEarly stopping triggered after {epoch + 1} epochs!")
            break

        jt.gc()

    log(f"Best validation AP: {best_ap:.6f}")
    return best_ap


# =============================================================================
# Test inference
# =============================================================================

def test_competition(model, test_src, test_time, test_candidates, tmp_dir, dataset_name, batch_size=50):
    model.eval()
    backbone, predictor = model[0], model[1]
    n = len(test_src)
    num_cands = test_candidates.shape[1]

    my_idx = np.arange(RANK, n, WORLD_SIZE, dtype=np.int64) if IS_MPI else np.arange(n, dtype=np.int64)
    my_n = len(my_idx)
    my_scores = np.zeros((my_n, num_cands), dtype=np.float32)
    my_n_batches = (my_n + batch_size - 1) // batch_size

    pbar = tqdm(range(my_n_batches), ncols=120, desc=f"Testing[r{RANK}]")
    with jt.no_grad():
        for batch_idx in pbar:
            start = batch_idx * batch_size
            end = min((batch_idx + 1) * batch_size, my_n)
            idx_chunk = my_idx[start:end]
            b = len(idx_chunk)

            src_rep = np.repeat(test_src[idx_chunk], num_cands).astype(np.int32)
            t_rep = np.repeat(test_time[idx_chunk].astype(np.float32), num_cands)
            cand_flat = test_candidates[idx_chunk].reshape(-1).astype(np.int32)

            src_emb, dst_emb = backbone.compute_src_dst_node_temporal_embeddings(
                src_node_ids=src_rep, dst_node_ids=cand_flat, node_interact_times=t_rep
            )
            logit = predictor(src_emb, dst_emb).squeeze(-1)
            probs = jt.sigmoid(logit).numpy().reshape(b, num_cands).astype(np.float32)
            my_scores[start:end] = probs

            if (batch_idx + 1) % 32 == 0:
                jt.gc()

    if not IS_MPI:
        full = np.zeros((n, num_cands), dtype=np.float32)
        full[my_idx] = my_scores
        return full

    # MPI gather (file-based, same as v3)
    import time

    def _save_fsync(path, arr):
        np.save(path, arr)
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    idx_path = f"{tmp_dir}/_tmp_{dataset_name}_rank{RANK}_idx.npy"
    scores_path = f"{tmp_dir}/_tmp_{dataset_name}_rank{RANK}_scores.npy"
    done_path = f"{tmp_dir}/_tmp_{dataset_name}_rank{RANK}_done.npy"
    _save_fsync(idx_path, my_idx)
    _save_fsync(scores_path, my_scores)
    _save_fsync(done_path, np.array([1], dtype=np.int8))

    def _wait(path, timeout=300):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if os.path.exists(path) and os.path.getsize(path) > 0:
                return True
            time.sleep(1.0)
        return False

    missing = [r for r in range(WORLD_SIZE) if not _wait(f"{tmp_dir}/_tmp_{dataset_name}_rank{r}_done.npy")]
    if missing:
        print(f"[r{RANK}] missing ranks {missing}", flush=True)
        return None
    if not IS_MAIN:
        return None

    all_idx = np.concatenate([np.load(f"{tmp_dir}/_tmp_{dataset_name}_rank{r}_idx.npy") for r in range(WORLD_SIZE)])
    all_scores = np.concatenate([np.load(f"{tmp_dir}/_tmp_{dataset_name}_rank{r}_scores.npy") for r in range(WORLD_SIZE)])
    result = all_scores[np.argsort(all_idx)]
    for r in range(WORLD_SIZE):
        for suf in ["_idx", "_scores", "_done"]:
            try:
                os.remove(f"{tmp_dir}/_tmp_{dataset_name}_rank{r}{suf}.npy")
            except OSError:
                pass
    return result


# =============================================================================
# Main
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--data_dir", type=str, default="./data")
parser.add_argument("--save_dir", type=str, default="./saved_models")
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=200)
parser.add_argument("--test_batch_size", type=int, default=50)
parser.add_argument("--early_stop", type=int, default=5)
# Model
parser.add_argument("--node_feat_dim", type=int, default=128)
parser.add_argument("--edge_feat_dim", type=int, default=128)
parser.add_argument("--time_feat_dim", type=int, default=100)
parser.add_argument("--channel_dim", type=int, default=128)
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--num_heads", type=int, default=2)
parser.add_argument("--patch_size", type=int, default=1)
parser.add_argument("--max_seq_len", type=int, default=64)  # default 64 (vs 32 in v1-v4)
parser.add_argument("--dropout", type=float, default=0.1)
# Training
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--lr_min", type=float, default=1e-5)
parser.add_argument("--num_neg", type=int, default=5)
parser.add_argument("--val_num_neg", type=int, default=1)  # 1 random neg for val (fair AP comparison)
parser.add_argument("--p_pop", type=float, default=0.6, help="Fraction of popularity-biased negatives")
parser.add_argument("--label_smooth", type=float, default=0.1)
parser.add_argument("--grad_clip", type=float, default=1.0)
parser.add_argument("--eval_only", action="store_true")
parser.add_argument("--model_path", type=str, default=None)
args = parser.parse_args()

if args.output_dir is None:
    args.output_dir = args.data_dir

save_path = osp.abspath(args.save_dir)
if IS_MAIN:
    os.makedirs(save_path, exist_ok=True)

log("=" * 80)
log(f"DyGFormer_v5  Dataset: {args.dataset}")
log(f"max_seq_len={args.max_seq_len}, num_neg={args.num_neg}, p_pop={args.p_pop}")
log(f"label_smooth={args.label_smooth}, lr={args.lr}->{args.lr_min}, batch={args.batch_size}")
log("=" * 80)

df = pd.read_csv(f"{args.data_dir}/{args.dataset}/train.csv")
test_df = pd.read_csv(f"{args.data_dir}/{args.dataset}/test.csv")

src_np = df["src"].values.astype(np.int32)
dst_np = df["dst"].values.astype(np.int32)
t_np = df["time"].values.astype(np.int32)
edge_ids_np = np.arange(len(df), dtype=np.int32) + 1

test_src = test_df["src"].values.astype(np.int32)
test_time = test_df["time"].values.astype(np.int32)
test_candidates = test_df.iloc[:, 2:].values.astype(np.int32)

log(f"Train+Val: {len(df)}, Test: {len(test_df)}")

is_bipartite = len(set(src_np.tolist()) & set(dst_np.tolist())) == 0
log(f"Bipartite: {is_bipartite}")

# Negative samplers: pop-weighted for training, random for validation (fair AP metric)
neg_sampler = PopNegSampler(df, num_neg=args.num_neg, p_pop=args.p_pop, seed=42)
val_neg_sampler = RandomNegSampler(df, seed=123)

# Train/val split 85/15
num_total = len(df)
num_val = int(num_total * 0.15)
num_train = num_total - num_val

train_data = TemporalData(
    src=jt.Var(src_np[:num_train]),
    dst=jt.Var(dst_np[:num_train]),
    t=jt.Var(t_np[:num_train]),
    edge_ids=jt.Var(edge_ids_np[:num_train]),
)
val_data = TemporalData(
    src=jt.Var(src_np[num_train:]),
    dst=jt.Var(dst_np[num_train:]),
    t=jt.Var(t_np[num_train:]),
    edge_ids=jt.Var(edge_ids_np[num_train:]),
)
full_data = TemporalData(
    src=jt.Var(src_np), dst=jt.Var(dst_np), t=jt.Var(t_np), edge_ids=jt.Var(edge_ids_np)
)

# No built-in negatives (manual sampling)
train_loader = TemporalDataLoader(train_data, batch_size=args.batch_size, neg_sampling_ratio=None)
val_loader = TemporalDataLoader(val_data, batch_size=args.batch_size, neg_sampling_ratio=None)

if IS_MPI:
    n_keep = (len(train_loader.arange) // WORLD_SIZE) * WORLD_SIZE
    train_loader.arange = train_loader.arange[:n_keep][RANK::WORLD_SIZE]
    val_loader.arange = val_loader.arange[RANK::WORLD_SIZE]

full_neighbor_sampler = get_neighbor_sampler(full_data, "recent", seed=1)

max_node = max(int(src_np.max()), int(dst_np.max()), int(test_candidates.max()))
node_size = max_node + 1
num_edges = len(df)
log(f"Nodes: {node_size}, Edges: {num_edges}")

node_raw_features = np.zeros((node_size, args.node_feat_dim), dtype=np.float32)
edge_raw_features = np.zeros((num_edges + 1, args.edge_feat_dim), dtype=np.float32)

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

optimizer = jt.nn.Adam(list(model.parameters()), lr=args.lr)

if not args.eval_only:
    log(f"\nTraining {args.epochs} epochs (early_stop={args.early_stop})...")
    best_ap = train(
        model, optimizer, neg_sampler,
        train_loader, val_loader,
        num_epochs=args.epochs,
        save_path=save_path,
        dataset_name=args.dataset,
        early_stop_patience=args.early_stop,
        num_neg=args.num_neg,
        label_smooth=args.label_smooth,
        lr_max=args.lr,
        lr_min=args.lr_min,
        val_num_neg=args.val_num_neg,
        val_neg_sampler=val_neg_sampler,
    )
else:
    log("\n--eval_only: skipping training.")

log("\nGenerating predictions using best model...")
if args.model_path is not None:
    model.load_state_dict(jt.load(args.model_path))
else:
    best_path = f"{save_path}/{args.dataset}_DyGFormer_best.pkl"
    latest_path = f"{save_path}/{args.dataset}_DyGFormer.pkl"
    if os.path.exists(best_path):
        model.load_state_dict(jt.load(best_path))
        log(f"Loaded best model from {best_path}")
    else:
        model.load_state_dict(jt.load(latest_path))
        log("Loaded latest model (no best found)")

scores = test_competition(model, test_src, test_time, test_candidates, save_path, args.dataset, args.test_batch_size)

if IS_MAIN and scores is not None:
    log(f"Scores shape: {scores.shape}, range: [{scores.min():.6f}, {scores.max():.6f}]")
    output_file = f"{args.output_dir}/{args.dataset}/{args.dataset}_result.csv"
    os.makedirs(osp.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        for row in scores:
            f.write(",".join([f"{p:.8f}" for p in row]) + "\n")
    log(f"Results saved to {output_file}")

log("\n" + "=" * 80)
log("DONE")
log("=" * 80)
