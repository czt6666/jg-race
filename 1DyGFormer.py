import os
import os.path as osp
import sys

# Add JittorGeometric to path
root = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, root)

os.environ['JT_SYNC'] = '1'

import jittor as jt
from jittor import nn
import types

# Monkey-patch: JittorGeometric's DyGFormer calls jt.nn.utils.rnn.pad_sequence,
# which does not exist in this Jittor version. Inject a compatible implementation
# before importing JittorGeometric modules.
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

if not hasattr(jt.nn, 'utils'):
    jt.nn.utils = types.ModuleType('utils')
if not hasattr(jt.nn.utils, 'rnn'):
    jt.nn.utils.rnn = types.ModuleType('rnn')
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

# MPI / multi-GPU helpers. When launched via `mpirun -np N python 1DyGFormer.py ...`,
# jt.in_mpi == True and jt.rank/jt.world_size are populated. Otherwise we run single-GPU.
IS_MPI = jt.in_mpi
RANK = jt.rank if IS_MPI else 0
WORLD_SIZE = jt.world_size if IS_MPI else 1
IS_MAIN = (RANK == 0)


def log(msg):
    """Only rank 0 prints, to avoid 4x duplicated logs."""
    if IS_MAIN:
        print(msg)


# Validation: each rank iterates its sharded slice of val_loader, then we
# aggregate scalar metrics across ranks via mpi_all_reduce so every rank
# returns the same global numbers (early-stop decision stays in sync).
def test_val(model, loader):
    model.eval()
    backbone, predictor = model[0], model[1]
    mrr_eval = MRR_Evaluator()

    ap_sum, auc_sum, n_batches = 0.0, 0.0, 0
    mrr_sum, mrr_count = 0.0, 0

    loader_tqdm = tqdm(loader, ncols=120, desc='Validation', disable=not IS_MAIN)
    for _, batch_data in enumerate(loader_tqdm):
        src = np.array(batch_data.src).astype(np.int32)
        dst = np.array(batch_data.dst).astype(np.int32)
        t = np.array(batch_data.t).astype(np.float32)
        neg_dst = np.array(batch_data.neg_dst).astype(np.int32)

        # positive pairs: (src, dst, t)
        src_emb_pos, dst_emb_pos = backbone.compute_src_dst_node_temporal_embeddings(
            src_node_ids=src, dst_node_ids=dst, node_interact_times=t)
        pos_logit = predictor(src_emb_pos, dst_emb_pos).squeeze(-1)
        pos_score = jt.sigmoid(pos_logit).numpy()

        # negative pairs: (src, neg_dst, t)
        src_emb_neg, neg_dst_emb = backbone.compute_src_dst_node_temporal_embeddings(
            src_node_ids=src, dst_node_ids=neg_dst, node_interact_times=t)
        neg_logit = predictor(src_emb_neg, neg_dst_emb).squeeze(-1)
        neg_score = jt.sigmoid(neg_logit).numpy()

        y_true = np.concatenate([np.ones_like(pos_score), np.zeros_like(neg_score)])
        y_score = np.concatenate([pos_score, neg_score])
        ap_sum += float(average_precision_score(y_true, y_score))
        auc_sum += float(roc_auc_score(y_true, y_score))
        n_batches += 1

        batch_mrr = mrr_eval.eval(pos_score, neg_score)
        mrr_sum += float(np.sum(batch_mrr))
        mrr_count += len(batch_mrr)

        # explicit cleanup to prevent memory accumulation during long validation
        del src_emb_pos, dst_emb_pos, pos_logit, pos_score
        del src_emb_neg, neg_dst_emb, neg_logit, neg_score
        jt.sync_all()

    # Aggregate scalars across ranks. Pack into one Var, single all-reduce.
    if IS_MPI:
        agg = jt.array(
            [ap_sum, auc_sum, mrr_sum, float(n_batches), float(mrr_count)],
            dtype='float32').mpi_all_reduce('add')
        ap_sum, auc_sum, mrr_sum, n_batches, mrr_count = agg.numpy().tolist()

    if n_batches == 0 or mrr_count == 0:
        return {'AP': 0.0, 'AUC': 0.0, 'MRR': 0.0}
    return {'AP': ap_sum / n_batches, 'AUC': auc_sum / n_batches, 'MRR': mrr_sum / mrr_count}


def train(model, optimizer, criterion, train_loader, val_loader,
          num_epochs, save_path, dataset_name, early_stop_patience=10):
    best_ap = 0
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        backbone, predictor = model[0], model[1]
        train_losses = []
        train_tqdm = tqdm(train_loader, ncols=120, desc=f'Epoch {epoch + 1}', disable=not IS_MAIN)

        for batch_idx, batch_data in enumerate(train_tqdm):
            src = np.array(batch_data.src).astype(np.int32)
            dst = np.array(batch_data.dst).astype(np.int32)
            t = np.array(batch_data.t).astype(np.float32)
            neg_dst = np.array(batch_data.neg_dst).astype(np.int32)

            # positive pairs
            src_emb_pos, dst_emb_pos = backbone.compute_src_dst_node_temporal_embeddings(
                src_node_ids=src, dst_node_ids=dst, node_interact_times=t)
            pos_logit = predictor(src_emb_pos, dst_emb_pos)

            # negative pairs (random sampled by TemporalDataLoader)
            src_emb_neg, neg_dst_emb = backbone.compute_src_dst_node_temporal_embeddings(
                src_node_ids=src, dst_node_ids=neg_dst, node_interact_times=t)
            neg_logit = predictor(src_emb_neg, neg_dst_emb)

            loss = criterion(pos_logit, jt.ones_like(pos_logit)) + \
                   criterion(neg_logit, jt.zeros_like(neg_logit))

            optimizer.zero_grad()
            optimizer.step(loss)
            jt.sync_all()
            train_losses.append(loss.item())
            train_tqdm.set_description(f'Epoch {epoch + 1}, loss: {loss.item():.4f}')

        log(f'Epoch {epoch + 1}, Train Loss (rank0 shard): {np.mean(train_losses):.4f}')
        val_res = test_val(model, val_loader)  # identical across ranks after all-reduce
        log(f'Epoch {epoch + 1}, Val: {val_res}')

        # All ranks compute the same decision (val_res is globally identical).
        # File IO is rank-0 only; jt.save does no MPI ops so it's safe inside the guard.
        current_ap = val_res['AP']
        if current_ap > best_ap:
            best_ap = current_ap
            patience_counter = 0
            if IS_MAIN:
                jt.save(model.state_dict(), f'{save_path}/{dataset_name}_DyGFormer_best.pkl')
                os.sync()
            if IS_MPI:
                jt.array([0], dtype='int32').mpi_all_reduce('add')
            log(f'  -> New best AP: {best_ap:.6f}, model saved!')
        else:
            patience_counter += 1
            log(f'  -> No improvement for {patience_counter} epoch(s), best AP: {best_ap:.6f}')

        if IS_MAIN:
            jt.save(model.state_dict(), f'{save_path}/{dataset_name}_DyGFormer.pkl')
            os.sync()
        if IS_MPI:
            jt.array([0], dtype='int32').mpi_all_reduce('add')

        if patience_counter >= early_stop_patience:
            log(f'\nEarly stopping triggered after {epoch + 1} epochs!')
            log(f'Best validation AP: {best_ap:.6f}')
            break

    return best_ap


# Test inference, parallel across ranks. Each rank scores its strided slice of
# queries [RANK::WORLD_SIZE], saves its (idx, scores) to a per-rank .npz, then
# rank 0 reads all four files and stitches them back into the full (n, 100) array.
#
# Why files instead of mpi_all_reduce: the score matrix is ~24MB. On a non-CUDA-
# aware OpenMPI, all-reducing a GPU Var of that size is brittle (can hang). File
# gather is unconditional and doesn't care about MPI build flavor.
def test_competition(model, test_src, test_time, test_candidates,
                      tmp_dir, dataset_name, batch_size=25):
    model.eval()
    backbone, predictor = model[0], model[1]
    n = len(test_src)
    num_cands = test_candidates.shape[1]

    my_idx = np.arange(RANK, n, WORLD_SIZE, dtype=np.int64) if IS_MPI else np.arange(n, dtype=np.int64)
    my_n = len(my_idx)
    my_n_batches = (my_n + batch_size - 1) // batch_size
    my_scores = np.zeros((my_n, num_cands), dtype=np.float32)

    # Show progress on EVERY rank (with rank tag) so we can spot stragglers.
    pbar = tqdm(range(my_n_batches), ncols=120,
                desc=f'Testing[r{RANK}]', position=RANK)
    for batch_idx in pbar:
        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, my_n)
        idx_chunk = my_idx[start:end]
        b = len(idx_chunk)

        src_rep = np.repeat(test_src[idx_chunk], num_cands).astype(np.int32)
        t_rep = np.repeat(test_time[idx_chunk].astype(np.float32), num_cands)
        cand_flat = test_candidates[idx_chunk].reshape(-1).astype(np.int32)

        src_emb, dst_emb = backbone.compute_src_dst_node_temporal_embeddings(
            src_node_ids=src_rep, dst_node_ids=cand_flat, node_interact_times=t_rep)
        logit = predictor(src_emb, dst_emb).squeeze(-1)
        probs = jt.sigmoid(logit).numpy().reshape(b, num_cands).astype(np.float32)
        my_scores[start:end] = probs
        # explicit cleanup to keep GPU / host memory low under heavy inference
        del src_emb, dst_emb, logit, probs
        jt.sync_all()

    # Single-GPU fast path
    if not IS_MPI:
        full = np.zeros((n, num_cands), dtype=np.float32)
        full[my_idx] = my_scores
        return full

    # Multi-rank: each rank dumps its slice with plain np.save (no zip overhead).
    idx_path = f'{tmp_dir}/_tmp_{dataset_name}_rank{RANK}_idx.npy'
    scores_path = f'{tmp_dir}/_tmp_{dataset_name}_rank{RANK}_scores.npy'
    np.save(idx_path, my_idx)
    np.save(scores_path, my_scores)
    os.sync()
    jt.array([0], dtype='int32').mpi_all_reduce('add')  # barrier

    if not IS_MAIN:
        return None  # only rank 0 returns the merged result; others skip CSV write

    # Gather shards, sort by original index, return concatenated scores.
    # Avoids a large zero-initialized 'full' matrix peak.
    all_idx, all_scores = [], []
    for r in range(WORLD_SIZE):
        p_idx = f'{tmp_dir}/_tmp_{dataset_name}_rank{r}_idx.npy'
        p_scores = f'{tmp_dir}/_tmp_{dataset_name}_rank{r}_scores.npy'
        if not os.path.exists(p_scores):
            raise RuntimeError(
                f'Rank {r} did not produce its shard file: {p_scores}. '
                f'This usually means the process was OOM-killed or crashed before saving. '
                f'Try reducing --test_batch_size (current={batch_size}).')
        all_idx.append(np.load(p_idx))
        all_scores.append(np.load(p_scores))
        os.remove(p_idx)
        os.remove(p_scores)
    all_idx = np.concatenate(all_idx)
    all_scores = np.concatenate(all_scores, axis=0)
    sort_order = np.argsort(all_idx)
    return all_scores[sort_order]


# Main
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
parser.add_argument('--save_dir', type=str, default='./saved_models', help='Model save directory')
parser.add_argument('--output_dir', type=str, default=None, help='Output directory (default: same as data_dir)')
parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=200, help='Batch size for train/val')
parser.add_argument('--test_batch_size', type=int, default=5,
                    help='Test batch size in queries (each fans out to 100 candidates)')
parser.add_argument('--early_stop', type=int, default=10, help='Early stopping patience')
# DyGFormer hyper-parameters
parser.add_argument('--node_feat_dim', type=int, default=128)
parser.add_argument('--edge_feat_dim', type=int, default=128)
parser.add_argument('--time_feat_dim', type=int, default=100)
parser.add_argument('--channel_dim', type=int, default=128)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--num_heads', type=int, default=2)
parser.add_argument('--patch_size', type=int, default=1)
parser.add_argument('--max_seq_len', type=int, default=32)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--eval_only', action='store_true',
                    help='Skip training and directly evaluate using the saved best model')
parser.add_argument('--model_path', type=str, default=None,
                    help='Explicit model checkpoint path to load in eval_only mode '
                         '(overrides the default best/latest auto-search)')
args = parser.parse_args()

if args.output_dir is None:
    args.output_dir = args.data_dir

save_path = args.save_dir
if IS_MAIN:
    os.makedirs(save_path, exist_ok=True)

log('=' * 80)
log(f'DyGFormer Competition - Dataset: {args.dataset}')
log(f'MPI: in_mpi={IS_MPI}, rank={RANK}, world_size={WORLD_SIZE}')
log('=' * 80)

# Load data - use int32 like main.py
df = pd.read_csv(f'{args.data_dir}/{args.dataset}/train.csv')
test_df = pd.read_csv(f'{args.data_dir}/{args.dataset}/test.csv')

src_np = df['src'].values.astype(np.int32)
dst_np = df['dst'].values.astype(np.int32)
t_np = df['time'].values.astype(np.int32)
edge_ids_np = np.arange(len(df), dtype=np.int32) + 1

test_src = test_df['src'].values.astype(np.int32)
test_time = test_df['time'].values.astype(np.int32)
test_candidates = test_df.iloc[:, 2:].values.astype(np.int32)

log(f'Train+Val: {len(df)}, Test: {len(test_df)}')

# Split (same 85/15 as main.py)
num_total = len(df)
num_val = int(num_total * 0.15)
num_train = num_total - num_val

train_data = TemporalData(
    src=jt.Var(src_np[:num_train]),
    dst=jt.Var(dst_np[:num_train]),
    t=jt.Var(t_np[:num_train]),
    edge_ids=jt.Var(edge_ids_np[:num_train])
)
val_data = TemporalData(
    src=jt.Var(src_np[num_train:]),
    dst=jt.Var(dst_np[num_train:]),
    t=jt.Var(t_np[num_train:]),
    edge_ids=jt.Var(edge_ids_np[num_train:])
)
full_data = TemporalData(
    src=jt.Var(src_np),
    dst=jt.Var(dst_np),
    t=jt.Var(t_np),
    edge_ids=jt.Var(edge_ids_np)
)

train_loader = TemporalDataLoader(train_data, batch_size=args.batch_size, neg_sampling_ratio=1.0)
val_loader = TemporalDataLoader(val_data, batch_size=args.batch_size, neg_sampling_ratio=1.0)

# Manual data-parallel sharding
if IS_MPI:
    n_keep = (len(train_loader.arange) // WORLD_SIZE) * WORLD_SIZE
    train_loader.arange = train_loader.arange[:n_keep][RANK::WORLD_SIZE]
    val_loader.arange = val_loader.arange[RANK::WORLD_SIZE]
    log(f'[MPI] Train batches/rank: {len(train_loader.arange)}, '
        f'Val batches/rank: {len(val_loader.arange)} '
        f'(global per-step batch = {args.batch_size * WORLD_SIZE})')

# neighbor sampler is built from FULL data so test queries can see all of train+val history
full_neighbor_sampler = get_neighbor_sampler(full_data, 'recent', seed=1)

# Determine sizes (must include test candidates so embedding table covers them)
max_node = max(int(src_np.max()), int(dst_np.max()), int(test_candidates.max()))
node_size = max_node + 1
num_edges = len(df)
log(f'Node size: {node_size}, Num edges: {num_edges}')

node_raw_features = np.zeros((node_size, args.node_feat_dim), dtype=np.float32)
edge_raw_features = np.zeros((num_edges + 1, args.edge_feat_dim), dtype=np.float32)

is_bipartite = len(set(src_np.tolist()) & set(dst_np.tolist())) == 0
log(f'Bipartite: {is_bipartite}')

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
    hidden_dim=args.node_feat_dim, output_dim=1)
model = nn.Sequential(backbone, predictor)

if IS_MPI:
    model.mpi_param_broadcast(root=0)

optimizer = jt.nn.Adam(list(model.parameters()), lr=args.lr)
criterion = jt.nn.BCEWithLogitsLoss()

if not args.eval_only:
    # Train
    log(f'\nTraining for {args.epochs} epoch(s) with early stopping (patience={args.early_stop})...')
    best_ap = train(model, optimizer, criterion, train_loader, val_loader,
                    args.epochs, save_path, args.dataset, args.early_stop)
    # Explicit barrier before any rank reads saved files
    if IS_MPI:
        jt.array([0], dtype='int32').mpi_all_reduce('add')
else:
    log('\n--eval_only is set, skipping training.')

log('\nGenerating predictions using best model...')
# If --model_path is given, use it directly; otherwise fall back to best/latest auto-search.
if args.model_path is not None:
    model.load_state_dict(jt.load(args.model_path))
    log(f'Loaded explicit model from {args.model_path}')
else:
    best_model_path = f'{save_path}/{args.dataset}_DyGFormer_best.pkl'
    latest_model_path = f'{save_path}/{args.dataset}_DyGFormer.pkl'
    if os.path.exists(best_model_path):
        model.load_state_dict(jt.load(best_model_path))
        log(f'Loaded best model from {best_model_path}')
    else:
        model.load_state_dict(jt.load(latest_model_path))
        log(f'Best model not found, using latest model')

# Parallel test inference
scores = test_competition(model, test_src, test_time, test_candidates,
                           save_path, args.dataset, args.test_batch_size)

if IS_MAIN:
    log(f'Scores shape: {scores.shape}, range: [{scores.min():.6f}, {scores.max():.6f}]')
    output_file = f'{args.output_dir}/{args.dataset}/{args.dataset}_result.csv'
    os.makedirs(osp.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        for row in scores:
            f.write(','.join([f'{p:.8f}' for p in row]) + '\n')

log('\n' + '=' * 80)
log('DONE')
log('=' * 80)
