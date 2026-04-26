import os
import os.path as osp
import sys

# Add JittorGeometric to path
root = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, root)

os.environ['JT_SYNC'] = '1'

import jittor as jt
from jittor import nn
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


# Used for validation
def test_val(model, loader):
    model.eval()
    backbone, predictor = model[0], model[1]
    mrr_eval = MRR_Evaluator()
    ap_list, auc_list, mrr_list = [], [], []
    loader_tqdm = tqdm(loader, ncols=120, desc='Validation')
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
        ap_list.append(average_precision_score(y_true, y_score))
        auc_list.append(roc_auc_score(y_true, y_score))
        mrr_list.extend(mrr_eval.eval(pos_score, neg_score))

    return {'AP': np.mean(ap_list), 'AUC': np.mean(auc_list), 'MRR': np.mean(mrr_list)}


def train(model, optimizer, criterion, train_loader, val_loader,
          num_epochs, save_path, dataset_name, early_stop_patience=10):
    best_ap = 0
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        backbone, predictor = model[0], model[1]
        train_losses = []
        train_tqdm = tqdm(train_loader, ncols=120, desc=f'Epoch {epoch + 1}')

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

        print(f'Epoch {epoch + 1}, Train Loss: {np.mean(train_losses):.4f}')
        val_res = test_val(model, val_loader)
        print(f'Epoch {epoch + 1}, Val: {val_res}')

        # Early stopping check (use AP, matches main.py)
        current_ap = val_res['AP']
        if current_ap > best_ap:
            best_ap = current_ap
            patience_counter = 0
            jt.save(model.state_dict(), f'{save_path}/{dataset_name}_DyGFormer_best.pkl')
            print(f'  -> New best AP: {best_ap:.6f}, model saved!')
        else:
            patience_counter += 1
            print(f'  -> No improvement for {patience_counter} epoch(s), best AP: {best_ap:.6f}')

        # Also save latest model
        jt.save(model.state_dict(), f'{save_path}/{dataset_name}_DyGFormer.pkl')

        if patience_counter >= early_stop_patience:
            print(f'\nEarly stopping triggered after {epoch + 1} epochs!')
            print(f'Best validation AP: {best_ap:.6f}')
            break

    return best_ap


# Used for generate test scores
def test_competition(model, test_src, test_time, test_candidates, batch_size=25):
    """
    For each query (src_i, t_i, [c_1..c_100]), score 100 candidates by expanding to
    B*100 (src, dst, t) triples and running the model in one pass per chunk.
    """
    model.eval()
    backbone, predictor = model[0], model[1]
    all_scores = []
    num_samples = len(test_src)
    num_batches = (num_samples + batch_size - 1) // batch_size
    num_cands = test_candidates.shape[1]

    pbar = tqdm(range(num_batches), ncols=120, desc='Testing')
    for batch_idx in pbar:
        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, num_samples)

        b = end - start
        # repeat src and time for each of the 100 candidates
        src_rep = np.repeat(test_src[start:end], num_cands).astype(np.int32)
        t_rep = np.repeat(test_time[start:end].astype(np.float32), num_cands)
        cand_flat = test_candidates[start:end].reshape(-1).astype(np.int32)

        src_emb, dst_emb = backbone.compute_src_dst_node_temporal_embeddings(
            src_node_ids=src_rep, dst_node_ids=cand_flat, node_interact_times=t_rep)
        logit = predictor(src_emb, dst_emb).squeeze(-1)
        probs = jt.sigmoid(logit).numpy().reshape(b, num_cands)
        all_scores.append(probs)

    return np.vstack(all_scores)


# Main
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
parser.add_argument('--save_dir', type=str, default='./saved_models', help='Model save directory')
parser.add_argument('--output_dir', type=str, default=None, help='Output directory (default: same as data_dir)')
parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=200, help='Batch size for train/val')
parser.add_argument('--test_batch_size', type=int, default=25,
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
args = parser.parse_args()

if args.output_dir is None:
    args.output_dir = args.data_dir

print('=' * 80)
print(f'DyGFormer Competition - Dataset: {args.dataset}')
print('=' * 80)

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

print(f'Train+Val: {len(df)}, Test: {len(test_df)}')

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

# neighbor sampler is built from FULL data so test queries can see all of train+val history
full_neighbor_sampler = get_neighbor_sampler(full_data, 'recent', seed=1)

# Determine sizes (must include test candidates so embedding table covers them)
max_node = max(int(src_np.max()), int(dst_np.max()), int(test_candidates.max()))
node_size = max_node + 1
num_edges = len(df)
print(f'Node size: {node_size}, Num edges: {num_edges}')

# DyGFormer requires node_raw_features and edge_raw_features (zeros works; co-occurrence
# + time encoder carry the signal, matching the upstream JODIE example)
node_raw_features = np.zeros((node_size, args.node_feat_dim), dtype=np.float32)
edge_raw_features = np.zeros((num_edges + 1, args.edge_feat_dim), dtype=np.float32)

# Detect bipartite: src/dst sets disjoint -> True
is_bipartite = len(set(src_np.tolist()) & set(dst_np.tolist())) == 0
print(f'Bipartite: {is_bipartite}')

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

optimizer = jt.nn.Adam(list(model.parameters()), lr=args.lr)
criterion = jt.nn.BCEWithLogitsLoss()

save_path = args.save_dir
os.makedirs(save_path, exist_ok=True)

# Train
print(f'\nTraining for {args.epochs} epoch(s) with early stopping (patience={args.early_stop})...')
best_ap = train(model, optimizer, criterion, train_loader, val_loader,
                args.epochs, save_path, args.dataset, args.early_stop)

# Load best model
print('\nGenerating predictions using best model...')
best_model_path = f'{save_path}/{args.dataset}_DyGFormer_best.pkl'
if os.path.exists(best_model_path):
    model.load_state_dict(jt.load(best_model_path))
    print(f'Loaded best model from {best_model_path}')
else:
    model.load_state_dict(jt.load(f'{save_path}/{args.dataset}_DyGFormer.pkl'))
    print(f'Best model not found, using latest model')

# Generate scores for test.csv
scores = test_competition(model, test_src, test_time, test_candidates, args.test_batch_size)

print(f'Scores shape: {scores.shape}, range: [{scores.min():.6f}, {scores.max():.6f}]')

# Save (same format as main.py output)
output_file = f'{args.output_dir}/{args.dataset}/{args.dataset}_result.csv'
os.makedirs(osp.dirname(output_file), exist_ok=True)
with open(output_file, 'w') as f:
    for row in scores:
        f.write(','.join([f'{p:.8f}' for p in row]) + '\n')

print('\n' + '=' * 80)
print('DONE')
print('=' * 80)
