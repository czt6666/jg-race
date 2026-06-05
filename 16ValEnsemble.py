"""
16ValEnsemble.py - 在验证集(split1)上评测单模型 vs 集成的 100-way MRR
只用 split0 训练的模型(可在split1公平验证)。固定99负样本保证可比。
"""
import os, sys, os.path as osp
root = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, root)
os.environ["JT_SYNC"] = "1"
import jittor as jt
import numpy as np
import pandas as pd
from jittor_geometric.nn.models.sasrec import SASRec
jt.flags.use_cuda = 1

DATA = "./data/dataset2"
SEQ = 64
VAL_NEG = 99

df = pd.read_csv(f"{DATA}/train.csv")
train_df = df[df["split"] == 0].reset_index(drop=True)
val_df = df[df["split"] == 1].reset_index(drop=True)
test_df = pd.read_csv(f"{DATA}/test.csv")
max_item = max(int(df["dst"].max()), int(test_df.iloc[:, 2:].values.max()))

# 用户历史(全量df,与训练时val一致)
def build_hist(s, d, t):
    raw = {}
    for a, b, c in zip(s, d, t):
        raw.setdefault(int(a), ([], []))
        raw[int(a)][0].append(float(c)); raw[int(a)][1].append(int(b))
    out = {}
    for u, (ts, it) in raw.items():
        ts = np.array(ts, np.float32); it = np.array(it, np.int32)
        o = np.argsort(ts, kind="stable")
        out[u] = (ts[o], it[o])
    return out

hist = build_hist(df["src"].values, df["dst"].values, df["time"].values)

def seqs(src, t):
    B = len(src)
    iseq = np.zeros((B, SEQ), np.int32); ilen = np.ones(B, np.int32)
    for i in range(B):
        u = int(src[i]); h = hist.get(u)
        if h is None: continue
        idx = int(np.searchsorted(h[0], float(t[i]), "left"))
        if idx == 0: continue
        items = h[1][max(0, idx-SEQ):idx]; n = len(items)
        iseq[i, :n] = items; ilen[i] = n
    return iseq, ilen

# 固定负样本
rng = np.random.RandomState(2024)
vsrc = val_df["src"].values.astype(np.int32)
vdst = val_df["dst"].values.astype(np.int32)
vt = val_df["time"].values.astype(np.float32)
N = len(vsrc)
vneg = rng.randint(1, max_item+1, size=(N, VAL_NEG)).astype(np.int32)

def load_model(path):
    m = SASRec(n_layers=2, n_heads=2, hidden_size=128, inner_size=256,
               hidden_dropout_prob=0.1, attn_dropout_prob=0.1, hidden_act='gelu',
               layer_norm_eps=1e-12, initializer_range=0.02,
               n_items=max_item, max_seq_length=SEQ)
    m.set_min_idx(0, 0); m.load_state_dict(jt.load(path)); m.eval()
    return m

def model_scores(m):
    """返回 pos[N], neg[N,99] (sigmoid分数)"""
    pos = np.zeros(N, np.float32); neg = np.zeros((N, VAL_NEG), np.float32)
    BS = 512
    with jt.no_grad():
        for st in range(0, N, BS):
            en = min(st+BS, N); b = en-st
            iseq, ilen = seqs(vsrc[st:en], vt[st:en])
            ur = m.forward(jt.Var(iseq).int32(), jt.Var(ilen).int32())  # [b,H]
            pe = m.item_embedding(jt.Var(vdst[st:en]).int32())
            pos[st:en] = jt.sigmoid((ur*pe).sum(-1)).numpy()
            for c in range(0, VAL_NEG, 33):
                ce = min(c+33, VAL_NEG); cs = ce-c
                ch = vneg[st:en, c:ce]
                urr = ur.unsqueeze(1).expand(-1, cs, -1).reshape(b*cs, -1)
                ne = m.item_embedding(jt.Var(ch.reshape(-1)).int32())
                neg[st:en, c:ce] = jt.sigmoid((urr*ne).sum(-1)).numpy().reshape(b, cs)
            jt.sync_all()
    return pos, neg

def mrr_of(pos, neg):
    ranks = 1 + np.sum(neg > pos.reshape(-1, 1), axis=1)
    return float(np.mean(1.0/ranks))

models = {
    "epoch5(0.5195)": "saved_models/run_0604/13sasrec_wd_val/dataset2_SASRec_best.pkl",
    "finetune(0.5244)": "saved_models/run_0604/13sasrec_wd_finetune_val/dataset2_SASRec_best.pkl",
    "finetune2(0.5250)": "saved_models/run_0604/13sasrec_wd_finetune2_val/dataset2_SASRec_best.pkl",
}

all_pos, all_neg = {}, {}
for name, path in models.items():
    if not osp.exists(path):
        print(f"SKIP {name}: 文件不存在"); continue
    m = load_model(path)
    p, ng = model_scores(m)
    all_pos[name] = p; all_neg[name] = ng
    print(f"{name}: 单模型 val MRR = {mrr_of(p, ng):.4f}", flush=True)
    del m; jt.gc()

names = list(all_pos.keys())

def ens_rank(name_subset):
    # rank-average: 对每行[pos]+[99neg]=100个候选排名平均
    accum = np.zeros((N, 100), np.float64)
    for nm in name_subset:
        allc = np.concatenate([all_pos[nm].reshape(-1,1), all_neg[nm]], axis=1)  # [N,100]
        order = np.argsort(-allc, axis=1)
        rank = np.empty_like(order); rows = np.arange(N)[:,None]
        rank[rows, order] = np.arange(1,101)[None,:]
        accum += rank
    accum /= len(name_subset)
    # pos在第0列，排名=accum[:,0]
    pos_rank = accum[:, 0]
    return float(np.mean(1.0/pos_rank))

print("\n=== 集成结果(rank-average) ===")
if len(names) >= 2:
    print(f"全部3模型集成: {ens_rank(names):.4f}")
    print(f"finetune+finetune2: {ens_rank(names[1:]):.4f}")
    print(f"epoch5+finetune2: {ens_rank([names[0], names[2]]):.4f}")
print("\n单模型最佳作对比基准。集成>最佳则集成有效。")
