'''
优化版：基于 GCN 的 Cora 节点分类任务

主要优化点：
  1. dropout 0.8 → 0.5（原论文值，避免欠拟合）
  2. 保存验证集最优模型参数，预测时加载最优而非最后
  3. Early stopping（连续50轮val_acc不涨则停）
  4. 训练/验证解耦，test()只在需要时调用
  5. hidden_dim 256 → 64（Cora上够用，训练更快）
  6. 训练轮数 200 → 500（配合early stopping不会真的跑满）
  7. weight_decay 5e-4 保持不变（GCN论文原值，已合适）
'''

import os.path as osp
import json
import pickle
import copy

import jittor as jt
from jittor import nn
import numpy as np
from jittor_geometric.nn import GCNConv
from jittor_geometric.ops import cootocsr, cootocsc
from jittor_geometric.nn.conv.gcn_conv import gcn_norm


# ============================================================
# 基本配置
# ============================================================
jt.flags.use_cuda = 1
jt.misc.set_global_seed(42)

# ============================================================
# 第一步：加载数据集
# ============================================================
data_path = osp.join('data', 'cora.pkl')

with open(data_path, 'rb') as f:
    raw = pickle.load(f)


class GraphData:
    pass


data = GraphData()
data.x           = jt.array(raw['x'].astype(np.float32))
data.y           = jt.array(raw['y'].astype(np.int64))
data.edge_index  = jt.array(raw['edge_index'].astype(np.int64))
data.train_mask  = jt.array(raw['train_mask'])
data.val_mask    = jt.array(raw['val_mask'])
data.test_mask   = jt.array(raw['test_mask'])
num_features     = raw['num_features']
num_classes      = raw['num_classes']

# 行归一化特征
row_sum  = data.x.sum(dim=1, keepdims=True)
row_sum  = jt.clamp(row_sum, min_v=1e-12)
data.x   = data.x / row_sum

# ============================================================
# 第二步：图的边归一化 + 稀疏格式转换
# ============================================================
v_num = data.x.shape[0]
edge_index, edge_weight = data.edge_index, None

edge_index, edge_weight = gcn_norm(
    edge_index, edge_weight, v_num,
    improved=False, add_self_loops=True
)

with jt.no_grad():
    data.csc = cootocsc(edge_index, edge_weight, v_num)
    data.csr = cootocsr(edge_index, edge_weight, v_num)

# ============================================================
# 第三步：定义 GCN 模型
# ============================================================
class GCNNet(nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim=64, dropout=0.5):
        super(GCNNet, self).__init__()
        # [优化] dropout 0.8→0.5，hidden_dim 256→64
        self.dropout = dropout

        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)

    def execute(self):
        x, csc, csr = data.x, data.csc, data.csr

        x = nn.relu(self.conv1(x, csc, csr))
        x = nn.dropout(x, self.dropout, is_train=self.training)
        x = self.conv2(x, csc, csr)

        return x


model = GCNNet(
    num_features=num_features,
    num_classes=num_classes,
    hidden_dim=64,    # [优化] 256→64，Cora上64已足够
    dropout=0.5,      # [优化] 0.8→0.5
)
optimizer = nn.Adam(
    params=model.parameters(),
    lr=0.01,
    weight_decay=5e-4  # 保持原值
)

# ============================================================
# 第四步：训练函数（只做前向+反向，不做eval）
# ============================================================
def train():
    model.train()
    pred  = model()[data.train_mask]
    label = data.y[data.train_mask]
    loss  = nn.cross_entropy_loss(pred, label)
    optimizer.step(loss)
    return loss.item()


# ============================================================
# 第五步：评估函数（解耦，只评估指定mask的节点）
# ============================================================
def evaluate(mask):
    '''返回 mask 对应节点的分类准确率'''
    model.eval()
    with jt.no_grad():
        logits = model()
    pred, _ = jt.argmax(logits[mask], dim=1)
    acc = pred.equal(data.y[mask]).sum().item() / mask.sum().item()
    return acc


# ============================================================
# 第六步：训练循环 + Early Stopping + 最优模型保存
# ============================================================
best_val_acc    = 0.0
best_model_state = None          # [优化] 保存最优参数
patience        = 50             # [优化] 连续50轮不涨就停
no_improve      = 0
max_epochs      = 500            # [优化] 200→500，配合early stopping

for epoch in range(1, max_epochs + 1):
    loss = train()

    # [优化] 每轮都计算val_acc用于early stopping判断
    val_acc = evaluate(data.val_mask)

    if val_acc > best_val_acc:
        best_val_acc     = val_acc
        best_model_state = copy.deepcopy(model.state_dict())  # [优化] 保存最优
        no_improve       = 0
    else:
        no_improve += 1

    if epoch % 20 == 0:
        train_acc = evaluate(data.train_mask)
        print(f'Epoch: {epoch:03d} | Loss: {loss:.4f} | '
              f'Train: {train_acc:.4f} | Val: {val_acc:.4f} | '
              f'Best Val: {best_val_acc:.4f}')

    # [优化] Early stopping
    if no_improve >= patience:
        print(f'\nEarly stopping triggered at epoch {epoch}')
        break

print(f'\n最终结果: Best Val Acc: {best_val_acc:.4f}')

# ============================================================
# 第七步：加载最优模型，生成预测结果
# ============================================================
# [优化] 加载验证集最优参数，而非最后一个epoch的参数
if best_model_state is not None:
    model.load_state_dict(best_model_state)

model.eval()
with jt.no_grad():
    logits = model()

pred, _ = jt.argmax(logits, dim=1)

test_indices = np.nonzero(raw['test_mask'])[0]

result = {}
for idx in test_indices:
    result[str(int(idx))] = int(pred[int(idx)])

output_path = 'result.json'
with open(output_path, 'w') as f:
    json.dump(result, f, indent=2)

print(f'\n预测结果已保存到 {output_path}')
print(f'共预测 {len(result)} 个测试节点')