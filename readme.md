# 一、快速开始

## 1.1 运行命令

```bash
conda activate rtdetr_env
python main.py --dataset dataset1

dataset1：Best AP: 0.975149
dataset2：Best AP: 0.935858
```

支持的数据集：`dataset1`、`dataset2`

## 1.2 输出文件

运行完成后，在 `./data/{dataset}/` 目录下生成 `{dataset}_result.csv`

---

# 二、A榜数据集简介

## 2.1 目录结构

```
./
├── main.py                    # 主程序入口
├── data/       
│   ├── dataset1/
│   │   ├── train.csv          # 训练数据
│   │   └── test.csv           # 测试数据
│   └── dataset2/
│       ├── train.csv
│       └── test.csv
```

## 2.2 数据格式

### 训练数据 (train.csv)

| 列名 | 类型 | 说明 |
|------|------|------|
| src | int | 源节点 ID |
| dst | int | 目标节点 ID |
| time | int | 时间戳 |

示例：
```csv
src,dst,time
862247,1,0
862248,53,3
862249,98,135528
```

### 测试数据 (test.csv)

| 列名 | 类型 | 说明 |
|------|------|------|
| src | int | 源节点 ID |
| time | int | 时间戳 |
| c1~c100 | int | 100 个候选目标节点 |

示例：
```csv
src,time,c1,c2,c3,...,c100
862300,150000,17,920,41,...,127
```

参赛选手需要对 100 个候选目标节点计算链接概率，输出到 `{dataset}_result.csv` 文件中。

## 2.3 数据集统计 (含测试边)

| 数据集   | 二部图 |                节点数 |      边数 | 重复率 |
| -------- | -----: | --------------------: | --------: | -----: |
| dataset1 |     否 |                23,852 |   751,899 |   0.87 |
| dataset2 |     是 | 63,348(12,708/50,640) | 2,414,703 |        |

## 2.4 提交格式

### 输出文件格式 ({dataset}_result.csv)

- 每行对应 test.csv 中的一个测试样本
- 每行包含 100 个概率值，逗号分隔
- 概率值顺序与 test.csv 中 c1~c100 的顺序一一对应
- 概率值保留 8 位小数

示例：
```csv
0.12345678,0.23456789,0.34567890,...,0.98765432
0.98765432,0.87654321,0.76543210,...,0.11111111
```

### 打包提交

```bash
zip result.zip \
    dataset1.csv \
    dataset2.csv
```

## 2.5 评估指标

使用 **MRR (Mean Reciprocal Rank)** 评估：

```
MRR = (1/N) × Σ(1/rank_i)
```

其中 `rank_i` 是第 i 个测试样本中正样本按概率降序排列后的排名位置。
