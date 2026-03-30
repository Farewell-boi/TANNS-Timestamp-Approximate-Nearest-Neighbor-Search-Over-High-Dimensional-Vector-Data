# TANNS: Timestamp Approximate Nearest Neighbor Search

Python 完整复现论文：

> **Timestamp Approximate Nearest Neighbor Search over High-Dimensional Vector Data**  
> Yuxiang Wang, Ziyuan He, Yongxin Tong, Zimu Zhou, Yiman Zhong  
> IEEE ICDE 2025

---

## 核心贡献

| 方法 | 搜索时间 | 更新时间 | 空间 |
|------|--------|--------|------|
| Naive Graph-based TANNS (Sec. III-B) | O(M log N) | O(MN log N) | O(MN²) |
| **Timestamp Graph** (Sec. IV) | O(log²N) | O(log²N) | O(M²N) |
| **Compressed Timestamp Graph** (Sec. V) | O(log²N) | O(log²N) | **O(MN)** |

---

## 项目结构

```
TANNS/
├── tanns/                          # 核心算法库
│   ├── __init__.py
│   ├── data_types.py               # Vector, TANNSQuery 数据结构
│   ├── distance.py                 # 欧式距离、余弦距离
│   ├── hnsw.py                     # HNSW 图索引（Algorithm 1 & 2）
│   ├── timestamp_graph.py          # 时间戳图（Algorithm 3 & 4，Sec. IV）
│   ├── historic_neighbor_tree.py   # 历史邻居树（Algorithm 5 & 6，Sec. V）
│   └── compressed_timestamp_graph.py  # 压缩时间戳图（完整 Sec. V）
│
├── experiments/                    # 实验代码
│   ├── data_generator.py           # 合成数据生成（Short/Long/Mixed/Uniform 模式）
│   ├── baselines.py                # 基准方法（Pre-Filtering, Post-Filtering HNSW）
│   └── benchmark.py                # 完整基准测试脚本
│
├── tests/
│   └── test_tanns.py               # 单元测试
│
├── demo.py                         # 快速演示脚本
├── requirements.txt
└── README.md
```

---

## 快速开始

### 安装依赖

```bash
pip install numpy
```

### 运行 Demo

```bash
cd d:/WorkBuddySpace/TANNS
python demo.py
```

### 运行单元测试

```bash
cd d:/WorkBuddySpace/TANNS
python -m pytest tests/ -v
```

### 运行基准测试

```bash
python experiments/benchmark.py --n 5000 --dim 64 --pattern uniform --k 10
```

参数说明：
- `--n`        数据集向量数量（默认 5000）
- `--dim`      向量维度（默认 64）
- `--pattern`  时间模式：`short` / `long` / `mixed` / `uniform`（默认 uniform）
- `--n_queries` 查询数量（默认 200）
- `--k`        返回近邻数（默认 10）
- `--metric`   距离度量：`euclidean` / `cosine`（默认 euclidean）
- `--M`        图邻居数（默认 16）
- `--M_prime`  候选邻居数（默认 200）

---

## 核心算法说明

### 1. TANNS 问题定义（Section II）

给定 N 个高维向量数据集 `D = {u₁, u₂, ..., uN}`，每个向量 `uᵢ` 关联时间戳范围 `[uᵢ.s, uᵢ.e)`（分别为生效和过期时间）。

查询 `TANNS(D, ts, q, k)` 在时间戳 `ts` 有效的向量中，返回与查询向量 `q` 最近的 k 个近似最近邻。

### 2. 时间戳图 Timestamp Graph（Section IV）

- 单一图索引管理所有历史时间戳的向量
- 每个节点维护：主邻居列表 `TG[u]`（M个）+ 备份邻居列表 `B[u]`（M个）
- 历史邻居列表按时间版本化记录，支持二分查找重建任意时间戳的邻居列表
- **点插入（Algorithm 3）**：类似 HNSW 插入，选 2M 候选，前 M 为主邻居，后 M 为备份
- **点过期（Algorithm 4）**：用备份邻居替换过期的主邻居，保持连通性

### 3. 历史邻居树 Historic Neighbor Tree（Section V）

- 受区间树启发的自底向上构建的平衡二叉树
- 每个节点存储在该层首次成为有效点的向量，每个点只存一处
- **邻居列表重建（Algorithm 5）**：O(log n + Mᵣ) 时间复杂度
- **追加邻居列表（Algorithm 6）**：动态维护，O(log n) 时间复杂度
- 空间复杂度 O(n)，将时间戳图压缩到单时间戳 HNSW 同级别

### 4. 压缩时间戳图 Compressed Timestamp Graph（Section V）

- 将 TG 中每个点的历史邻居列表替换为 HNT 存储
- 查询时通过 Algorithm 5 动态重建邻居列表
- 整体空间复杂度 O(MN)（最优）

---

## 对应论文算法映射

| 论文算法 | 实现位置 | 说明 |
|---------|---------|------|
| Algorithm 1: HNSW Search | `tanns/hnsw.py: HNSW._search_layer` | 带贪心路由的 HNSW 搜索 |
| Algorithm 2: HNSW Construction | `tanns/hnsw.py: HNSW.add` | 增量式 HNSW 构建 |
| Select-Nbrs (启发式邻居选择) | `tanns/hnsw.py: HNSW._select_neighbors` | 排除支配点的邻居选择 |
| Algorithm 3: Point Insertion | `tanns/timestamp_graph.py: TimestampGraph.insert` | 带备份邻居的点插入 |
| Algorithm 4: Point Expiration | `tanns/timestamp_graph.py: TimestampGraph.expire` | 点过期处理 |
| Algorithm 5: Neighbor List Reconstruction | `tanns/historic_neighbor_tree.py: HistoricNeighborTree.reconstruct` | HNT 重建邻居列表 |
| Algorithm 6: Append Neighbor List | `tanns/historic_neighbor_tree.py: HistoricNeighborTree.append` | HNT 动态更新 |

---

## 实验复现说明

论文实验在 C++ 上运行（Intel Xeon Gold 6240, 768GB RAM），使用 SIFT/GIST/DEEP/GloVe 数据集（各 100 万向量）。

本 Python 实现提供了完整的算法逻辑，适合：
- 理解算法原理
- 小规模验证正确性
- 作为大规模实现（C++）的参考

若需要大规模实验，建议：
1. 下载 [ann-benchmarks](https://github.com/erikbern/ann-benchmarks) 数据集
2. 参考本实现用 C++ 重写核心算法（HNSW + 时间戳图）
3. 使用 GCC -Ofast 优化，单线程测试

---

## 引用

```bibtex
@inproceedings{wang2025tanns,
  title={Timestamp Approximate Nearest Neighbor Search over High-Dimensional Vector Data},
  author={Wang, Yuxiang and He, Ziyuan and Tong, Yongxin and Zhou, Zimu and Zhong, Yiman},
  booktitle={IEEE 41st International Conference on Data Engineering (ICDE)},
  year={2025}
}
```
