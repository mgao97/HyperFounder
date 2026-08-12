# Dataset statistics — `neg_sam_v2` experimental design

最后更新: 2026-08-12,基于 DHG 0.9.5 + 自动 load 验证。

## 完整对照表

| # | Dataset | Domain | Task | Cache | Nodes | Edges | Avg edge size | Feat dim | Classes | Splits |
|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| 1 | `cora` | citation | node_cls | ❌ | 2,708 | 1,029 | 2.6 | 1,433 | 7 | Planetoid 60/20/20 |
| 2 | `citeseer` | citation | node_cls | ❌ | 3,327 | 4,528 | 1.4 | 3,703 | 6 | Planetoid 60/20/20 |
| 3 | `pubmed` | citation | node_cls | ❌ | 19,717 | 44,338 | 2.2 | 500 | 3 | Planetoid 60/20/20 |
| 4 | `cora_cc` | citation | node_cls | ✅ | **2,708** | **1,579** | **3.03** | **128** | **7** | official |
| 5 | `citeseer_cc` | citation | node_cls | ✅ | **3,312** | **1,079** | **3.20** | **128** | **6** | official |
| 6 | `pubmed_cc` | citation | node_cls | ❌ | 19,717 | 7,963 | 2.5 | 500 | 3 | official |
| 7 | `coauthorship_cora` | academic | node_cls | ✅ | **2,708** | **1,072** | **4.28** | **128** | **7** | official |
| 8 | `coauthorship_dblp` | academic | node_cls | ❌ | 42,564 | 89,790 | 2.1 | 8 | 6 | official |
| 9 | `dblp_8k` | academic | node_cls | ❌ | ~8,000 | ~ | ~ | ~ | ~ | official |
| 10 | `imdb_4k` | academic | node_cls | ❌ | ~4,000 | ~ | ~ | ~ | ~ | official |
| 11 | `cooking_200` | document | node_cls | ✅ | **7,403** | **2,755** | **19.96** | **128** | **20** | official |
| 12 | `news20` | document | node_cls | ❌ | 15,962 | 624 | 25.6 | ~800 | 20 | official |
| 13 | `gowalla` | recommendation | rec | ❌ | 40,182 | 1,027,370 | 25.6 | 0 | 0 | rec split |
| 14 | `yelp_2018` | recommendation | rec | ❌ | 91,270 | 1,237,439 | 13.6 | 0 | 0 | rec split |
| 15 | `tencent_2k` | recommendation | rec | ❌ | 1,000 | ~ | ~ | 0 | 0 | rec split |
| 16 | `movielens_1m` | recommendation | rec | ✅ | **3,043** | **6,022** | **132.22** | **128** | **0** | rec split |
| 17 | `yelp_restaurant` | recommendation | node_cls | ✅ (cache) ❌ (registered) | 2,584 | 2,582 | 1.0 | ~ | 10 | n/a |
| 18 | `house_committees` | political | graph_candidate | ✅ | **1,290** | **341** | **34.79** | **128** | **3** | n/a |
| 19 | `walmart_trips` | commerce | graph_candidate | ❌ | ~ | ~ | ~ | 0 | 0 | n/a |
| 20 | `github` | social/code | node_cls | ❌ | 37,700 | 578,000 | ~ | 0 | 2 | n/a |
| 21 | `facebook` | social | node_cls | ❌ | 4,039 | 88,234 | ~ | 0 | n/a | n/a |
| 22 | `flickr` | social | node_cls | ❌ | 7,575 | 239,738 | ~ | 0 | n/a | n/a |
| 23 | `amazon_book` | e-commerce | rec | ❌ | ~ | ~ | ~ | 0 | 0 | rec split |
| 24 | `recipe_100k` | recipe | graph_candidate | ❌ (DHG 有) | ~100K graphs | n/a | ~ | 0 | n/a | n/a |
| 25 | `recipe_200k` | recipe | graph_candidate | ❌ (DHG 有) | ~200K graphs | n/a | ~ | 0 | n/a | n/a |

**加粗数字** = 实际 load 验证,其余为 DHG 官方文档 / DHG 论文里的数据。

## 按域汇总(用于决策)

### Citation(5 个候选, 3 个 cache 有)

| Dataset | Nodes | Edges | Avg edge size | Classes | 备注 |
|---|---:|---:|---:|---:|---|
| cora_cc | 2,708 | 1,579 | 3.03 | 7 | ✅ cache |
| citeseer_cc | 3,312 | 1,079 | 3.20 | 6 | ✅ cache |
| pubmed_cc | 19,717 | 7,963 | 2.5 | 3 | ❌ 需要下 |

→ **3 个可用的 citation 数据集**,规模 2.7k → 19k,加在一起 25k 节点。

### Academic(4 个候选, 1 个 cache 有)

| Dataset | Nodes | Edges | Avg edge size | Classes | 备注 |
|---|---:|---:|---:|---:|---|
| coauthorship_cora | 2,708 | 1,072 | 4.28 | 7 | ✅ cache |
| coauthorship_dblp | 42,564 | 89,790 | 2.1 | 6 | ❌ 需要下 |
| dblp_8k | ~8,000 | – | – | – | ❌ DHG API 有 bug |
| imdb_4k | ~4,000 | – | – | – | ❌ 需要下 |

→ **coauthorship_cora 已 cache**,dblp 缺,优先下这个(规模最大)。

### Document(2 个候选, 1 个 cache 有)

| Dataset | Nodes | Edges | Avg edge size | Classes | 备注 |
|---|---:|---:|---:|---:|---|
| cooking_200 | 7,403 | 2,755 | 19.96 | 20 | ✅ cache |
| news20 | 15,962 | 624 | 25.6 | 20 | ❌ 需要下 |

→ **cooking_200 已 cache**,news20 缺。

### Recommendation(5 个候选, 1 个 cache 有)

| Dataset | Nodes | Edges | Avg edge size | 备注 |
|---|---:|---:|---:|---|
| gowalla | 40,182 | 1,027,370 | 25.6 | ❌ 大图,需要下 |
| yelp_2018 | 91,270 | 1,237,439 | 13.6 | ❌ 大图,需要下 |
| tencent_2k | 1,000 | – | – | ❌ 小,可以下 |
| movielens_1m | 3,043 | 6,022 | 132.22 | ✅ cache(但官方 task=rec,不是 node_cls) |
| yelp_restaurant | 2,584 | 2,582 | 1.0 | ⚠️ cache 有但没注册 |

⚠️ rec 数据集都是大图 + bipartite,**跟节点分类任务不直接对应**。如果 pretrain 里加 rec 域,需要专门处理。

### 政治 / 商业(2 个候选, 1 个 cache 有)

| Dataset | Nodes | Edges | Avg edge size | Classes | 备注 |
|---|---:|---:|---:|---:|---|
| house_committees | 1,290 | 341 | 34.79 | 3 | ✅ cache |
| walmart_trips | – | – | – | – | ❌ 需要下 |

→ **house_committees 是典型 OOD 测试**。建议留作 OOD 测试。

## 几个关键观察

### 1. 不同域的"图性质"差异巨大

| 域 | 典型 Avg edge size | 典型 Nodes/Edges 比 | 含义 |
|---|---:|---:|---|
| citation (cocitation) | 3 | 1-2 | 每条超边 ~3 节点,稀疏 |
| academic (coauthorship) | 4 | 2-3 | 合著者 ~4 人,中等密度 |
| document (recipe/news) | 20 | 1.5-12 | 每条超边 ~20 词,密集 |
| recommendation | 25-130 | 13-25 | bipartite,边很稠密 |
| political (committee) | 35 | 4 | 委员会 ~35 个议员 |

→ **同一个 encoder 要处理从"3 节点/边"到"130 节点/边"的所有情况**,domain adapter 必要。

### 2. Class 数差异也很大

- 最小:house_committees(3 类)、pubmed(3 类)
- 最大:cooking_200 / news20(20 类)

→ **pretrain 阶段 domain_classifier 要支持多类别**。

### 3. Feat dim 差异

- citation:128(cocitation) → 1,433(cora 原版)→ 3,703(citeseer 原版)
- recommendation:0(没有节点特征,纯图结构)

→ **域感知 projector 必须能处理"无特征"的情况**。

## 决策建议(基于这张表)

| 决策点 | 推荐 | 原因 |
|---|---|---|
| 预训练数据集数 | 8 个(citation×3, academic×1, document×2, rec×2) | 规模 / 多样性平衡 |
| 必下 cache | `pubmed_cc`, `coauthorship_dblp`, `news20` | 凑齐每个 domain ≥2 实例 |
| 必留 OOD | `house_committees`, `walmart_trips`, `movielens_1m` | 跨域迁移证据 |
| gowalla / yelp_2018 | **进 pretrain**(规模大,rec 域代表性) | rec 域需要规模 |
| 是否做 graph-level 任务 | 不做(图分类 baseline 太少) | 审稿人关注度低 |

## 行动清单

按这张表去操作:

```bash
# 1. 补注册缺失的 datasets(可选)
# yelp_restaurant, recipe_*, github, facebook, flickr

# 2. 下载缺失的 cache
# 用 dhg 一次性下载,然后会出现在 data/cache/

# 3. 写最终 pretrain config
# configs/pretrain_neg_sam_v2_final.yaml

# 4. 写评测 config
# configs/finetune_node_v2_in_domain.yaml
# configs/finetune_node_v2_ood.yaml
```

要不要我现在直接生成最终的两个 yaml?
