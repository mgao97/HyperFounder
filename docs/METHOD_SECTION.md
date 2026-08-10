# Method Section — `neg_sam_v2` Pretraining

英文 draft,匹配 GFSE(WWW'26)的写作风格:**每个子任务一节 + 一个公式**,下游整合独立成节。
每节都标注了对应代码模块,方便核对。

---

## 3. Method

### 3.1 Overview and Notation

Let $\mathcal{H} = (\mathcal{V}, \mathcal{E})$ denote a hypergraph with node set $\mathcal{V}$ and hyperedge set $\mathcal{E}$, and let $\mathbf{B} \in \{0,1\}^{|\mathcal{V}| \times |\mathcal{E}|}$ be its incidence matrix, where $\mathbf{B}_{v,e} = 1$ iff node $v$ belongs to edge $e$. Each node $v \in \mathcal{V}$ carries a feature vector $\mathbf{x}_v \in \mathbb{R}^{d_{\text{in}}}$.

We pretrain a single encoder across $K$ source domains $\{\mathcal{D}^{(k)}\}_{k=1}^{K}$ (e.g., citation / academic / document / recommendation). At each step, a hyperedge-centred subhypergraph (§3.7) is sampled from one domain, encoded by a dual-token transformer (§3.2), and used to compute eight self-supervised objectives (§3.4). Two auxiliary mechanisms — **shared–private disentanglement with multi-granularity alignment** (§3.5) and **quality-aware negative sampling** (§3.6) — encourage the encoder to learn cross-domain transferable structural features without forgetting domain-specific statistics.

The total loss is a weighted sum of all task losses:
$$\mathcal{L}_{\text{total}} = \sum_{t \in \mathcal{T}} w_t \, \mathcal{L}_t,$$
where the weights $\{w_t\}$ are given in Table 2.

(*对应:`docs/NEG_SAM_V2_REVIEW.md` 总览 + `configs/pretrain_neg_sam_v2.yaml::loss_weights`*)

---

### 3.2 Dual-Token Hypergraph Encoder

(*对应:`models/encoder.py::UnifiedHypergraphEncoder` + `models/cross_domain_modules.py`*)

#### 3.2.1 Initial Node and Hyperedge Features
We maintain two parallel token streams in a shared $d$-dimensional space:
$$\mathbf{N}^{(0)} = \mathbf{W}_{\text{proj}}^{(k)}(\mathbf{X}) + \text{PE}_{\text{node}}(\mathbf{B}), \quad \mathbf{E}^{(0)} = \mathbf{W}_{\text{proj}}^{(k)}(\mathbf{0}) + \text{PE}_{\text{edge}}(\mathbf{B}),$$
where $k$ indexes the source domain and $\mathbf{W}_{\text{proj}}^{(k)}$ is a per-domain projector registered to handle heterogeneous feature types (numerical / textual / categorical). The zero input for $\mathbf{E}^{(0)}$ reflects that hyperedge features are constructed structurally rather than observed.

#### 3.2.2 Structural Positional Encoding via Hyperedge Random Walks
Because hyperedges have variable cardinality, we use a **random-walk-based structural PE** over the node–hyperedge bipartite graph rather than Laplacian eigenvectors. Let $\mathbf{M} = \mathbf{D}_v^{-1}\mathbf{B}\,\mathbf{D}_e^{-1}\mathbf{B}^\top \in \mathbb{R}^{|\mathcal{V}| \times |\mathcal{V}|}$ be the node-side random-walk matrix. Following prior work, we define the $d$-dimensional node and pair-wise encodings as:
$$\mathbf{P}_i = [\mathbf{I}, \mathbf{M}, \mathbf{M}^2, \dots, \mathbf{M}^{K-1}]_{i,i}, \qquad \mathbf{R}_{i,j} = [\mathbf{I}, \mathbf{M}, \mathbf{M}^2, \dots, \mathbf{M}^{K-1}]_{i,j}, \quad K = 5.$$
These are passed through small MLPs and added to the corresponding token streams. Edge-side PE additionally concatenates the per-edge cardinality and the mean pairwise overlap $\frac{1}{|\mathcal{V}|}\sum_{j \neq e}|\mathbf{B}_{*,e} \odot \mathbf{B}_{*,j}|$.

#### 3.2.3 Layer-wise Encoder with Biased Attention
Each of the $L$ layers applies sparse top-$k$ structural self-attention over the node–edge bipartite graph. At layer $\ell$, the attention between node $i$ and the $k$-th most-overlapping neighbour $j$ of edge $e$ is biased by the hyperedge structural encoding:
$$\mathbf{N}^{(\ell+1)} = \mathbf{N}^{(\ell)} + \text{MHA}^{(\ell)}\big(\mathbf{N}^{(\ell)}, \mathbf{E}^{(\ell)}, \mathbf{B}_{\text{top-}k}\big),$$
followed by residual connection, layer-norm, and dropout. (*`EncoderLayer`*)

#### 3.2.4 Hierarchical Pooling and Readout
A learnable assignment matrix reduces both streams to a fixed budget of $P$ pooled nodes and $Q$ pooled edges:
$$\tilde{\mathbf{N}} = \text{softmax}(\mathbf{N}\mathbf{W}_{\text{node-assign}})^\top \mathbf{N}, \quad \tilde{\mathbf{E}} = \text{softmax}(\mathbf{E}\mathbf{W}_{\text{edge-assign}})^\top \mathbf{E}.$$
The graph embedding is the readout over the concatenated pooled streams: $\mathbf{g} = \mathbf{W}_{\text{readout}}[\tilde{\mathbf{N}}.\text{mean} \| \tilde{\mathbf{E}}.\text{mean}]$.

#### 3.2.5 Domain Adapter
To specialize the shared encoder to each source domain without bloating parameters, we add a lightweight residual adapter on top of each layer's output. We support two flavours:
- **Bottleneck adapter:** $\mathbf{h}' = \mathbf{h} + \mathbf{W}_{\text{up}}\sigma(\mathbf{W}_{\text{down}}^{(k)}\mathbf{h})$, $\mathbf{W}_{\text{down}}^{(k)} \in \mathbb{R}^{d \times r}$, $r = 32$.
- **Mixture-of-Experts adapter:** $M = 4$ experts; per-domain gating selects the top-2 experts: $\mathbf{h}' = \mathbf{h} + \sum_{i \in \text{top-2}} g_i^{(k)}(\mathbf{h})\mathbf{W}_i\mathbf{h}$.

The adapter is applied residually, so the shared backbone remains the dominant source of transferable features.

---

### 3.4 Self-Supervised Pre-training Tasks

(*对应:`models/pretext_tasks_neg_sam.py::compute_pretraining_losses`*)

Each task highlights one aspect of hypergraph structure. Let $\mathbf{N}, \mathbf{E}, \mathbf{g}$ denote the encoder outputs for the current subhypergraph, and let $k$ be its source domain.

#### 3.4.1 Masked Node Prediction
A fraction $\rho_f = 0.15$ of node features are zeroed out (feature-masking augmentation). The decoder reconstructs them:
$$\mathcal{L}_{\text{mnp}} = \frac{1}{|\mathcal{M}|}\sum_{i \in \mathcal{M}}\|f_{\text{dec}}(\tilde{\mathbf{N}}_i) - \mathbf{x}_i\|^2,$$
where $\mathcal{M}$ is the set of masked nodes.

#### 3.4.2 Hyperedge Discrimination with Negative Sampling
For each positive edge $e^+$, we sample $N_e = 2$ negative hyperedges from three modes — *replace* (swap one node), *overlap* (use a partially overlapping edge), and *random* (uniform sample). The score margin loss is:
$$\mathcal{L}_{\text{hr}} = -\mathbb{E}_{(e^+, e^-)}\log\sigma\big(s(\mathbf{E}_{e^+}) - s(\mathbf{E}_{e^-})\big),$$
with $s(\cdot)$ a 2-layer MLP scorer. (*`sample_hyperedge_negatives` + `compute_hyperedge_discrimination_loss`*)

#### 3.4.3 Cross-View Contrastive Learning
A second augmented view is produced by node-dropping ($\rho_n = 0.15$) and edge-dropping ($\rho_e = 0.20$). A 2-layer projector maps both views into a shared space and we apply the InfoNCE loss with temperature $\tau = 0.1$:
$$\mathcal{L}_{\text{cl}} = -\frac{1}{|\mathcal{V}|}\sum_{i \in \mathcal{V}}\log\frac{\exp(\langle z_i^{(1)}, z_i^{(2)}\rangle/\tau)}{\sum_{j}\exp(\langle z_i^{(1)}, z_j^{(2)}\rangle/\tau)}.$$

#### 3.4.4 Hyperedge Size Prediction
A 2-layer regressor predicts the log-cardinality of each hyperedge:
$$\mathcal{L}_{\text{sp}} = \frac{1}{|\mathcal{E}|}\sum_{e \in \mathcal{E}}\big(f_{\text{reg}}(\mathbf{E}_e) - \log|\mathbf{B}_{*,e}|\big)^2.$$

#### 3.4.5 Node–Hyperedge Membership Contrast (neg_sam)
For each positive $(v, e^+)$ pair, we sample $N_v = 2$ negative edges that are either **2-hop neighbours** in the edge graph (hard) or random non-incident edges (easy):
$$\mathcal{L}_{\text{mc}} = -\mathbb{E}_{(v, e^+), (v, e^-)}\log\sigma\big(m(\mathbf{N}_v, \mathbf{E}_{e^+}) - m(\mathbf{N}_v, \mathbf{E}_{e^-})\big),$$
with the bilinear scorer $m(\mathbf{N}_v, \mathbf{E}_e) = \mathbf{N}_v^\top \mathbf{W}_m \mathbf{E}_e$. This is the central **negative-sampling** objective of `neg_sam_v2`: hop-based hard negatives force the encoder to discriminate fine-grained membership, complementing the hyperedge-level negatives of §3.4.2. (*`sample_membership_negatives` + `compute_membership_contrast_loss`*)

#### 3.4.6 Domain Alignment via Gradient Reversal
A domain classifier is attached to node embeddings through a gradient-reversal layer. The encoder is encouraged to produce domain-confusable features:
$$\mathcal{L}_{\text{da}} = -\sum_{i \in \mathcal{V}}\log q(k_i \mid \mathbf{N}_i), \quad q = \text{softmax}(\mathbf{W}_{\text{dom}}\cdot\text{GRL}(\mathbf{N})).$$

#### 3.4.7 Motif and Community Prediction
Hyperedge-induced motifs and node-overlap-induced communities are sampled (budget $B = 32$). A classifier predicts the motif type from the concatenated readout $(\tilde{\mathbf{N}} \| \tilde{\mathbf{E}})$. (*`compute_motif_classification_loss`, `compute_community_alignment_loss`*)

#### 3.4.8 Structure Alignment and Subgraph Discrimination
A structure-aware head projects both augmented views into a 64-dim subspace and we minimise their cosine distance (alignment). In parallel, a binary discriminator distinguishes weak subhypergraphs from strong ones, preventing the encoder from collapsing on noisy samples. (*`compute_structure_alignment_loss`, `compute_subgraph_discrimination_loss`*)

---

### 3.5 Shared–Private Disentanglement and Multi-Granularity Alignment

(*对应:`models/shared_private_module.py` + `models/domain_alignment.py`*)

#### 3.5.1 Shared–Private Disentangler
Each encoder output is split by two MLPs into shared and private parts:
$$\mathbf{N}_s, \mathbf{N}_p = f_{\text{node-shared}}(\mathbf{N}), f_{\text{node-private}}(\mathbf{N}); \quad \mathbf{E}_s, \mathbf{E}_p = f_{\text{edge-shared}}(\mathbf{E}), f_{\text{edge-private}}(\mathbf{E}).$$
The shared branch captures domain-invariant structure; the private branch captures domain-specific statistics.

#### 3.5.2 Orthogonality and Private-Domain Losses
We force orthogonality between shared and private representations:
$$\mathcal{L}_{\text{orth}} = \frac{1}{B}\sum_b\big(1 - \cos^2(\mathbf{N}_{s,b}, \mathbf{N}_{p,b})\big),$$
and simultaneously train a small **private-domain predictor** on $\mathbf{N}_p, \mathbf{E}_p$ to retain domain information:
$$\mathcal{L}_{\text{priv-dom}} = \text{CE}(\mathbf{W}_{\text{pdp}}(\mathbf{N}_p), k) + \text{CE}(\mathbf{W}_{\text{pdp}}(\mathbf{E}_p), k).$$

#### 3.5.3 Confidence-Based Selective Routing
Not all nodes/edges are equally confident about their domain membership. We compute a structural-confidence score from edge-overlap entropy and only align high-confidence tokens:
$$\text{mask}_v = \mathbb{1}[\text{conf}(\mathbf{N}_v) \geq \tau_{\text{node}}], \quad \text{mask}_e = \mathbb{1}[\text{conf}(\mathbf{E}_e) \geq \tau_{\text{edge}}].$$

#### 3.5.4 Multi-Granularity Prototype Alignment
Each domain owns $M = 32$ learnable **structural prototypes**. We minimise an InfoNCE between masked shared embeddings and their assigned prototypes across domains:
$$\mathcal{L}_{\text{align}} = -\frac{1}{\sum\text{mask}}\sum_{v \in \text{mask}}\log\frac{\exp(\langle\mathbf{N}_{s,v}, \mathbf{p}_{\pi(v)}\rangle/\tau)}{\sum_m\exp(\langle\mathbf{N}_{s,v}, \mathbf{p}_{m,k_v}\rangle/\tau)},$$
with the analogous loss for edges. This pulls cross-domain shared representations towards a common prototype bank.

---

### 3.6 Negative Sampling and Quality-Aware Routing

(*对应:`models/negative_sampling_neg_sam.py` + `utils/negative_bank.py` + `utils/minibatch_sampling.py::*_quality_score`*)

#### 3.6.1 Subhypergraph Quality Scoring
Each sampled subhypergraph is scored along four dimensions:
$$q(\mathcal{H}') = 0.25\,s_{\text{size}} + 0.30\,s_{\text{conn}} + 0.25\,s_{\text{overlap}} + 0.20\,s_{\text{density}} \in [0,1],$$
where the components capture node/edge counts, fraction of nodes incident to at least one edge, mean pairwise overlap, and incidence fill rate, respectively.

#### 3.6.2 Quality-Based Task Routing
Each pretraining task is gated by a quality threshold:

| Task | Min quality $q$ |
|---|---|
| Membership contrast (§3.4.5) | $\geq 0.40$ |
| Hyperedge reconstruction / Contrastive (§3.4.2, §3.4.3) | $\geq 0.55$ |
| Motif / Community / Discrimination (§3.4.7, §3.4.8) | $\geq 0.70$ |
| Hard-negative reservoir | $< 0.40$ |

Invalid subhypergraphs (size / edge count below minimum) are skipped. This routing prevents the encoder from being misled by degenerate samples and naturally upweights informative ones.

#### 3.6.3 Hard-Negative Bank
Subhypergraphs that fail routing but are otherwise valid are pushed into a **tiered bank** with three quality tiers and FIFO eviction. During contrastive training the bank is sampled to provide hard negatives for the InfoNCE objectives in §3.4.3 and §3.4.5.

---

### 3.7 Subhypergraph Sampling and Training

(*对应:`utils/minibatch_sampling.py` + `trainers/pretrain_trainer_neg_sam.py`*)

For large hypergraphs, full-graph attention is intractable. We adopt a **hyperedge-centred subhypergraph** sampling strategy:

1. Pick a seed edge $e_0$.
2. BFS-expand $H$ hops over the node–hyperedge incidence, including any edge sharing at least one node with the current frontier, capped at $|\mathcal{V}| \leq V_{\max} = 256$ and $|\mathcal{E}| \leq E_{\max} = 128$.
3. Induce the subhypergraph on the visited node/edge sets.

For graphs with $|\mathcal{V}| > 5000$, an offline pool of $P = 64$ pre-sampled subhypergraphs is built once and uniformly sampled at training time.

Per step we sample $K_d = 2$ domains and $K_g = 2$ subhypergraphs per domain, forming a batch of size $K_d K_g$. We optimise with AdamW (lr $= 10^{-3}$, weight decay $= 10^{-4}$), bf16 autocast on Ampere+ GPUs, and gradient clipping at norm $1.0$. Early stopping with patience $80$ is applied on the best epoch's averaged total loss.

---

### 3.8 Combination with Downstream Feature Encoders

(*对应:`trainers/finetune_trainer.py` + `trainers/recommendation_trainer.py` + `scripts/run_transfer.py`*)

The pretrained encoder is decoupled from any specific downstream model. It can be integrated in three ways, illustrated in Figure 1.

**B.1 Feature-enriched graphs (vectorized features).** Let $\mathbf{X}_0 \in \mathbb{R}^{N \times d_x}$ be the input node features and $\mathbf{N}^{(L)} \in \mathbb{R}^{N \times d_e}$ be the pretrained node embeddings. They are concatenated to form a new feature matrix $\mathbf{X}_{\text{new}} = [\mathbf{X}_0 \| \mathbf{N}^{(L)}] \in \mathbb{R}^{N \times (d_x + d_e)}$, which is then fed to any downstream GNN (HGNN, HyperGCN, AllSetTransformer, …) or to a simple MLP. During finetuning, the pretrained encoder is fine-tuned end-to-end together with the downstream head. This is the default mode used in `finetune_node_v2.yaml`.

**B.2 Linear-probe / frozen feature extractor.** For a quick transferability check (§4.2), the encoder is frozen and only a linear classifier $\mathbf{W}_{\text{lin}} \in \mathbb{R}^{d_e \times C}$ is trained on top. This isolates the encoder's representation quality from finetune noise. (*`scripts/linear_probe_neg_sam.py`*)

**B.3 Cold-start scratch baseline.** The same architecture is initialised randomly and trained from scratch with the same downstream data and optimiser. This is the no-pretrain control used in `finetune_node_v2_scratch.yaml`. The transfer gain is reported as $\Delta = \text{Acc}_{\text{pretrained}} - \text{Acc}_{\text{scratch}}$.

**B.4 Recommendation-style graph encoder.** For bipartite user–item graphs (e.g., Gowalla, Yelp2018) the pretrained encoder is wrapped by a bilinear scorer $f_{\text{rec}}(\mathbf{N}_u, \mathbf{N}_i) = \mathbf{N}_u^\top \mathbf{W}_r \mathbf{N}_i$ and trained with a BPR-style pairwise loss. This re-uses the same pretrained weights as the node-classification setup without modification.

---

### 3.9 Computational Complexity

(*对应代码:`models/negative_sampling_neg_sam.py::sample_hyperedge_negatives`, `compute_node_to_edge_membership_sets`, `utils/minibatch_sampling.py::sample_subhypergraph_batch_with_quality`*)

The pretraining cost decomposes into four stages:

| Stage | Per-step cost | Notes |
|---|---|---|
| Subhypergraph sampling | $O(K_d H\,|\mathcal{V}|)$ | $K_d = 2$ domains, $H = 2$ expansion hops |
| Negative sampling | $O(|\mathcal{E}|^2 d)$ | vectorised via incidence$^T \cdot$ incidence |
| Membership BFS | $O(K \cdot |\mathcal{E}|^2)$ | $K = 3$ hops, sparse matmul |
| Encoder forward (3 views) | $O(L \cdot |\mathcal{V}| d^2 + L \cdot |\mathcal{E}| d^2)$ | bf16, top-$k$ sparse attention |

The full-batch attention cost is $O(L \cdot |\mathcal{V}|^2 d)$; the subhypergraph cap $V_{\max} = 256$ keeps the per-step forward tractable even for million-node hypergraphs. The hard-negative-bank and quality-routing add $O(1)$ per step (table lookups).

---

## 4. (Placeholder for Experiments — not in scope here)

Per-domain accuracy, transfer gains, and ablation tables will live in §4.

---

## Table 2 — Loss weights

| Task | Weight | Task | Weight |
|---|---|---|---|
| masked_node | 1.0 | orth_node | 0.02 |
| hyperedge_recon | 1.0 | orth_edge | 0.02 |
| contrastive | 1.0 | private_domain_node | 0.05 |
| size_pred | 1.0 | private_domain_edge | 0.05 |
| domain_align | 0.1 | motif | 0.5 |
| membership_contrast | 0.2 | community | 0.5 |
| structure_align | 0.3 | structure_discrimination | 0.3 |

