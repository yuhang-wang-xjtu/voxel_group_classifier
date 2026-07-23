# VoxelGroupClassifier：基于分类器熵的自适应掩码策略

## 一、动机

Sonata 的标准预训练流程中，输入 token 的掩码是**随机**生成的：

```
mask ~ Bernoulli(p)      p 是固定比例（如 0.7）
```

每个 token 被遮住的概率完全相同，无论模型是否已经学会该区域。

**我们的核心假设**：

> 模型自己对某个体素所属语义组的判断**肯定度**，比任何外部信号（几何复杂度、注意力权重）都更能反映"模型还需要从这个区域学多少东西"。

我们用一个**可学习的分类器**来量化这种肯定度——分类器的预测熵越低，说明模型对该体素的语义归属越确定。

---

## 二、方法总览

Teacher 的 `OnlineCluster` 生成的是**原始 cosine 相似度**（未经过 SK 锐化）。分类器以此为软目标，KL 散度驱动训练。

```
┌─────────────────────────────────────────┐
│ 1. Teacher backbone（EMA，冻结）         │
│    │                                     │
│    ├─ mid_features（第 L/2 层输出）       │
│    └─ OnlineCluster → raw_cos_sim [K]    │
│       （BEFORE Sinkhorn-Knopp sharpening）│
│       使用更高的温度 τ=2.0 做软目标      │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│ 2. VoxelGroupClassifier（可学习）         │
│                                           │
│    MLP: D → 128 → K                      │
│    预测每个 subvoxel 的组归属             │
│                                           │
│    H = -Σ p·log(p)   ← 香农熵            │
│    L_cls = KL(softmax(logits) || teacher) │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────┐
│ 3. 自适应掩码生成                    │
│                                       │
│    difficulty = topK_mean(entropy)    │
│    p_mask = f(difficulty)             │
│    mask ~ Bernoulli(p_mask)           │
└──────────────────┬──────────────────┘
                   │
┌──────────────────▼──────────────────┐
│ 4. Student backbone                  │
│    │                                 │
│    处理被掩码的输入                   │
│    对被遮住的 token 预测 teacher 特征 │
│    蒸馏损失 L_distill                 │
└──────────────────────────────────────┘
```

---

## 三、为什么用可学习分类器？

### 3.1 已有的难度度量方案

| 方法 | 度量方式 | 问题 |
|------|---------|------|
| 随机掩码（Sonata 默认） | 无度量 | 无区分度 |
| 几何复杂度（GeoMask3D） | 法向量方差、曲率 | 几何复杂 ≠ 语义难 |
| 注意力显著性（Point-CSMAE） | self-attention 权重 | 注意了不等于没学会 |
| 原型分配熵（GroupContrast） | cos 相似度 → softmax → entropy | **非参数、batch 依赖、密度敏感** |

### 3.2 可学习分类器的三个优势

**（1）跨场景泛化**

Sonata teacher 的 `prototyp_head` 计算的是 `cos(feature, prototype) → softmax`。这是逐样本独立的——同一个墙面在两个 batch 里可能得到不同的分配概率。

分类器是**全局拟合**的：它的参数一旦训练好，对任何场景中的相同类型特征区域都给出稳定的预测。entropy 反映的是"这个特征空间区域是否本身就模棱两可"，而不是"当前 batch 里这个样本排在什么位置"。

**（2）平滑的难度流形**

分类器（MLP）学习的是特征空间上的平滑决策面。稀疏采样的噪声点无法改变分类器的整体决策边界，因此不会像非参数方法那样在低密度区域产生虚假的高 entropy。

**（3）自驱动的课程学习**

训练早期，分类器还没有学会有意义的决策边界 → 所有 subvoxel 的 entropy 都接近 log(K)（均匀分布）→ 自适应掩码退化为随机掩码。

随着训练推进，分类器逐渐学会区分不同的特征簇 → entropy 在平坦区域（墙面、地面）快速下降到 0 → 在语义边界、复杂物体上保持高值 → 掩码自然从"均匀"过渡到"聚焦困难区域"。

**整个过程不需要手动设计 schedule，是一个 emergent property。**

---

## 四、核心算法

### 4.1 输入

| 符号 | 维度 | 含义 |
|------|------|------|
| `subvoxel_features` | `[M, P³, D]` | M 个 token，每个有 P³=64 个 subvoxel，特征维度 D=384 |
| `teacher_raw_sim` | `[M, P³, K]` | Teacher 的 OnlineCluster 原始 cos 相似度（**BEFORE SK**），K=4096 个原型 |
| `prototypes` | `[K, D]` | 可学习的原型向量（已在 Sonata OnlineCluster 中定义） |

> **关键设计**：Sonata teacher 的完整流程是 `feat → L2 norm → Linear(prototypes) → SK(sim/0.04)`。SK 输出是近乎 one-hot 的。如果直接用它做分类器目标，分类器也会输出近乎 one-hot → entropy 处处为零 → 无效。我们改用 **SK 之前的 raw similarity + 高温度 softmax（τ=2.0）** 做目标，保留 teacher 输出中的 ambiguity structure。

### 4.2 分类器前向

```
logits = Classifier(subvoxel_features)    → [M, P³, K]
probs  = softmax(logits / τ_cls)          → [M, P³, K]

entropy = -Σ_k probs_k · log(probs_k + ε) → [M, P³]
```

### 4.3 分类器训练损失

```
teacher_soft = softmax(teacher_raw_sim / τ_target)   // τ_target = 2.0
teacher_soft = label_smooth(teacher_soft, 0.1)

L_cls = KL(log_softmax(logits) || teacher_soft.detach())

总损失 = L_distill（Sonata 蒸馏损失）+ λ_cls · L_cls
```

**为什么用 KL 散度而不是 CrossEntropy**：

- CE 隐含地将 teacher 输出视为 one-hot 硬标签 → 信息损失
- KL 散度保留 teacher softmax 的完整概率分布 → 保留模糊性
- label smoothing 防止 teacher 中局部噪声导致分类器过拟合

### 4.4 教师信号的温度对比

| 用途 | 温度 | 结果 |
|------|:--:|------|
| Teacher SK（原始） | **τ = 0.04** | 近乎 one-hot → 信息量极低 → 不能做分类器目标 |
| Teacher raw_sim → softmax | **τ = 2.0** | 软分布 → 保留语义模糊性 → 可以用做分类器目标 |
| Student distillation | **τ = 0.1** | Sonata 标准 student 温度 |

### 4.4 Entropy → 难度 → 掩码概率

```
# 步骤 1：Subvoxel entropy → token difficulty（取 top 10% 平均）
difficulty_i = mean(top_10%(entropy_i)) / log(K)     → [M], 取值 [0,1]

# 步骤 2：中位数锚定的 sigmoid 掩码概率
median   = difficulty.median()
p_adapt  = r_target * (1 + σ((difficulty - median) / τ_mask))

# 步骤 3：多样性惩罚——防止反复掩码同一区域
p_diverse = p_adapt * (1 - diversity_penalty)   ← 见 4.6 节

# 步骤 4：课程权重 α
p_final  = (1-α) * r_target + α * p_diverse

# 步骤 5：伯努利采样
mask_i ~ Bernoulli(p_final_i)
```

### 4.5 冷启动与课程机制

### 4.6 多样性约束：防止反复掩码同一区域

**问题**：如果分类器始终认为某个区域是 "高难度"（例如语义边界、罕见物体），该区域会每个 epoch 都被 mask → 模型永远看不到 → 永远学不会 → 形成负反馈。

**方案**：`MaskDiversityScheduler` 追踪每个 token 最近被 mask 的频率，对高频 token 施加概率惩罚。

```
mask_history[i] = 0.9 * mask_history[i] + 0.1 * mask[i]     // EMA，衰减系数 0.9
penalty[i]      = min(mask_history[i], 0.5)                  // 最高惩罚 50%

p_diverse[i]    = p_adapt[i] * (1 - penalty[i])              // 密度惩罚后的概率
```

**效果**：某 token 被 mask 多次后，其被 mask 概率逐渐降低 → 偶尔被 "释放" 出来做 visible → 模型有机会学习该区域 → 分类器 entropy 可能下降 → 自然地不再高概率 mask 它。

这是 `difficulty_to_mask_probs` 中的可选参数 `diversity_penalty`。

```
α = curriculum_alpha(epoch, warmup_epochs=总轮数 * 0.2)

    epoch < warmup    → α = 0  → 纯随机掩码 = Sonata baseline
    epoch ≥ warmup    → α : 0→1  (余弦上升)
```

**为什么这样做**：训练初期 teacher 的原型分配本身就是噪声。如果此时就让分类器去追噪声目标，然后用噪声 entropy 选掩码，训练的早期信号完全不可靠。因此前 20% epoch 等价于 Sonata 基线，分类器在这段时间"观察但不干预"——它仍然在更新参数（L_cls 回传），但它的 entropy 不用于生成掩码。

---

## 五、与已有方法的对比

| 方法 | 难度度量 | 可学习？| 场景级？| 用途 |
|------|---------|:------:|:------:|------|
| **Sonata（基线）** | 无（随机） | — | ✅ | 掩码位置 |
| **GroupContrast** (CVPR 2024) | 原型分配熵 | ❌ | ✅ | 损失加权 |
| **GeoMask3D** (2024) | 几何复杂度 | ❌ | ✅ | 掩码位置 |
| **Point-CSMAE** (2025) | 注意力显著性 | ❌ | ❌ 单物体 | 掩码位置 |
| **P²CS** (CVPR 2026) | 余弦分组 | ❌ | ❌ 单物体 | 组级掩码 |
| **Ours (VGC)** | **分类器预测熵** | **✅** | ✅ | 掩码位置 |

---

## 六、关键设计决策：为什么不用 SK 输出做分类器目标

### 6.1 问题

Sonata teacher 的在线聚类流程是：

```
feat → L2 normalize → Linear(4096 prototypes) → raw_cos_sim
       │
       └─→ SK(raw_cos_sim / 0.04, 3 iters) → teacher_target
```

SK 使用的温度仅 **0.04**。`exp(sim / 0.04)` 将差异放大 25 倍，使 SK 输出近乎 one-hot（如 `[0.997, 0.001, 0.001, ...]`）。

如果直接把 SK 输出作为分类器的训练目标：

```
分类器 → CE loss(SK one-hot) → 分类器也学会输出 one-hot → entropy ≈ 0 处处 → 失效
```

### 6.2 解决方案

用 SK 之前的 `raw_cos_sim`，套一个显著高于 SK 的温度（τ=2.0，vs SK 的 0.04），得到软目标：

```
teacher_soft = softmax(raw_cos_sim / 2.0)       ← 保留模糊性
classifier   → KL(log_softmax(logits) || teacher_soft)  ← KL 而非 CE
```

- **温度 2.0**：50 倍于 SK 温度，不做任何锐化，保留原始的相似度差异
- **KL 散度而非 CrossEntropy**：KL 保留概率分布的完整信息；CE 隐含地做 argmax 转为硬标签
- **label smoothing 0.1**：防止 teacher 解中偶然的噪声峰值污染训练信号

### 6.3 在 Sonata 代码中的集成位置

`raw_cos_sim` 在 `OnlineCluster.forward()` 中已经在 SK 调用之前被计算了，不需要新增 forward 开销：

```python
# Sonata original (sonata_v1m2_uni_teacher_head.py L409-414):
for clustered_idx, cluster_proj in enumerate(self.cluster_projs):
    cluster_embedding = cluster_proj(
        self.cluster_z_projs[clustered_idx](teacher_backbone_feat)
    )                                           # → raw_cos_sim [N, 4096]
    cluster_targets = self.sinkhorn_knopp(       # → SK one-hot
        cluster_embedding, self.teacher_temp
    )
    # ---- 在这里提取 raw_cos_sim（cluster_embedding）即可 ----
```

---

## 七、实现细节

### 分类器架构

```
输入: D 维特征（来自 teacher 第 L/2 层 encoder）
隐藏层: 128 维，GELU 激活 + LayerNorm
输出: K 维 logits（K = num_prototypes = 256）
参数量: D × 128 + 128 × 256 ≈ 82K（可忽略）
```

### 关键超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_groups` | 4096 | 与 Sonata OnlineCluster 的原型数量一致 |
| `hidden_dim` | 128 | 分类器隐藏层维度 |
| `label_smooth` | 0.1 | 对 teacher 软目标的平滑 |
| `τ_cls` | 1.0 | 分类器 softmax 温度 |
| `τ_target` | **2.0** | teacher raw_sim softmax 温度（远高于 SK 的 0.04） |
| `λ_cls` | 0.1 | 分类器损失权重 |
| `warmup_epochs` | 总轮数 × 0.2 | 随机掩码 warmup |
| `τ_mask` | 1.0 | 掩码 sigmoid 温度 |
| `diversity_decay` | 0.9 | 掩码历史 EMA 衰减率 |
| `diversity_max_penalty` | 0.5 | 最大多样性惩罚（50%） |

### 更新策略

- Teacher backbone + prototyp_head → **EMA 更新**（与 Sonata 一致）
- Student backbone → **梯度更新**（与 Sonata 一致）
- Classifier → **梯度更新**（与 student 同步优化）
- Teacher 赋值 detach → 分类器梯度不回传进 teacher

---

## 七、核心设计决策：掩码方向

### 7.1 两种可能的方向

| 方向 | 策略 | 直觉 |
|:----:|------|------|
| **A** | 掩码**高熵** token（不确定的 → mask） | "我不知道这是什么 → 遮住 → 强迫学" |
| **B** | 掩码**低熵** token（确定的 → mask） | "我已经学会这个了 → 遮住 → 不需要再看" |

### 7.2 各自的问题

**方向 A 的问题**：

> 高熵区域正是模型信息最少的区域。MAE 的核心机制是 "从可见部分推理被遮住的部分"。如果可见部分只剩下简单区域（墙面、地面），模型没有足够的 context 去推理那些本身就理解不了的复杂区域。冷启动时更致命——全屏高熵 → 全屏 mask → 无可见内容 → 崩溃。

**方向 B 的问题**：

> 低熵区域不一定是 "不重要的"。如果模型很确定某个桌子腿是什么，遮住它——但桌子的完整语义需要腿来提供上下文。而且后期 entropy 普遍偏低时，方向 B 可能只剩下噪声/tail 区域作为可见——模型只看到自己不理解的，推理自己已经理解的，这种学习信号价值有限。

### 7.3 不应该二选一

正确的做法是**概率倾斜，而非硬选择**：用分类器 entropy 调整每个 token 被 mask 的**概率**，但不完全决定。

```
高熵 token → mask 概率提高（但仍然保留一部分可见）
低熵 token → mask 概率降低（但仍然有一部分被 mask）
```

这正是 `difficulty_to_mask_probs` 中 sigmoid 中位数锚定的效果。配合 curriculum α，训练早期所有 token 接近随机 mask（概率差异小），后期概率差异加大（但所有 token 仍有非零的 mask 和可见概率）。

---

## 八、理论基础

### 8.1 为什么分类器 entropy 能度量学习进度？

假设 teacher 的 `prototyp_head` 输出 $p_T$ 可以近似为 "理想的语义归属"（因为 teacher 通过 EMA 积累了更多训练步数的知识）。分类器 $p_C$ 是 student 视角对同一问题的回答。

如果 $p_C$ 接近 $p_T$（低 entropy），说明 student 已经掌握了这个区域的语义分组——特征空间中这个区域的流形是明确的。
如果 $p_C$ 远离 $p_T$（高 entropy），说明 student 还在 "犹豫"——特征流形在这个区域是模糊的，或者类间边界尚未确立。

因此 **$H(p_C)$ 反映的是 student 在特定空间位置上的学习完成度**，而非静态的内容重要性。这与注意力显著性、几何复杂度等信号有本质区别。

### 8.2 为什么是可学习的，而非直接算 teacher 的 entropy？

如果直接用 teacher prototyp_head 的 softmax entropy：

$$H(p_T) = -\sum_k p_T^k \log p_T^k$$

这个值受三个因素污染：

1. **Batch 依赖**：prototyp_head 的 softmax 分母是所有原型的和，分子是单条样本。同一场景在不同的 batch 里可能得到不同的 assignment
2. **密度敏感**：sparse voxel 的特征噪声大 → cosine similarity 波动大 → 虚假高 entropy
3. **非参数性**：teacher 不会 "记住" 某个区域难不难——每次都要重新计算，无法跨场景泛化

而分类器 $p_C = f_\theta(features)$ 是一个**全局函数**，其参数 $\theta$ 通过大量样本拟合了特征空间到原型分布的整张映射——$H(p_C)$ 反映的不是单样本噪声，而是特征空间本身的模糊性。

### 8.3 自驱动课程学习：理论保证

**命题**：如果分类器能渐近地逼近 teacher 赋值，那么分类器 entropy 会自然形成一个从 "均匀" 到 "结构化" 的过渡。

- 训练初期：teacher 的原型分配接近随机（student/teacher 都未学到好特征）
- 分类器试图逼近随机目标 → 学到的是接近常数的决策边界 → $H(p_C) \approx \log K$ 处处均匀
- 随 teacher 原型分配开始呈现结构 → 分类器逐渐学到了非平凡决策边界
- 简单区域（平坦特征，分类器快速学会）→ entropy 快速下降
- 复杂区域（模糊特征，分类器学得慢）→ entropy 保持高位
- **结果**：无需任何手动调度的自适应课程

**关键假设**：teacher 的原型分配必须足够稳定，才能让分类器渐进逼近。如果 teacher 分配剧烈震荡，分类器无法学到有意义的决策边界，entropy 始终均匀 → 退化为随机 mask。

---

## 九、消融实验设计

### 9.1 核心消融：掩码方向

| 实验 | 策略 | 目的 |
|:----:|------|------|
| A | 随机掩码（Sonata 原版） | **基线** |
| B | 掩码高熵 token（`p ∝ entropy`） | 验证 "mask hard" 是否更有效 |
| C | 掩码低熵 token（`p ∝ 1-entropy`） | 验证 "mask easy" 是否更有效 |
| D | 概率倾斜（sigmoid 锚定，当前实现） | 验证 "概率倾斜" 是否优于硬方向 |

### 9.2 分类器组件消融

| 实验 | 条件 | 目的 |
|:----:|------|------|
| E | 无 warmup（α = 1 从头开始） | 验证冷启动的影响 |
| F | 无 label smoothing | 验证 smoothing 是否防止 entropy 崩塌 |
| G | 无 curriculum（α 始终为 1） | 验证 curriculum ramp 的必要性 |
| H | λ_cls = 0（分类器不更新） | 验证分类器是否确实学到了有用信息 |

### 9.3 分类器权重敏感性

| 实验 | λ_cls | 目的 |
|:----:|-------|------|
| I | 0.01 | 验证损失权重极小值 |
| J | 0.1（默认） | — |
| K | 1.0 | 验证分类器是否压倒主任务 |

### 9.4 需要监控的实验指标

| 指标 | 为什么重要 |
|------|-----------|
| **分类器 entropy 分布直方图**（按 epoch 展开） | 验证 entropy 是否从均匀分布逐渐分化 |
| **高 entropy 子体素的可视化**（映射颜色到点云） | 验证高 entropy 对应的是语义边界而非噪声 |
| **分类器 top-1 准确率**（vs teacher 硬标签） | 验证分类器是否收敛、是否崩塌到 1-2 个类 |
| **预训练 loss 曲线**（vs 随机 baseline） | 验证自适应 mask 是否让学习更困难（loss 应更高） |
| **下游 mIoU**（线性 probing + 全微调） | 最终判据 |
| **Entropy-难度相关性**（Pearson r，按语义类别分组） | 验证分类器 entropy 是否与语义复杂度相关 |
