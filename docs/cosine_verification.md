# Cosine Difficulty Metric: 验证方法

运行 `masking_mode=cosine`（10 epoch smoke test）后，在 TensorBoard 中逐一检查以下信号。

---

## 1. Difficulty 是否有区分度？

### 观察指标
| 指标 | 期望 | 解释 |
|------|------|------|
| `cosine/difficulty_std` | > 0.1 且不趋零 | 不同 token 之间 student 的预测质量有差异 |
| `cosine/p10_p90_gap` | 随 epoch 逐渐增大 | 容易和困难的 token 被拉开——说明确有"难"和"易"之分 |

### 验证判据
- Epoch 1: gap ≈ 0（初始没有区分）→ **正常**
- Epoch 3-5: gap 应 > 0.1 → **通过**；gap < 0.05 → **失败**
- Epoch 10: gap 应 > 0.15 → **通过**；gap ≈ 0.02 → **失败**

---

## 2. Difficulty 是否和语义结构一致？

### 实验设计
在同一个 Sonata pretraining 框架下，跑 3 组对比：

| 实验 | `masking_mode` | 配置 |
|:----:|:--------------:|------|
| A | `random` | 原版 Sonata 基线 |
| B | `cosine` | 用 cos(t,s) 指导 mask |
| C | `cosine_invert` | 用 1 - difficulty 指导 mask（即 mask 最容易的） |

### 验证判据
- B > A（cosine 比 random 好）→ cosine difficulty 是**有用的**
- C < A（mask 容易的比 random 差）→ difficulty 正确地定位了"真正困难的区域"
- B ≈ A 且 C ≈ A → cosine difficulty 没有额外信息

---

## 3. Difficulty 是否随时间稳定？

### 观察指标
| 指标 | 期望 |
|------|------|
| `cosine/difficulty_mean` | 随 epoch 下降（模型在学）或至少不爆炸上升 |
| `cosine/difficulty_std` | 在中期（epoch 3-7）达到峰值后稳定 |
| `cosine/cos_sim_mean` | 随 epoch 上升（学生越来越能预测老师） |

### 验证判据
- cos_sim_mean 单调上升 → **通过**（模型确实在学）
- cos_sim_mean 在高位震荡但 difficulty_std 仍 > 0.05 → **通过**（即使整体 cos_sim 已经很高，仍然有部分 token 相对更难）
- cos_sim_mean 收敛到接近 1 且 difficulty_std → 0 → **失败**（任务太简单，所有 token 都学会了）

---

## 4. 快速验证流程（推荐）

```
1. 跑 random masking baseline: 10 epoch, 记录 train/mask_loss
2. 跑 cosine masking:          10 epoch, 记录 train/mask_loss + cosine/*
3. 跑 cosine_invert masking:   10 epoch, 记录 train/mask_loss

如果 10 epoch 太短看不出来，延长到 30 epoch。
初始结果 10 epoch 内即可看到区分。
```

---

## 5. 判定矩阵

| `cosine/difficulty_std` | `cosine/p10_p90_gap` | 掩码实验 B vs A | 结论 |
|:--:|:--:|:--:|------|
| > 0.15 | > 0.3 | B > A | ✅ **Difficulty 有效，方法完整可行** |
| > 0.10 | > 0.2 | B ≈ A | 🟡 信号弱，需更长时间训练验证 |
| > 0.10 | > 0.2 | B < A | ⚠️ 方向反了，可能应该 mask 容易的 |
| < 0.05 | < 0.1 | — | ❌ 无区分度，cos(t,s) 在该框架下不是好的难度信号 |
