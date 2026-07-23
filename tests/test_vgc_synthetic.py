"""
Minimal, self-contained test of VoxelGroupClassifier.
Uses SYNTHETIC features with guaranteed class separability.
No downloads, no dependencies beyond torch.
"""

import sys, math
import torch
import torch.nn.functional as F
import numpy as np

# ----- Add VGC -----
sys.path.insert(0, r"F:\作业2.0\大四下\研究\Volt\sonata-clean\pointcept\models\sonata")
from voxel_group_classifier import VoxelGroupClassifier, MaskDiversityScheduler

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ============================================================
# SYNTHETIC DATA: 2 classes, CLEARLY separable features
# ============================================================
M, P3, D_signal, K = 100, 16, 4, 2  # 100 tokens, 16 subvoxels, 4 signal dims, 2 classes

# Each subvoxel gets a feature vector with a clear class signature
# Class 0: features = [1, 0, 0, 0] + noise
# Class 1: features = [0, 1, 0, 0] + noise
labels = torch.randint(0, K, (M, P3), device=device)  # random class per subvoxel

signal = torch.zeros(M, P3, D_signal, device=device)
for c in range(K):
    mask = labels == c
    signal[mask, c] = 1.0  # one-hot signal channel per class

# Add controlled noise (not too much, not too little)
noise = torch.randn(M, P3, D_signal, device=device) * 0.3
features = signal + noise
features = F.normalize(features, dim=-1)  # normalize to unit length

# Teacher: one-hot with scale
teacher_sim = F.one_hot(labels, K).float() * 10.0

# Difficulty: label entropy per token
label_ent = torch.zeros(M, device=device)
for i in range(M):
    uniq, cnt = torch.unique(labels[i], return_counts=True)
    p = cnt.float() / cnt.sum()
    label_ent[i] = -torch.sum(p * torch.log(p + 1e-8))
q33, q66 = label_ent.quantile(0.33), label_ent.quantile(0.66)
diff_label = torch.zeros(M, dtype=torch.long, device=device)
diff_label[label_ent > q66] = 2
diff_label[(label_ent > q33) & (label_ent <= q66)] = 1

print(f"Features: {features.shape}, Teacher: {teacher_sim.shape}")
print(f"Easy: {(diff_label==0).sum().item()}  Med: {(diff_label==1).sum().item()}  Hard: {(diff_label==2).sum().item()}")
print(f"Max entropy: log({K}) = {math.log(K):.3f}")

# ============================================================
# VGC TRAINING
# ============================================================
vgc = VoxelGroupClassifier(
    embed_dim=D_signal, hidden_dim=64, num_groups=K,
    label_smooth=0.0,
    temperature=0.5,       # τ for entropy computation
    loss_weight=1.0,
    teacher_target_temp=0.5,  # τ for teacher target softmax
).to(device)
print(f"VGC params: {sum(p.numel() for p in vgc.parameters()):,}")

opt = torch.optim.AdamW(vgc.parameters(), lr=0.01)
occupancy = torch.ones(M, P3, 1, device=device)

# Run: every 50 epochs, print entropy stats
for ep in range(200):
    opt.zero_grad()
    out = vgc(features, teacher_sim, occupancy)
    out["cls_loss"].backward()
    opt.step()

    if ep % 50 == 0 or ep == 199:
        with torch.no_grad():
            diff = vgc.get_difficulty(features)
            ent = out["entropy"]
            print(f"  Epoch {ep:3d}:  loss={out['cls_loss'].item():.4f}  "
                  f"entropy_mean={ent.mean().item():.3f}  "
                  f"entropy_min={ent.min().item():.3f}  "
                  f"entropy_max={ent.max().item():.3f}  "
                  f"easy_diff={diff[diff_label==0].mean().item():.3f}  "
                  f"hard_diff={diff[diff_label==2].mean().item():.3f}")

# ============================================================
# FINAL CHECK
# ============================================================
with torch.no_grad():
    out = vgc(features, teacher_sim, occupancy)
    ent = out["entropy"]
    ent_easy = ent[diff_label == 0].mean().item()
    ent_hard = ent[diff_label == 2].mean().item()
    gap = ent_hard - ent_easy

    # Also check per-subvoxel classification accuracy
    logits = vgc.layers(features)
    preds = logits.argmax(dim=-1)
    acc = (preds == labels).float().mean().item()

print(f"\n{'='*60}")
print(f"FINAL:  loss={out['cls_loss'].item():.6f}")
print(f"  Entropy    easy={ent_easy:.3f}  hard={ent_hard:.3f}  gap={gap:+.3f}")
print(f"  Difficulty easy={vgc.get_difficulty(features)[diff_label==0].mean().item():.3f}  "
      f"hard={vgc.get_difficulty(features)[diff_label==2].mean().item():.3f}")
print(f"  Classification accuracy: {acc*100:.1f}%")
print(f"  Max possible entropy: log({K}) = {math.log(K):.3f}")

if gap > 0.1:
    print(f"\n✅ VGC WORKS: entropy separates easy from hard regions (gap={gap:.3f})")
elif acc > 0.8:
    print(f"\n✅ VGC LEARNED (acc={acc*100:.1f}%) but entropy gap is small ({gap:.3f})")
    print("   → Try increasing temperature or decreasing teacher_target_temp")
else:
    print(f"\n❌ VGC NOT LEARNING: acc={acc*100:.1f}%, gap={gap:.3f}")
    print("   → Check gradient flow or feature quality")
