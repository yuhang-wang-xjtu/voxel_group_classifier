"""
Minimal test: does cos(teacher, student) produce meaningful difficulty?
No CUDA, no Pointcept, no training. Pure torch, 2 seconds.

Hypothesis: If teacher and student logits differ more in "hard" regions,
then cos(t,s) difficulty will have variance and be useful for mask guidance.
"""
import torch
import torch.nn.functional as F
import math

device = "cpu"
N = 2000   # tokens
K = 4096   # prototype dim (match Sonata's head_num_prototypes)

# ---- Simulate teacher logits ----
# 50% tokens: "easy" — one dominant prototype with noise
# 30% tokens: "medium" — 2-3 prototypes compete
# 20% tokens: "hard" — spread across 10+ prototypes
teacher_logits = torch.randn(N, K, device=device) * 0.5  # base noise

easy_mask = torch.zeros(N, dtype=torch.bool)
easy_mask[:1000] = True      # 50% easy
med_mask = torch.zeros(N, dtype=torch.bool)
med_mask[1000:1600] = True   # 30% medium
hard_mask = torch.zeros(N, dtype=torch.bool)
hard_mask[1600:] = True      # 20% hard

for i in range(N):
    if easy_mask[i]:
        teacher_logits[i, :100] += 3.0    # strong signal in first 100 prototypes
    elif med_mask[i]:
        teacher_logits[i, 100:200] += 1.5  # weaker, competing signal
        teacher_logits[i, 200:300] += 1.5
    elif hard_mask[i]:
        teacher_logits[i, 300:400] += 0.8  # very weak, spread
        teacher_logits[i, 400:500] += 0.8

# ---- Simulate student logits (worse version of teacher) ----
# Student = teacher + noise. Harder tokens → more noise (student struggles more)
student_logits = teacher_logits.clone()
student_noise = torch.randn(N, K, device=device) * 0.3

for i in range(N):
    if easy_mask[i]:
        student_logits[i] += student_noise[i] * 0.2     # easy: low noise
    elif med_mask[i]:
        student_logits[i] += student_noise[i] * 0.8     # medium: more noise
    elif hard_mask[i]:
        student_logits[i] += student_noise[i] * 2.0     # hard: high noise

# ---- Compute cosine difficulty ----
cos_sim = F.cosine_similarity(teacher_logits, student_logits, dim=-1)
difficulty = 1.0 - cos_sim  # [0, 2]

# ---- Results ----
print("=" * 50)
print("Cosine Difficulty Metric — Feasibility Test")
print("=" * 50)
for name, mask in [("Easy", easy_mask), ("Medium", med_mask), ("Hard", hard_mask)]:
    d = difficulty[mask]
    c = cos_sim[mask]
    print(f"  {name:>8}:  cos={c.mean():.4f}±{c.std():.4f}  "
          f"difficulty={d.mean():.4f}±{d.std():.4f}")

print(f"\n  Overall:  cos={cos_sim.mean():.4f}±{cos_sim.std():.4f}  "
      f"difficulty={difficulty.mean():.4f}±{difficulty.std():.4f}")
print(f"  Difficulty range: [{difficulty.min():.4f}, {difficulty.max():.4f}]")

# Verdict
gap = (difficulty[hard_mask].mean() - difficulty[easy_mask].mean()).item()
if gap > 0.1 and difficulty.std() > 0.05:
    print(f"\n✅ PASS: Hard regions have significantly higher difficulty (gap={gap:.3f})")
    print(f"   cos(t,s) IS a viable difficulty metric for mask guidance.")
else:
    print(f"\n❌ FAIL: Gap too small (gap={gap:.3f})")
    print(f"   Check if teacher-student divergence is meaningful enough.")
