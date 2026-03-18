# Experiment Suite: MoE Router Variants

**Date:** 2024-03-18  
**Researcher:** [Your Name]  
**Objective:** Test effect of routing temperature and compute penalty on MoE performance

## Quick Start

```bash
# Submit all 5 experiments
cd conf/experiment && bash submit_all.sh

# Check status
qstat -u $USER

# View results (after completion)
# https://wandb.ai/your-entity/moe-5g-nrx
```

## Experiments

### 01: Baseline Dense
**Config:** `exp01_baseline_dense.yaml`  
**Purpose:** Reference point - standard dense neural receiver  
**Key Params:**
- Model: static_dense
- Hidden dim: 256
- Depth: 8

**Expected:** BER ~0.0015, high compute (100% active)

---

### 02: MoE Default
**Config:** `exp02_moe_default.yaml`  
**Purpose:** MoE with balanced settings  
**Key Params:**
- Temperature: 1.0 (balanced exploration)
- FLOPs penalty: 0.0001
- Experts: tiny(64,2), medium(128,4), heavy(256,8)

**Expected:** Similar BER to dense, ~40% compute reduction

---

### 03: MoE Hard Routing
**Config:** `exp03_moe_hard_routing.yaml`  
**Purpose:** Sharper routing decisions  
**Key Params:**
- Temperature: 0.3 (harder gating)
- FLOPs penalty: 0.0001

**Hypothesis:** Cleaner expert specialization, possibly better accuracy but less flexibility

---

### 04: MoE Soft Routing
**Config:** `exp04_moe_soft_routing.yaml`  
**Purpose:** Softer, more adaptive routing  
**Key Params:**
- Temperature: 2.0 (softer gating)
- FLOPs penalty: 0.0001

**Hypothesis:** Better generalization but might mix experts too much

---

### 05: MoE High Compute Penalty
**Config:** `exp05_moe_high_penalty.yaml`  
**Purpose:** Aggressive compute reduction  
**Key Params:**
- Temperature: 1.0
- FLOPs penalty: 0.001 (10x default)

**Hypothesis:** Forces use of tiny/medium experts, significant compute savings, possible accuracy drop

---

## Results Summary

| Exp | Model | Temp | Penalty | Final BER | Compute Saved | Status |
|-----|-------|------|---------|-----------|---------------|--------|
| 01 | Dense | - | - | TBD | 0% | ⏳ Queued |
| 02 | MoE | 1.0 | 0.0001 | TBD | TBD | ⏳ Queued |
| 03 | MoE | 0.3 | 0.0001 | TBD | TBD | ⏳ Queued |
| 04 | MoE | 2.0 | 0.0001 | TBD | TBD | ⏳ Queued |
| 05 | MoE | 1.0 | 0.001 | TBD | TBD | ⏳ Queued |

**Last Updated:** 2024-03-18  
**Next Action:** Analyze results when all jobs complete
