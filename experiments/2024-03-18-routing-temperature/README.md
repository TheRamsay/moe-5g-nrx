# Experiment: Router Temperature Sweep

**Date:** 2024-03-18  
**Researcher:** [Your Name]  
**Objective:** Test effect of routing temperature and compute penalty on MoE performance

## Quick Start

```bash
cd experiments/2024-03-18-routing-temperature
bash submit.sh
```

## Experiments

| Exp | Name | Temperature | Penalty | Expected BER | Expected Compute |
|-----|------|-------------|---------|--------------|------------------|
| 01 | Baseline Dense | - | - | ~0.0015 | 100% |
| 02 | MoE Default | 1.0 | 0.0001 | ~0.0015 | ~60% |
| 03 | Hard Routing | 0.3 | 0.0001 | ~0.0013 | ~55% |
| 04 | Soft Routing | 2.0 | 0.0001 | ~0.0017 | ~65% |
| 05 | High Penalty | 1.0 | 0.001 | ~0.0018 | ~40% |

## Results

| Exp | Final BER | Compute Used | Wall Time | Status | Notes |
|-----|-----------|--------------|-----------|--------|-------|
| 01 | TBD | TBD | TBD | ⏳ | |
| 02 | TBD | TBD | TBD | ⏳ | |
| 03 | TBD | TBD | TBD | ⏳ | |
| 04 | TBD | TBD | TBD | ⏳ | |
| 05 | TBD | TBD | TBD | ⏳ | |

## Key Findings

*Fill in after experiments complete*

## Next Steps

*Based on results, what to try next*

## Wandb Links

- [Project Dashboard](https://wandb.ai/[entity]/moe-5g-nrx)
- [This Batch Group](https://wandb.ai/[entity]/moe-5g-nrx/groups/routing-temperature-2024-03-18)
