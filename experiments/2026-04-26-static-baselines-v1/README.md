# Static + SNR-Oracle Cascade Baselines (v1)

## Question

Does the learned MoE router beat hand-engineered routing rules?

If a simple rule (per-profile static, or per-sample SNR cascade) achieves the
same BLER/FLOPs as exp26, the learned-router story is weak. If exp26 wins,
we've shown the router learned something a rule can't replicate.

## Background

The full evidence chain so far:
- exp26 (3-expert, learned router): 0.902 avg BLER / 56% FLOPs
- Random-router ablation (exp30): channel-aware features matter (BLER craters
  6.6 pp without them)
- 2-expert ablation (exp31): nano earns its keep (0.7 pp + 9 pp FLOPs without)

Missing: a **rule-based routing baseline** that the learned router has to
beat. Without it, "the router learned channel-aware routing" is unfalsifiable
relative to "any reasonable routing rule works."

## Plan

Three rule-based baselines, computed from per-SNR-bin BLER on the test set:

1. **Always-X** (X ∈ {nano, small, large}): degenerate — always pick one
   expert. Reference points.
2. **Per-profile static**: pick the cheapest dense expert per profile that's
   within 1 pp of dense_large's BLER. Best fixed rule that uses only the
   profile label.
3. **SNR-oracle cascade**: per (profile, SNR bin), pick cheapest expert
   within `+X pp` of dense_large's BLER. Uses **true per-sample SNR**, which
   is an oracle (not deployable but informative as upper bound).

All baselines are evaluated against the same test set (`dense-v1/test`) with
the same dense_{nano,small,large} checkpoints already in W&B.

## Implementation

`scripts/analyze_static_baselines.py` pulls per-SNR-bin BLER from W&B eval
runs and computes the cascade analytically. **No new training** — just one
analysis script and 3 cheap re-evals to refresh the dense evals (the old
ones don't have per-SNR bins in summary; the current evaluate.py does).

## Cluster

- 3 dense re-eval jobs, ~15 min each (eval-only, GPU, fast)
- 1 analysis script run locally afterward

## Submission

```bash
# Re-eval dense baselines so per-SNR bins land in W&B summary
bash submit_dense_reeval.sh qsub

# Once jobs land, run the analysis locally
uv run python scripts/analyze_static_baselines.py --quality-tolerance 0.05 \
    --out experiments/2026-04-26-static-baselines-v1/results.md
```

## Jobs

| Eval | Job ID | Status |
|---|---|---|
| dense_nano | _tbd_ | pending |
| dense_small | _tbd_ | pending |
| dense_large | _tbd_ | pending |

## Results — fill after analysis runs

| Strategy | TDLC BLER | UMA BLER | Avg BLER | Avg FLOPs % | Notes |
|---|---:|---:|---:|---:|---|
| Always-nano | _tbd_ | _tbd_ | _tbd_ | 20% | dense baseline |
| Always-small | _tbd_ | _tbd_ | _tbd_ | 43% | dense baseline |
| Always-large | _tbd_ | _tbd_ | _tbd_ | 100% | dense baseline |
| Per-profile static | _tbd_ | _tbd_ | _tbd_ | _tbd_ | best fixed rule |
| SNR-oracle cascade | _tbd_ | _tbd_ | _tbd_ | _tbd_ | upper bound (oracle SNR) |
| **exp26 learned MoE** | 0.867 | 0.937 | **0.902** | **56%** | for comparison |

## Decision Criteria

| Outcome | Story |
|---|---|
| **exp26 strictly dominates SNR-oracle cascade** | Learned router > any hand-rule with even oracle access. Strongest possible claim. |
| exp26 ≈ SNR-oracle cascade | Learned router matches oracle SNR rule. Honest framing: "learns the SNR-binning policy without oracle access." |
| SNR-oracle wins on FLOPs at same BLER | Oracle has more info — learned router approaches but doesn't reach. Still a defensible result. |
| Per-profile static wins | The whole channel-aware story is questionable. Reframe. |
