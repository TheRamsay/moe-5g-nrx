# HF Loader Confirm V1

Goal: confirm the best HuggingFace loader setting from the first sweep using the
same GPU class for both runs.

## Why This Exists

The initial sweep mixed multiple GPU families, so worker/prefetch settings were
confounded with hardware differences. This follow-up compares only:

- `w2_p1` = `hf_num_workers=2`, `hf_prefetch_factor=1`
- `w2_p2` = `hf_num_workers=2`, `hf_prefetch_factor=2`

Both jobs are pinned to the same MetaCentrum GPU class:

- `gpu_cap=sm_89`
- `gpu_mem=46068mb`
- `cl_alfrid=True`

## Run Design

- Model: `exp05_dense_capacity_large`
- Dataset: `Vack0/moe-5g-nrx`
- Training profile: `mixed`
- Validation: disabled
- Max steps: 300
- Logging cadence: every 25 steps

## Launch

```bash
bash experiments/2026-04-10-hf-loader-confirm-v1/submit.sh qsub
```

## Selection Criterion

Winner is the run with the lower walltime on the same GPU class, assuming both
finish successfully and land in the same training-loss range.
