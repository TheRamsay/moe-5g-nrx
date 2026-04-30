# DeepMIMO O1_3p5 OOD Eval (v1)

## Question

Was the DeepMIMO ASU campus failure (BLER ~0.99) **specific to ASU's complex
geometry**, or does our model fail on **all ray-traced outdoor scenarios**?

## Background

We tested asu_campus1 earlier — all models (dense_large, exp26, exp31) failed
catastrophically (~0.99 BLER). Brief few-shot fine-tune (500 steps, 2k samples)
didn't help.

ASU campus is one of DeepMIMO's harder scenarios — complex urban geometry,
many reflections. Testing **O1_3p5** (Outdoor Street Canyon at 3.5 GHz) is
the key diagnostic:
- Same carrier frequency (3.5 GHz) — no frequency-mismatch confound
- Simpler geometry (single street vs full ASU campus)
- Most popular DeepMIMO benchmark in the literature

If O1 also fails → confirms fundamental synthetic-vs-geometric gap.
If O1 works → ASU was specifically pathological, model can do *some*
ray-traced data.

## Configs

Pipeline:
1. `submit_generate.sh` — generate 32k test samples from O1_3p5 scenario.
   Uses `scripts/generate_deepmimo_dataset.py` (existing) with
   `generation.deepmimo.scenario=O1_3p5`.
2. `submit_eval_exp26.sh` — eval52 — exp26 on the new test data
3. `submit_eval_dense_large.sh` — eval53 — dense_large on the new test data
4. `submit_eval_lmmse.sh` — LMMSE LS-MRC on the new test data (classical reference)

## Cluster

- Generate: 8 cpus, 32 GB RAM, 20 GB scratch_ssd, 3h walltime
- Each eval: 4 cpus, 1 GPU, 16 GB RAM, 1h walltime

## Data setup

`O1_3p5.zip` (3 GB) uploaded from local Downloads to
`/storage/.../moe-5g-datasets/deepmimo-train/O1_3p5.zip`. Needs unzipping
before generation:

```bash
ssh skirit.metacentrum.cz
cd /storage/brno2/home/ramsay/moe-5g-datasets/deepmimo-train
unzip -q O1_3p5.zip
```

Then submit `submit_generate.sh`.

## Status

Upload finished 2026-04-30. Awaiting unzip + generation submission.

## Expected outcomes

- **O1 also fails (~0.99 BLER)** → fundamental synthetic-vs-geometric gap,
  ASU isn't special.
- **O1 partially works (e.g., 0.5–0.9 BLER)** → ASU was specifically hard;
  some outdoor ray-traced is decodable.
- **O1 works as well as in-distribution (~0.90)** → unlikely but would mean
  ASU was pathological; we'd need to investigate why.
