# In-Family OOD Generalisation — TDL-A / TDL-D / CDL-A Test Sets (v1)

## Question

Does the model trained on UMa+TDL-C generalise to *other* 3GPP channel
models (in-family OOD), or does it fail similarly to ray-traced OOD?

## Background

Currently we have only two OOD data points:
- In-distribution (UMa, TDL-C): exp26 0.902 BLER ✓
- Far OOD (DeepMIMO ASU ray-traced): exp26 0.992 BLER ✗ (catastrophic)

That's binary. Adding **in-family OOD** points fills the middle of the
generalisation spectrum and tells us *where the cliff is*:
- If TDL-A/D/CDL-A also fail → model brittle to small statistics changes
- If they work → model generalises within 3GPP family but fails on ray-traced
- Mixed → tells us specifically which channel features matter

## Channel profiles tested

| Profile | Description | What's different from training |
|---|---|---|
| TDL-A | Low delay spread, NLOS | Different multipath structure, much shorter delays |
| TDL-D | Rician fading, LOS-dominant | LOS regime — model trained on NLOS only |
| CDL-A | Clustered delay line, NLOS with spatial structure | Adds spatial correlation between antennas |

All three use the same 5G NR configuration (3.5 GHz carrier, 16-QAM, 1×4 SIMO,
14 OFDM symbols × 128 subcarriers, pilots at symbols 2 and 11) — only the
channel model differs.

## Configs

Code changes:
- `src/data/constants.py` — added TDLA, TDLD, CDLA to ChannelProfile enum
- `src/data/sionna_generator.py` — added dispatch for TDL-A/D and CDL channel
  generation (CDL needs antenna array setup like UMa)
- `conf/dataset/tdla.yaml`, `tdld.yaml`, `cdla.yaml` — dataset configs

Generation: 32k samples per profile (matches existing UMa/TDLC test set size)
with deterministic seed (base_seed=67 + TEST_SEED_OFFSET) so OOD comparisons
share the same RNG family as the existing data.

## Cluster

- `select=1:ncpus=8:ngpus=1:mem=32gb:scratch_ssd=40gb`, walltime 4h
- ~30-60 min generation per profile × 3 profiles = ~2-3h total

## Output

Files saved to `/storage/.../dense-v1/test/{tdla,tdld,cdla}.pt` alongside
existing `uma.pt` and `tdlc.pt`. Each ~6.2 GB. Existing test files untouched.

## Follow-up

Once test data is generated, eval exp26 + dense_large + LS-MRC on each new
profile. Same pipeline as standard eval — just point `--data-dir` at the new
profile files.

## Status

Generation job 19584191 in flight. Eval submissions to follow.
