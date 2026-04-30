# Classical LMMSE / MRC Receiver Baselines (v1)

## Question

How does the compute-aware MoE compare to **standard non-neural baselines**?
Teacher specifically requested comparison against "default solution".

## Background

NRX literature compares neural receivers against each other but rarely
publishes head-to-head against classical 5G receivers. We add three classical
baselines forming a ladder of assumed knowledge:

1. **Single-antenna detection** — naive lower bound, uses only 1 of 4 receive
   antennas. No diversity gain.
2. **LS-MRC** — realistic deployable classical: pilot-based LS channel
   estimate + MRC combining + max-log 16-QAM demodulation.
3. **Genie-MRC** — upper bound: same MRC pipeline but uses TRUE channel from
   Sionna instead of LS estimate. Not deployable (needs perfect channel info).

## Configs

Implementation: `src/baselines/lmmse.py` (pure PyTorch, vectorised, no
trainable parameters). Eval: `scripts/evaluate_lmmse.py` matches metrics
format of `evaluate.py`.

| Submit | Mode | Description |
|---|---|---|
| `submit.sh` | `ls_mrc` | Pilot LS + 4-antenna MRC (realistic) |
| `submit_genie.sh` | `genie_mrc` | True channel + 4-antenna MRC (upper bound) |
| `submit_single_ant.sh` | `single_ant` | Pilot LS + 1-antenna only (naive lower bound) |

All three: same test set (UMa+TDLC, 32k samples each), 7 SNR bins.

## Cluster

- Resources: `select=1:ncpus=4:ngpus=1:mem=16gb:scratch_ssd=10gb`
- Walltime: 1h (eval is fast — ~30 sec compute per profile)

## Results

| Baseline | UMa BLER | TDLC BLER | Avg BLER | TDLC waterfall (15-20dB) |
|---|---:|---:|---:|---:|
| Single-antenna | 0.992 | 0.998 | 0.995 | 0.984 |
| **LS-MRC** | **0.939** | **0.861** | **0.900** | **0.155** |
| **Genie-MRC** | **0.908** | **0.800** | **0.854** | **0.027** |

Compared to neural:

| Model | Avg BLER | FLOPs % |
|---|---:|---:|
| LS-MRC (realistic classical) | 0.900 | ~0.005% |
| **exp26 MoE** | 0.902 | 56% |
| dense_large (full neural) | 0.901 | 100% |
| Genie-MRC (oracle classical) | 0.854 | ~0.005% |

## Key findings

1. **Antenna diversity gain is huge:** 1→4 antennas drops BLER from 0.995 →
   0.900 on average.
2. **Channel estimation is the classical bottleneck:** LS-MRC vs Genie-MRC
   gap of 5pp average (and 13 pp at high TDLC SNR) is due to LS estimate
   error, not detection.
3. **Neural beats realistic classical in the waterfall:** at TDLC SNR=14,
   dense_large ~0.75 vs LS-MRC 0.864 (~10 pp improvement). On average they
   tie because low-SNR bins where everything fails dominate.
4. **Neural loses to oracle classical at high SNR:** Genie-MRC 0.027 vs
   dense_large ~0.085 at TDLC 18 dB. Neural can't beat optimal classical
   with perfect channel.

## Implications for the consultation

The Pareto plot now has **three classical reference points** (single-ant,
LS-MRC, Genie-MRC) plus our neural results. Direct response to the teacher's
"add a standard baseline" feedback. Story: neural earns its compute in the
waterfall region where capacity matters; classical with imperfect channel
fails there.
