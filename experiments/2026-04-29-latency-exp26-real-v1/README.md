# Wall-Clock Latency on Real Test Data — exp26 + dense baselines (v1)

## Question

Does the 56% FLOPs reduction of exp26 translate to wall-clock speedup on real
OFDM test data, or does the routing dispatch overhead negate it?

## Background

Earlier benchmark (job 19473464) measured **synthetic-input** latency on
RTX PRO 6000 Blackwell:
- dense_large: 2.92 ms/batch
- exp26 MoE:   1.51 ms/batch (1.93× speedup)

But: synthetic Gaussian input might cause router to behave differently than
on real channel data. This study runs the same benchmark on **actual UMa+TDLC
test slots** to check the speedup holds with real routing distributions.

## Configs

`scripts/benchmark_latency.py` extended with `--models` filter and `--data-dir`
flag for cycling through real test batches. Two submit modes:
- `submit.sh` — all 4 models (dense_nano/small/large + exp26) on real data
- (initial single-model variant was used in the first attempt; superseded)

## Cluster

- Resources: `select=1:ncpus=8:ngpus=1:mem=32gb:scratch_ssd=40gb`
- Walltime: 4h (PTX JIT for MoE on cold node can take 30+ min)

## Result (job 19549167, NVIDIA A40)

| Model | synth ms/batch | real ms/batch |
|---|---:|---:|
| dense_nano | 1.05 | 1.06 |
| dense_small | 2.09 | 2.07 |
| dense_large | 3.32 | 3.28 |
| **exp26 MoE** | **1.93** | **5.53** |

**Honest finding:** exp26 is **1.67× SLOWER** than dense_large on real data.
The 56% FLOPs reduction does not translate to wall-clock at batch=64 because
the hard-top-1 routing dispatches 3 sequential sub-batches (mask indexing +
scatter), and that overhead dominates the FLOPs savings.

Dense models: real ≈ synthetic (no dispatch overhead). MoE: real >> synthetic.

## Implications

- Pareto frontier is reported in **FLOPs** (hardware-agnostic), NOT latency
- Wall-clock speedup requires production-grade dispatch kernels (Mixtral, vLLM)
- Future work: re-benchmark with efficient sparse-MoE dispatch
