# Function-Specialized MoE — Sink + Channel-Only + Decoder (v2, CLEAN)

## Why v2

v1 (job 19588534) ran with a **broken Hydra config**: `override /model: moe_nl`
+ `model.experts: {sink, channel_only, large}` deep-merged into moe_nl's
existing `{nano, small, large}` instead of replacing it. The instantiated
model had **5 experts** (nano + small + large + channel_only + sink).
v1 results (BLER 0.911 / real_flops 0.35) are valid but for a 5-expert
architecture, not the intended 3-expert design.

v2 fixes the config by introducing a dedicated `conf/model/moe_func.yaml`
that defines only `{sink, channel_only, large}` from scratch. No inheritance
from moe_nl's experts dict.

## Verified architecture (local instantiation)

```
expert_names: ['sink', 'channel_only', 'large']
expert_types: {'sink': 'sink', 'channel_only': 'channel_only', 'large': 'decoder'}
  sink:         0       params  (class _SinkExpert)
  channel_only: 109,608 params  (class _ChannelOnlyExpert — backbone + readout_channel)
  large:        369,868 params  (class _ExpertHead — full)
TOTAL:          566,575 params

expert_flops: tensor([0.0, 3.89e8, 1.32e9])  # sink, channel_only, large
```

For comparison: v1 (broken) had 692k total params; exp26 reference has 583k.

## Recipe (unchanged from v1)

- alpha = 2e-3 (FLOPs penalty)
- beta = 0.1 (load balance)
- batch_size = 128
- 12k training steps, seed = 67
- Asym warm start:
  - Stem from `dense_large_final20k_constant_lr_s67-55l1dpby:best`
  - channel_only from `dense_small_final20k_constant_lr_s67-kivdz4qu:best` (lenient
    load — bit-LLR head weights filtered out automatically)
  - large: random init
  - sink: no warm-start (no params)

## Predicted vs v1 result

v1 (5 experts, broken): BLER 0.911 at real_flops 0.35
v2 (3 experts, clean): expect comparable or better BLER at similar/lower FLOPs.
With sink as a true zero-FLOPs option and channel_only as the only "soft" expert,
the router's choice space is much cleaner.

If v2 matches or beats v1 → "function-specialized works" is a clean publishable claim.
If v2 is worse → 5-expert v1 was a confound, and the correct interpretation is that
the extra nano/small experts were doing something useful that wasn't captured by
the per-expert success-rate analysis.

## Cluster

`select=1:ncpus=8:ngpus=1:mem=32gb:scratch_ssd=40gb`, walltime 3h.
