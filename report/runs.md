# Run-ID and JSON traceability for the report

This file maps every numerical claim in `projekt-01-kapitoly-chapters-en.tex`
to the underlying Weights & Biases run or analysis JSON, so the run IDs do
not clutter the report body itself.

All test-set numbers use the locked `dense-v1` test split (32,768 samples
per profile).

## Section 3 — Training Recipe

| Claim in report | Source (W&B / file) |
|---|---|
| Phase 1 BLER 0.926 / 48% FLOPs | `xdb6fzll` (train `2op33pak`) |
| Phase 2 BLER 0.879 / 100% FLOPs | `experiments/2026-04-11-moe-phase2-v1/` (cluster job logs) |
| Switch-aux at weight 1e-3 collapses to 100% large | `j6vwy0hu` (train) |
| Switch-aux + capacity penalty 8-mechanism sweep | `experiments/2026-04-30-anti-collapse-sweeps-v1/` |
| Routing-trajectory 3-paradigm figure | `docs/figures/routing_trajectories_collapse_modes.png` (script `scripts/plot_routing_trajectories.py`) |
| Routing-trajectory 8-mechanism figure | `docs/figures/routing_trajectories_anti_collapse.png` |
| Linear-probe figure | `docs/figures/router_mechanism_linear_probing.png` |
| Large-warmup stabilisation (3-seed collapse) | `experiments/2026-04-26-moe-largewarmup-v1/` |
| β-warmup stabilisation (mean BLER 0.936±0.043) | `experiments/2026-04-26-moe-betawarmup-v1/` |
| Alpha sweep test eval — exp24 (α=5e-4) | `002cwsy2` |
| Alpha sweep test eval — exp25 (α=1e-3) | `5jswm490` (train `3xzxkddv`) |
| Alpha sweep test eval — exp26 (α=2e-3) | `2zboo1rh` (train `t6lkdep2`) |
| Alpha sweep test eval — exp27 (α=5e-3) | `dh4x0qmu` |
| Random-router ablation, BLER 0.968 | `ag3qbw52` (train `cd2w6l31`) |
| Linear probes R² values | `docs/figures/router_mechanism_linear_probing.json` |
| Symmetric sweep — cold-small (exp56) | `87yjni5r` |
| Symmetric sweep — cold-nano (exp57) | `z837iapf` |
| Multi-seed bimodality — s32 (exp28) collapse | `ywvyzlia` |
| Multi-seed bimodality — s42 (exp29) reproduction | `121ex9e6` |
| 100k+α=1e-3 recovery (exp60) | `k0tjo3m2` (train `yscpku2h`) |
| Phase 1 reference (cold-start) | `xdb6fzll` (train `2op33pak`) |

## Section 4 — Training-Scaffold Finding

| Claim in report | Source |
|---|---|
| Per-expert block success rate | `docs/figures/router_mechanism_success_rate.json` |
| Per-expert specialisation figure (SNR distributions, BLER curves) | `docs/figures/router_mechanism_expert_specialization.png` |
| Middle-expert counterfactual on small's samples | `docs/figures/middle_expert_results.json` |
| Mode B α-sweep table | `docs/figures/inference_mask_exp{25,26,27}.json` |
| Drop-small training ablation (exp41) | `lsn2jr1k` (train) |
| Drop-nano training ablation (exp31) | `nyvfkxl0` (train `5c0kshem`) |
| Sink + channel_only + large (exp61 v2) | `6g5tshu1` |
| Sink + small + large (exp64) | `p0vetpmg` |
| Sink + nano + large (exp65) | `cia6p3i8` |

## Section 5 — Honest Scope

| Claim in report | Source |
|---|---|
| LMMSE LS-MRC test BLER + per-SNR | `lmmse_snr20_results.json` (job 19583606) |
| LMMSE Genie-MRC test BLER | `genie_mrc_eval_results.json` (job 19583116) |
| Single-antenna test BLER | `single_ant_eval_results.json` (job 19583117) |
| dense_nano test BLER | `bx7hylp6` |
| dense_small test BLER | `8haq7zuz` |
| dense_large test BLER | `gpmfhn6k` |
| Per-SNR TDL-C + UMa waterfall (exp26) | `19583604` snr_binned.table.json (eval46) |
| Per-SNR TDL-C + UMa waterfall (dense_large) | `19583605` snr_binned.table.json (eval47) |
| Per-SNR UMa LMMSE LS-MRC | `19583606` run.log (text-format per-SNR table) |
| 3GPP in-family OOD (TDL-A, TDL-D, CDL-A) — dense_large | `0kyguffw` |
| 3GPP in-family OOD — exp26 | `h25pbo10` |
| Neural-vs-LMMSE crosstab | `docs/figures/neural_vs_lmmse_results.json` |
| DeepMIMO ASU OOD — dense_large zero-shot | `gpmfhn6k` (asu_campus1 fields) |
| DeepMIMO ASU OOD — exp26 zero-shot | `bpimn1to` |
| DeepMIMO ASU OOD — dense_large few-shot fine-tune | `t4yo37am` (train `go74dlm7`) |
| DeepMIMO ASU OOD — exp26 few-shot fine-tune | `kjc12s5p` (train `9t2wyyus`) |
| LMMSE on DeepMIMO O1 ray-traced (0.976) | `experiments/2026-04-30-deepmimo-o1-ood-v1/` |
| SNR-oracle cascade baseline | `experiments/2026-04-26-static-baselines-v1/` + `scripts/analyze_static_baselines.py` |
| Explicit SNR-input ablation (exp38, collapsed) | `experiments/2026-04-29-moe-snr-input-v1/` |
| Wall-clock latency benchmark (1.7× slower on real data) | `experiments/2026-04-29-latency-exp26-real-v1/` |
| PCA OOD overlay figure | cluster job 19654175, `docs/figures/pca_ood_overlay.{pdf,png,npz}`, script `scripts/visualize_ood_pca.py` |

## Section 2 — Architecture

| Claim | Source |
|---|---|
| Total parameter count 582,655 | exp26 W&B summary `model/num_parameters` |
| FLOPs per expert (320M / 695M / 1604M) | `src/utils/compute.py` formula, verified vs `max_flops` buffer to 0.01% |
| Dense baseline parameter counts (89,892 / 168,324 / 449,540) | `bx7hylp6` / `8haq7zuz` / `gpmfhn6k` `model/num_parameters` |
