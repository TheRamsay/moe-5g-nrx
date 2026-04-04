# Resource Presets

Source one of these presets before running a study helper or the interactive MetaCentrum launcher.

Example:

```bash
source experiments/resources/gpu-16gb.sh
bash experiments/2026-03-29-dense-baseline-v1/submit.sh qsub
```

The presets only export environment variables consumed by the existing helper scripts:

- `SELECT_RESOURCES`
- `WALLTIME` (optional suggestion)

Available presets:

- `gpu-12gb.sh` - older smaller GPU class, use only for smoke tests
- `gpu-16gb.sh` - recommended default for dense training/eval
- `gpu-46gb.sh` - larger A40/L40-style class when you want extra headroom

You can still override anything after sourcing a preset:

```bash
source experiments/resources/gpu-16gb.sh
WALLTIME=02:00:00 bash experiments/2026-03-29-dense-baseline-v1/submit.sh qsub
```
