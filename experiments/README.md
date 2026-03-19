# Experiments Directory

One folder per experiment batch. Self-contained, no dependencies on conf/.

## Creating a New Experiment Batch

```bash
# 1. Create dated folder
cd experiments
mkdir 2024-03-20-batch-name

# 2. Copy template
cd 2024-03-20-batch-name
cp ../template.sh submit.sh

# 3. Create experiment configs in conf/experiment/
#    e.g., conf/experiment/exp01_variant1.yaml

# 4. Submit
bash submit.sh
```

## Structure

```
experiments/
├── README.md                    # This file
├── 2024-03-18-example/          # Example batch
│   ├── README.md                # What was tested, results
│   └── submit.sh                # Submission script
└── template.sh                  # Copy for new batches

conf/experiment/                 # Experiment configs
├── exp01_baseline.yaml
└── exp02_variant.yaml
```

## To Git or Not to Git?

**Commit:**
- Paper-related experiments
- Final sweeps
- Anything you want to reproduce later

**Don't commit (.gitignore):**
- Quick debugging runs
- Failed experiments
- Parameter tuning noise

## Guidelines

1. **One hypothesis per batch** - don't mix unrelated changes
2. **Document in batch README** - what worked, what didn't
3. **Simple flat configs** - no complex Hydra overrides
4. **Use descriptive names** - `2024-03-18-batch-norm` not `exp01`

## Current Batches

- `2024-03-18-example/` - Testing router temperature (0.3, 1.0, 2.0)
