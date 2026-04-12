# DeepMIMO OOD Dataset v1

Generate a DeepMIMO-based OOD dataset for evaluation in Arrow format and log it as a W&B dataset artifact.

## Goal

- Produce exactly `32,768` OOD samples
- Save to `data/test/deepmimo` (Arrow dataset directory)
- Log artifact as `dataset-test-deepmimo:latest`
- Reuse with `scripts/evaluate.py` via artifact download

When the selected DeepMIMO slice has fewer unique channels than requested samples,
the generator resamples channels with replacement to preserve exact sample count.

## Usage

From repo root:

```bash
SCENARIO=asu_campus_3p5 bash experiments/2026-04-12-deepmimo-ood-dataset-v1/submit.sh print
SCENARIO=asu_campus_3p5 bash experiments/2026-04-12-deepmimo-ood-dataset-v1/submit.sh local
SCENARIO=asu_campus_3p5 bash experiments/2026-04-12-deepmimo-ood-dataset-v1/submit.sh qsub
```

Optional overrides:

```bash
SCENARIO=asu_campus_3p5 \
DATASET_FOLDER=./data/deepmimoV3 \
NUM_SAMPLES=32768 \
PROFILE_NAME=deepmimo \
OUTPUT_DIR=data \
SPLIT=test \
EXTRA_ARGS='generation.deepmimo.download_if_missing=true' \
bash experiments/2026-04-12-deepmimo-ood-dataset-v1/submit.sh qsub
```

## Evaluation

```bash
uv run python scripts/evaluate.py \
    evaluation.checkpoint_artifact=knn_moe-5g-nrx/moe-5g-nrx/model-<exp-name>-<run-id>:best \
    evaluation.profiles=[deepmimo]
```

Or include ID + OOD in one run:

```bash
uv run python scripts/evaluate.py \
    evaluation.checkpoint_artifact=knn_moe-5g-nrx/moe-5g-nrx/model-<exp-name>-<run-id>:best \
    evaluation.profiles=[uma,tdlc,deepmimo]
```
