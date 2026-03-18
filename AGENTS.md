# Context for Agents: Compute-Aware MoE for Efficient 5G Neural Receivers

## 1. Core Idea & Problem Statement

We are building a **Compute-Aware Neural Receiver (NRX)** for the 5G physical layer.

- **The Problem:** Current state-of-the-art neural receivers are static, dense networks. They consume maximum computational power (FLOPs) to decode every single signal, which severely drains battery life on mobile devices, even when the signal is perfectly clear.
- **The Solution:** We are implementing a **Mixture-of-Experts (MoE)** architecture that dynamically scales compute. It routes clean signals to a lightweight expert and only activates a computationally heavy expert when it detects severe interference or deep fading.

## 2. The Baselines (What we are comparing against)

1. **Classical LMMSE (Linear Minimum Mean Square Error):**
    - _What it is:_ The traditional math-based standard.
    - _Compute:_ Extremely low FLOPs.
    - _Performance (BLER):_ Good in perfect conditions, fragile/poor in complex real-world channels (deep fading).
2. **Static Dense NRX (Neural Baseline):**
    - _What it is:_ A standard, non-routing neural network (e.g., ResNet/ViT).
    - _Compute:_ Extremely high FLOPs (100% active).
    - _Performance (BLER):_ Excellent, highly robust.

- **Our Goal:** Achieve the excellent BLER of the Dense NRX while driving average FLOPs down closer to the LMMSE baseline.

## 3. The MoE Architecture (Compute-Aware)

- **Input Data:** A 4D tensor `[batch, 16, freq, time]`. Represents an OFDM Resource Grid in a SIMO 1x4 configuration. Contains 8 channels for the received signal (Re/Im) and 8 channels for coarse Least Squares (LS) channel estimates (physical prior).
- **Task:** Binary classification to predict LLR logits for 16-QAM soft-bit decoding.
- **Components:**
    - **Feature Extractor:** Evaluates implicit channel quality.
    - **Channel-Aware Router:** Directs data to exactly one of 3 heterogeneous experts.
    - **Experts:** Tiny, Medium, Heavy (varying hidden layer depths/widths).
- **Routing Mechanics:**
    - _Training:_ Gumbel-Softmax (soft gating) for differentiability.
    - _Inference:_ Top-1 hard gating (argmax) to strictly bypass unselected experts for true FLOP/memory bandwidth savings.
- **Loss Function:** `L_total = BCEWithLogitsLoss + alpha * expected_FLOPs` (Compute-aware penalty).

## 4. Tech Stack & Implementation Details

- **Python Version:** strictly `3.10` to maintain compatibility between TensorFlow 2.15.0 (legacy Keras backend) and NVIDIA Sionna.
- **Data Generation:** NVIDIA Sionna (TensorFlow). Dynamic, on-the-fly simulation of 5G signals passed through AWGN/UMa/TDL-C channels. TF complex tensors are converted to PyTorch real/imag stacked tensors.
- **Model Framework:** PyTorch 2.x.
- **Experiment Tracking:** Weights & Biases (`wandb`).
- **Configurations:** Managed via Hydra.
