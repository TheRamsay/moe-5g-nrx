# KNN Project Proposal: Compute-Aware MoE for Efficient 5G Neural Receivers
**Team:** Dominik Huml (xhumld00), Jakub Kontrík (xkontr02), Martin Vaculík (xvaculm00)
**Repository:** https://github.com/TheRamsay/moe-5g-nrx
### 1. Problem & Goal
Standard 5G Neural Receivers (NRX) are static, dense networks that consume maximum computational power (FLOPs) regardless of signal quality, severely draining mobile batteries. 
**Our Goal:** Inspired by the recent MEAN architecture (van Bolderik et al.) and standard-compliant dense receivers (Wiesmayr et al.), we aim to develop a Compute-Aware NRX using a Mixture-of-Experts (MoE) architecture. It will dynamically route clean signals to lightweight experts, activating computationally heavy experts only during deep fading or severe interference.
### 2. Architecture & Routing (SIMO 1x4 OFDM)
* **Input:** A 4D tensor `[batch, 16, freq, time]` containing 8 channels for the received signal and 8 channels for the coarse Least Squares (LS) channel estimate (physical prior).
* **Model Structure:** A shared Feature Extractor evaluates channel quality and feeds a Channel-Aware Router, directing data to one of 3 heterogeneous experts (Tiny, Medium, Heavy).
* **Routing:** 
	* *Training:* Gumbel softmax (soft gating) maintains differentiability for backpropagation.
	* *Inference:* Top-1 hard gating (argmax) strictly selects one expert, bypassing others to guarantee real world FLOP and memory bandwidth savings.
* **Loss Function:** Binary classification for 16-QAM LLR logits. Optimized via BCEWithLogitsLoss with a compute-aware penalty: $L_{total} = BCE + alpha * {expected}_{FLOPs}$
### 3. Datasets & Methodology
* **Training (Dynamic):** On the fly NVIDIA Sionna simulations using standard 3GPP channel models (UMa, TDL-C) to prevent static dataset overfitting.
* **Evaluation (Deterministic):** DeepMIMO dataset (3D ray-tracing of urban scenarios) to verify out of distribution robustness.
### 4. Implementation Plan
* **Phase 1 (Baselines):** Set up Sionna pipeline. Evaluate classical LMMSE and static Dense NRX to establish reference BLER and FLOP metrics.
* **Phase 2 (MoE Development):** Implement the experts and router. Test a two stage training strategy (pretrain experts, then freeze them to train the router).
* **Phase 3 (Evaluation):** Plot a Pareto Frontier (BLER vs. Average FLOPs) to definitively prove computational savings against the dense baseline.