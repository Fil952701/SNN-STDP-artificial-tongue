# Artificial Tongue – Spiking Neural Network with STDP, Eligibility Trace and Reinforcement Learning

This repository implements an **Artificial Tongue** using a **Spiking Neural Network (SNN)** in Brian2.  
The model continuously learns to recognize multiple *tastes* through **STDP (Spike-Timing-Dependent Plasticity)**, **eligibility traces**, **dopaminergic reinforcement learning**, **intrinsic homeostasis**, and **lateral inhibition (Winner-Take-All dynamics)**.  

It is conceived as a biologically-inspired simulation where each "taste neuron" corresponds to a different basic flavor, and the network evolves online while exposed to both pure and mixed taste stimuli.

---

## ✨ Main Features

- **Conductance-based LIF neurons** with adaptive thresholds and intrinsic homeostasis.  
- **STDP with eligibility traces** for temporally precise plasticity.  
- **Reinforcement learning with dopamine signals** (positive and negative rewards).  
- **Column normalization** and weight scaling for stable synaptic growth.  
- **Lateral inhibition (WTA)** for competition between neurons.  
- **Always-on simulation loop**: the system continuously processes stimuli without explicit epochs.  
- **Multi-taste learning**: supports recognition of single tastes and mixtures.  
- **Unknown taste detection**: outputs *UNKNOWN* when neurons do not spike consistently.  
- **Detailed metrics**: precision, recall, F1-score, IoU, Jaccard similarity, and confusion matrix.  
- **Visualization utilities**: plots for spikes, synaptic weight evolution, and membrane potentials.

---

## 🧪 Taste Representation

The system defines **8 taste classes**:

1. SWEET 🍬 → *"Ouh... yummy!"*  
2. BITTER 🍵 → *"So acid!"*  
3. SALTY 🧂 → *"Need water... now!"*  
4. SOUR 🍋 → *"Mehhh!"*  
5. UMAMI 🍄 → *"So delicious!"*  
6. FATTY 🍔 → *"Oh, I'm a big fat boy!"*  
7. SPICY 🌶️ → *"I'm a blazing dragon!"*  
8. UNKNOWN ❓ → *"WTF!"* (catch-all when no neuron dominates)

The network is trained on **pure stimuli** and **mixtures** (e.g., *SWEET + SOUR*, *BITTER + UMAMI + SPICY*), and then tested with previously unseen combinations.

---

## ⚙️ Key Parameters

- **Simulation time step**: `0.1 ms` (high temporal precision).  
- **Intrinsic homeostasis target firing**: `~50 Hz`.  
- **STDP time constant**: `30 ms` with eligibility trace decay `50 ms`.  
- **Training duration per stimulus**: `1000 ms`.  
- **Test duration per stimulus**: `500 ms`.  
- **Repetitions per taste**: 10.  

Hyperparameters are fully configurable (see the Python script).

---

## 📊 Training and Testing Pipeline

1. **Training phase**  
   - Pure tastes and mixtures presented with Poisson spike trains.  
   - Weights updated via STDP + eligibility + dopamine (reward/punishment).  
   - Column normalization stabilizes synaptic scaling.  

2. **Test phase**  
   - STDP frozen, homeostasis off.  
   - Neurons respond to pure and mixed test stimuli.  
   - Classification thresholds are computed using Gaussian stats, quantiles, and EMA.  
   - Metrics (accuracy, IoU, Jaccard) are reported.  

3. **Visualization**  
   - Spike raster plots.  
   - Weight trajectories for diagonal synapses.  
   - Membrane potentials across neurons.  

---

## 📈 Example Outputs

- **Training logs**: progress bar with % completed, elapsed time, ETA, and neuron reactions.  
- **Test logs**: expected vs. predicted taste(s) for each mixture, with exact hit/miss reporting.  
- **Final report**: accuracy, per-class precision/recall/F1, IoU, macro/micro metrics, and weight changes during test.  

---

## 🔬 Research Context

This project is part of my independent research in **neuromorphic computing** and **Spiking Neural Networks**.  
The artificial tongue is a prototype for more general artificial sensory systems that can:  

- Learn **online** in a continuous environment.  
- Distinguish **overlapping stimuli**.  
- Exhibit **biologically inspired plasticity** with reinforcement learning.  
- Integrate **lateral inhibition** for exclusivity in decision-making.  

Potential applications extend to neuromorphic AI, robotics, and bio-inspired sensory processing.

---

## ⚡ Installation

You can set up the environment either with **pip** or **conda**:

[![Pip](https://img.shields.io/badge/install%20with-pip-blue?logo=python)](https://pip.pypa.io/)
[![Conda](https://img.shields.io/badge/install%20with-conda-green?logo=anaconda)](https://docs.conda.io/)

### Option 1 – pip
```bash
pip install -r requirements.txt
```

### Option 2 – conda
If you prefer using conda, you can create the environment directly from the provided `environment.yml`:

```bash
conda env create -f environment.yml
conda activate artificial-tongue-snn
```

---

## 👤 Author

**Filippo Matteini** – Pianist, AI Engineer, and Neuromorphic Computing Researcher  

- 🎹 YouTube (Dexteris): [@dexteris27](https://www.youtube.com/@dexteris27)  
- 💼 LinkedIn: [Filippo Matteini](https://www.linkedin.com/in/filippo-matteini-29554a355)  
- 🖥️ GitHub: [Fil952701](https://github.com/Fil952701)  

---

## 📜 License

This project is released under the **MIT License**.  
Feel free to use, modify, and build upon this work with proper attribution.  

---
