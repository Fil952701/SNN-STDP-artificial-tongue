# 🧠 Artificial Tongue SNN – Neuromorphic Taste Recognition

This project implements a biologically-inspired **Spiking Neural Network (SNN)** designed to simulate an **artificial tongue** capable of recognizing and classifying **multiple tastes**.  
It integrates **STDP plasticity**, **dopaminergic reinforcement learning**, **multi-sensory encoding**, and **competitive cross-synaptic dynamics** to mimic neural taste processing.

The simulation is implemented using the **Brian2** library in Python and is designed for both **supervised** and **unsupervised** learning experiments.

---

## 🚀 Features

- **Multi-taste SNN architecture**  
  Simulates recognition of multiple basic tastes:  
  `SWEET, BITTER, SALTY, SOUR, UMAMI, FATTY, SPICY, UNKNOWN`
- **STDP Plasticity**  
  Implements Hebbian-based Spike-Timing-Dependent Plasticity for continuous learning.
- **Dopaminergic Reinforcement Learning**  
  Simulates biologically-inspired reward-based learning to strengthen correct taste associations.
- **Competitive Cross-Synaptic Connections** *(planned)*  
  Introduces competition between neurons to make a **specialist neuron** emerge for each taste.
- **Winner-Takes-All (WTA) Inhibition** *(planned)*  
  Prevents multiple neurons from dominating simultaneously by lateral inhibition.
- **Poisson-based Sensory Input Encoding**  
  Encodes taste intensity via probabilistic spiking input streams.
- **Support for Multi-Taste Food Recognition** *(next step)*  
  Will classify complex foods composed of multiple combined tastes.
- **Planned GUI in Pygame**  
  Interactive interface to visualize spikes, weights, and taste classification in real-time.


---

## 🔬 Methodology

### 1. **Sensory Encoding**
Each taste stimulus is encoded into spike trains using a **PoissonGroup** in Brian2:
- High taste intensity → higher spiking rate (e.g., 250 Hz).
- Allows simulating noisy biological input like real taste buds.

### 2. **Spiking Neural Network Architecture**
- Input layer: 8 sensory neurons (1 per taste).
- Hidden layer: Optional for competitive learning experiments.
- Output layer: 8 taste-class neurons.
- Dense synaptic connections between input and output neurons.

### 3. **Learning Mechanisms**
- **STDP**:  
  Strengthens synapses based on spike timing (Hebbian learning).
- **Dopamine-based Reinforcement**:  
  When the network correctly predicts a taste → reward signal boosts associated weights.
- **Punishment Mechanism** *(planned)*:  
  Incorrect predictions will trigger synaptic decay to improve future discrimination.

### 4. **Multi-Taste Recognition**
- Allows simultaneous activation of multiple input neurons.
- Uses **cross-synaptic dynamics** to see which neuron best specializes in each taste.
- Planned integration of **winner-takes-all lateral inhibition** to avoid confusion between tastes.

---

## 📊 Results & Progress

| **Feature**                  | **Status** | **Notes** |
|----------------------------|------------|-----------|
| Single-taste recognition   | ✅ Working |
| Multi-taste input          | ✅ Working |
| Dopamine-based RL         | ✅ Working |
| Synaptic punishment      | ✅ Working |
| Cross-synaptic learning   | 🟡 In progress |
| Winner-Takes-All          | 🟡 Planned |
| Food recognition (multi) | 🟡 Planned |
| Pygame GUI               | 🟡 Planned |

---

## 🧠 Technologies Used

- **Python 3.11**
- [**Brian2**](https://brian2.readthedocs.io/) → SNN simulation
- **NumPy / Matplotlib** → Data handling & visualization
- **Pygame** *(planned)* → Interactive GUI

---

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/artificial-tongue-snn.git
cd artificial-tongue-snn

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt


## 📚 References
Gerstner, W., Kistler, W. M., Naud, R., & Paninski, L. (2014). Neuronal Dynamics: From Single Neurons to Networks and Models of Cognition.
Brian2 Documentation: https://brian2.readthedocs.io

👤 Author
Filippo Matteini
AI Engineer & Neuromorphic Computing Researcher
🌐 GitHub
🔗 LinkedIn
🎹 Dexteris YouTube

