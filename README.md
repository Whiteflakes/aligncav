# AlignCav

**Automated Fabry-Pérot Cavity Beam Alignment using Deep Learning**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A machine learning system that automates laser cavity alignment by combining CNN-based mode classification with Deep Q-Learning for intelligent mirror control. Developed as part of MS thesis research at IISER Pune.

## The Problem

Aligning a Fabry-Pérot optical cavity to achieve the fundamental TEM₀₀ mode is tedious and requires expert intuition. Misaligned cavities produce higher-order Hermite-Gaussian modes that reduce coupling efficiency and system performance.

## The Solution

AlignCav trains a CNN to recognize 121 distinct HG modes (orders 0-10 in both axes), then uses a DQN agent that learns to adjust four mirror motors based on real-time camera feedback until TEM₀₀ is achieved.

---

## Installation

```bash
git clone https://github.com/whiteflakes/aligncav.git
cd aligncav
uv sync            # recommended
# or: pip install -e .
```

## Usage

### Generate Hermite-Gaussian Modes

```python
from aligncav.simulation import HGModeGenerator

generator = HGModeGenerator(wavelength=632.8e-9, w0=1e-3, resolution=128)
tem00 = generator.generate_mode(0, 0)  # Fundamental Gaussian
tem21 = generator.generate_mode(2, 1)  # Higher-order mode with 2 horizontal, 1 vertical node
```

### Classify Beam Modes

```python
from aligncav.models import ModeClassifier
import torch

model = ModeClassifier(num_classes=121, input_size=128)
model.load_state_dict(torch.load("checkpoint.pth"))

# Predict mode from camera image
logits = model(image_tensor)
predicted_class = logits.argmax().item()
m, n = predicted_class // 11, predicted_class % 11
print(f"Detected: TEM{m}{n}")
```

### Train the Alignment Agent

```python
from aligncav.models import DQNAgent
from aligncav.training import RLTrainer

agent = DQNAgent(input_size=128, action_size=81)
trainer = RLTrainer(agent, env, episodes=2000)
trainer.train()
```

### CLI Tools

```bash
aligncav-train-classifier --config configs/classifier.yaml
aligncav-train-rl --config configs/rl.yaml
aligncav-simulate --modes 0,0 1,0 0,1 2,2 --output modes/
aligncav-predict --image beam.png --model checkpoint.pth
```

---

## How It Works

### Mode Classification

The CNN classifies camera images into 121 classes representing HG modes (m,n) where m,n ∈ [0,10]:

```
class_index = m × 11 + n
```

Architecture: 3 conv layers (32→64→128 channels) with batch norm, followed by FC layers (512→256→121).

### Reinforcement Learning

The DQN controls 4 stepper motors (2 mirrors × tip/tilt each). Action space: 81 discrete actions encoding {-1, 0, +1} steps for each motor:

```
action_tuple = itertools.product([-1, 0, 1], repeat=4)  # 3⁴ = 81 combinations
```

**Reward function** measures TEM₀₀ similarity:
```python
reward = 0.01 + 0.99 × correlation × exp(-50 × variance_mismatch)
```

where correlation is the L2-normalized inner product with an ideal Gaussian, and variance_mismatch penalizes incorrect beam width.

---

## Physics

### Hermite-Gaussian Modes

Electric field distribution for mode HG_mn:

$$E_{mn}(x,y) = \frac{w_0}{w(z)} H_m\left(\frac{\sqrt{2}x}{w_0}\right) H_n\left(\frac{\sqrt{2}y}{w_0}\right) \exp\left(-\frac{x^2+y^2}{w_0^2}\right)$$

where H_m are Hermite polynomials of order m, and w₀ is the beam waist.

### Cavity Alignment

Mirror tilts introduce linear phase gradients that couple power from TEM₀₀ into higher-order modes. The agent learns to minimize this coupling by iteratively adjusting mirror angles based on the observed mode pattern.

---

## Hardware Integration

For deployment on real optical setups:

| Component | Implementation |
|-----------|---------------|
| **Camera** | Raspberry Pi + MJPEG stream via HTTP |
| **Motors** | Arduino + stepper drivers, serial protocol |
| **Power Meter** | Thorlabs PM100A |

Simulated hardware interfaces are provided for development and testing.

---

## Citation

```bibtex
@mastersthesis{nandi2025aligncav,
  author = {Haraprasad Nandi},
  title  = {ML Techniques for Fabry-Pérot Cavity Alignment},
  school = {IISER Pune},
  year   = {2025}
}
```

## License

MIT
