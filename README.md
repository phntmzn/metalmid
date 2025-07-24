# 🎼 eva_ascii — GPU-Accelerated MIDI Generator with Metal

**eva_ascii** is a high-performance, Metal-powered music generation system that produces hundreds of layered, rhythmically complex MIDI files in parallel.  
It uses macOS’s Metal framework to compute musical velocity and dynamics in real-time, then composes full MIDI sequences for use in production, sampling, or further synthesis.

---

## ⚡ Features

- 🔁 Generates **200+ unique MIDI files**
- 🎸 Uses **Metal GPU shaders** to compute **velocities** in parallel
- 🧠 Algorithmically generates chord progressions and trap/punk hybrid drum rhythms
- 🚀 Multi-core processing via Python’s `concurrent.futures`
- 🧱 Outputs `.mid` files to `~/Desktop/eva_ascii/`

---

## 🧰 Requirements

- macOS with **Metal support**
- Python 3.9+
- Metal toolchain (`xcrun`, `metal`, `metallib`) available via Xcode Command Line Tools

### Python dependencies:

```bash
pip install numpy tqdm midiutil
