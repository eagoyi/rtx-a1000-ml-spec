# Infrastructure Verification: Ubuntu + NVIDIA ML Specification Seat

This repository acts as an automated verification suite confirming that the **Ubuntu + NVIDIA x86_64** hardware profile fulfills all local compute, pipeline, and container profiling criteria required for development.

## Hardware Profile Under Audit

* **Host Operating System:** Ubuntu Linux
* **Target Graphics Processor:** NVIDIA RTX A1000 Laptop GPU (6GB Dedicated VRAM)
* **Microarchitecture:** Ampere Backend (Compute Capability 8.6, Native Tensor Cores)
* **Driver / CUDA Environment:** NVIDIA-SMI 595.91.07 / CUDA 13.2 Baseline

## Project Pipeline Structure

* `pytorch_rl_pipeline.py`: Validates foundational model pretraining and registers interactive Man-in-the-Loop (MITL) hooks utilizing PyTorch's native CUDA layer.
* `tensorflow_rl_pipeline.py`: Exercises identical execution parameters under the TensorFlow runtime engine utilizing auto-growth memory structures.
* `unified_dashboard.py`: Evaluates compiled local performance outputs and aggregates computing speeds alongside peak system VRAM metrics.

## Prerequisites

1. Install the Ubuntu Python virtual environment manager

 ```bash
sudo apt update
sudo apt install python3-venv python3-pip -y
```

2. Create and activate your virtual environment

Create a virtual environment named 'venv'

 ```bash
python3 -m venv venv
```

Activate it (you will see '(venv)' appear at the beginning of your terminal prompt)

```bash
source venv/bin/activate
```

3. Install the GPU-compatible frameworks

Run this specific command to pull down PyTorch with CUDA acceleration enabled, alongside TensorFlow:

```bash
pip install --upgrade pip
pip install torch tensorflow numpy
```

------------------------------

run Your Pipeline

Now that the dependencies are installed inside your virtual environment, run your execution sequence:

python3 pytorch_rl_pipeline.py
python3 tensorflow_rl_pipeline.py
python3 unified_dashboard.py

## Execution Instructions

1. **Provision Environment Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Execute PyTorch Run Profile:**

   ```bash
   python3 pytorch_rl_pipeline.py
   ```

3. **Execute TensorFlow Run Profile:**

   ```bash
   python3 tensorflow_rl_pipeline.py
   ```

4. **Launch Unified Metrics Dashboard Evaluation:**

   ```bash
   python3 unified_dashboard.py
   ```

## Strategic Engineering Parameters

* **Zero Strategy Disruption:** Zero codebase bifurcation required. Modern abstract definitions easily handle system device routing via single-line checks.
* **Production Parity Alignment:** Native execution on Ubuntu completely removes emulation layers, providing an exact mirror of standard target enterprise cloud architectures.
