# 🦙 LLaMA 3.2 1B – Fine-Tuning en Español (QLoRA + Unsloth + HPC-UCR)

Python · PyTorch 2.7.1 · Transformers · Unsloth · QLoRA · CUDA 12.8 · HPC-UCR  


## 🧠 Overview

This project adapts the **LLaMA 3.2 1B Instruct** model to **Spanish** using efficient fine-tuning with **QLoRA (4-bit)** through the **Unsloth** library, executed on the **HPC-UCR cluster**.

The objective of this work is to build a small but specialized model capable of responding to **academic instructions in Spanish**, using a dataset generated from PDFs processed to JSONL.

This project demonstrates how it is possible to train language models in university infrastructures using modern optimization and GPU consumption techniques.

---

## ⚙️ Model Summary

| Component          | Description |
|-------------------|-------------|
| Framework         | PyTorch **2.7.1**, TorchVision 0.22.1, TorchAudio 2.7.1 |
| Transformer Stack | HuggingFace Transformers |
| Training Strategy | QLoRA (4-bit) + Low Rank Adapters |
| Base Model        | `meta-llama/Llama-3.2-1B-Instruct` |
| Sequence Length   | 4096 tokens |
| Optimizer         | AdamW |
| Dataset Format    | JSONL (instruction, input, output) |
| Infraestructura   | HPC-UCR GPU partition (A100 80GB) |
| Scheduler         | Warmup + Cosine Decay |
| Evaluation        | Eval Loss & Perplexity |

---

## 📊 Results

From `training_summary_full.json`:

| Metric      | Value |
|-------------|--------|
| Eval Loss   | **3.08** |
| Perplexity  | **21.74** |
| Epochs      | **60** |
| Runtime     | **3 blocks × 6 hours** (GPU HPC-UCR) |

---

## 🗂️ Project Structure

```bash
llama32_qlora/
├── scripts/
│   ├── train_llama32_gpu.py          # Entrenamiento QLoRA con Unsloth
│   ├── train_block_full_gpu.sbatch   # Job Slurm para HPC-UCR
│   └── infer_llama.py                # Inferencia con el modelo final
│
├── outputs/
│   └── llama32_block1_full/
│       ├── adapter_model.safetensors
│       ├── adapter_config.json
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       ├── training_args.bin
│       └── training_summary_full.json
│
├── logs/
│   └── llama32_qlora_full_*.out      # Logs de entrenamiento (Slurm)
│
└── data/
    └── base.jsonl                    # Dataset privado
```

---

## 🚀 Setup & Training

### 1️⃣ Create and activate a virtual environment

```bash
python3 -m venv llama_env
source llama_env/bin/activate
```

---

### 2️⃣ Install dependencies

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1
python -m pip install transformers datasets unsloth accelerate sentencepiece
```

---

### 3️⃣ Verify GPU / CUDA availability

```bash
python - << 'PY'
import torch
print("torch:", torch.__version__, "build CUDA:", torch.version.cuda)
print("cuda available?:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

---

### 4️⃣ Launch training on HPC-UCR

```bash
cd ~/llama32_qlora/scripts
sbatch train_block_full_gpu.sbatch
```

---

### 5️⃣ Inspect training logs & metrics

```bash
# Ver log completo del entrenamiento
less ~/llama32_qlora/logs/llama32_qlora_full_28137.out

# Mostrar métricas resumidas
cat ~/llama32_qlora/outputs/llama32_block1_full/training_summary_full.json | jq
```

```bash
# Revisar scripts
nano ~/llama32_qlora/scripts/train_llama32_gpu.py
nano ~/llama32_qlora/scripts/train_block_full_gpu.sbatch
less ~/llama32_qlora/scripts/train_block_full_gpu.sbatch
```

---

## 🧠 Outputs

| File                         | Description |
|-----------------------------|-------------|
| `adapter_model.safetensors` | QLoRA adapters |
| `training_summary_full.json`| Final metrics |
| `tokenizer.json`            | Tokenizer used |
| `*.out` / `*.err`           | HPC-UCR logs |

---

## ⚠️ Notes

Dataset is not included for privacy and licensing reasons.  
Academic project (UCR, TICAL, HPC-UCR).

---

## 👩‍💻 Author

**Alison Lobo Salas**  
Universidad de Costa Rica (UCR)  
📍 San José, Costa Rica
```
