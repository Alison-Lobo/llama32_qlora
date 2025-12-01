# 🦙 LLaMA 3.2 1B – Fine-Tuning en Español (QLoRA + Unsloth + HPC-UCR)

Python · PyTorch 2.7.1 · Transformers · Unsloth · QLoRA · CUDA 12.8 · HPC-UCR  


## 🧠 Overview

This project adapts the **LLaMA 3.2 1B Instruct** model to **Spanish** using efficient fine-tuning with **QLoRA (4-bit)** through the **Unsloth** library, executed on the **HPC-UCR cluster**.

The objective of this work is to train a base model capable of responding to instructions in Spanish, using an academic data repository processed to JSONL.

This project demonstrates how it is possible to train language models in university infrastructures using modern optimization techniques and GPU consumption.

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

Training was run for **60 epochs** using QLoRA (4-bit) on an NVIDIA **A100 80GB** GPU (HPC-UCR), with a warmup + cosine decay scheduler.

From the training logs (`llama32_qlora_full_28137.out`) and `training_summary_full.json`, the model shows a consistent improvement on the validation set:

| Metric                              | Value (approx.)                      |
|------------------------------------|--------------------------------------|
| Initial eval loss (epoch ≈ 1)      | **3.08**                             |
| Final eval loss (epoch 60)         | **1.70**                             |
| Initial perplexity (epoch ≈ 1)     | **21.74**                            |
| Final perplexity (epoch 60)        | **≈ 5.47**                           |
| Training epochs                    | **60**                               |
| Effective GPU runtime              | **3 blocks × 6 hours (A100 80GB)**   |

These values indicate a **monotonic decrease in loss and perplexity** on the validation set, showing that the model improves its ability to model Spanish-language sequences as training progresses.

### 📈 Metrics by epoch (summary)

The table below summarizes the behaviour of the main metrics at selected epochs:

| Epoch | Train loss | Eval loss | Perplexity (≈ e^loss) | Learning rate        | Grad norm |
|------:|-----------:|----------:|----------------------:|----------------------|----------:|
| 1     | 3.2204     | 3.0790    | 21.74                 | 2.66×10⁻⁵           | 4.09      |
| 10    | 0.2142     | 3.0143    | 20.37                 | 1.71×10⁻⁴           | 1.17      |
| 20    | 0.1775     | 2.9798    | 19.68                 | 1.37×10⁻⁴           | 1.07      |
| 30    | 0.1523     | 2.8126    | 16.65                 | 1.03×10⁻⁴           | 0.36      |
| 40    | 0.1519     | 2.6656    | 14.38                 | 6.88×10⁻⁵           | 0.33      |
| 50    | 0.1454     | 2.4571    | 11.67                 | 3.40×10⁻⁵           | 0.30      |
| 60    | 0.1392     | 1.7000    | 5.47                  | 4.35×10⁻⁸           | 0.29      |

- **Train loss** drops rapidly from ~3.22 to ~0.14 and then stabilizes.
- **Eval loss** decreases from ~3.08 to ~1.70, with an associated reduction of **perplexity** from ~21.7 to ~5.5.
- The **learning rate** increases after the warmup phase (peaking around 1.9×10⁻⁴) and then decays almost to zero at the end, following a *cosine decay* pattern.
- The **gradient norm** goes from ~4.1 to values close to 0.3, showing large updates at the beginning and progressively smaller, more stable steps in later epochs.

Taken together, these curves show a **stable training process without exploding gradients** and a sustained improvement on the validation set – an appropriate behaviour for a **1B-parameter model** fine-tuned with QLoRA in an academic setting.


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
less ~/llama32_qlora/logs/llama32_qlora_full_*.out

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
| `llama32_qlora_full_*.out`| Final metrics |
| `tokenizer.json`            | Tokenizer used |
| `*.out` / `*.err`           | HPC-UCR logs |

---

## ⚠️ Notes

Dataset is not included for privacy and licensing reasons.  
Academic project (UCR, HPC-UCR).

---

## 👩‍💻 Author

**Alison Lobo Salas**  
Universidad de Costa Rica (UCR)  
📍 San José, Costa Rica
```
