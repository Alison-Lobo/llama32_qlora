#!/bin/bash
#SBATCH --job-name=llama32_qlora_full
#SBATCH --output=/home/alison.lobo/llama32_qlora/logs/llama32_qlora_full_%j.out
#SBATCH --error=/home/alison.lobo/llama32_qlora/logs/llama32_qlora_full_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1                # 🔹 Usa una GPU completa (A100)
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G                   # 🔹 Un poco más de memoria
#SBATCH --time=10:00:00
#SBATCH --mail-user=alison.lobo@ucr.ac.cr
#SBATCH --mail-type=END,FAIL

echo "==((====))==  Ejecutando entrenamiento QLoRA + Unsloth (FULL) en GPU"
date

# --------------------------------------------------
# Preparar entorno
# --------------------------------------------------
module purge
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ~/llama_finetune
export UNSLOTH_DISABLE_RL='1'
echo "🔹 RL patch desactivado (UNSLOTH_DISABLE_RL=1)"

export HF_HOME=/home/alison.lobo/llama32_qlora/models/hf_home
export TRANSFORMERS_CACHE=/home/alison.lobo/llama32_qlora/models/hf_cache

# --------------------------------------------------
# Variables de control del entrenamiento
# --------------------------------------------------
export MODEL="meta-llama/Llama-3.2-1B-Instruct"          # ✅ modelo base
export DATA="/home/alison.lobo/llama32_qlora/pdfs_uned.jsonl"
export MAX_STEPS=0          # 0 = sin límite de pasos, usa épocas completas
export EPOCHS=1500             # número de épocas completas
export EVAL_STEPS=200       # evalúa cada 200 pasos
export LOG_STEPS=20         # imprime logs cada 20 pasos
export OUT=/home/alison.lobo/llama32_qlora/outputs/llama32_block1_full

# --------------------------------------------------
# Ejecutar entrenamiento
# --------------------------------------------------
echo "MODEL: $MODEL"
echo "DATA: $DATA"
export UNSLOTH_DISABLE_RL=1
python -u /home/alison.lobo/llama32_qlora/scripts/train_llama32_gpu.py

echo "✅ Job finalizado correctamente"
date

