# ==============================================================
# Entrenamiento QLoRA con Unsloth (versión estable y sin RL)
# ==============================================================
import os, math, time, json, traceback
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from unsloth.trainer import SFTTrainer
from transformers import EarlyStoppingCallback

# =====================
# Configuración general
# =====================
MODEL_NAME  = os.getenv("MODEL",  "meta-llama/Llama-3.2-1B-Instruct")
DATASET     = os.getenv("DATA",   "/home/alison.lobo/llama32_qlora/base.jsonl")
OUTPUT_DIR  = os.getenv("OUT",    "/home/alison.lobo/llama32_qlora/outputs/llama32_block1_full")
MAX_STEPS   = int(os.getenv("MAX_STEPS", "1500"))
EVAL_STEPS  = int(os.getenv("EVAL_STEPS", "200"))
LOG_STEPS   = int(os.getenv("LOG_STEPS", "20"))
EPOCHS      = int(os.getenv("EPOCHS", "60"))

print("🚀 Iniciando entrenamiento QLoRA (FULL) con Unsloth")
print("📁 Dataset:", DATASET)
print("🧠 Modelo:", MODEL_NAME)
print("GPU disponible:", torch.cuda.is_available(), "| Dispositivo:",
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")

start_ts = time.time()

try:
    # =====================
    # Cargar modelo y LoRA
    # =====================
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        load_in_4bit=True,
        max_seq_length=4096,
        device_map="auto",
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,  # cambiar a 0.0 si querés máximo rendimiento
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    )

    # =====================
    # Dataset JSONL
    # =====================
    if DATASET.endswith(".jsonl"):
        ds = load_dataset("json", data_files=DATASET)
    else:
        ds = load_dataset(DATASET)

    def format_row(ex):
        inst = ex.get("instruction", "")
        inp  = ex.get("input", "")
        out  = ex.get("output", "")
        if inp:
            txt = f"### Instrucción:\n{inst}\n\n### Entrada:\n{inp}\n\n### Respuesta:\n{out}"
        else:
            txt = f"### Instrucción:\n{inst}\n\n### Respuesta:\n{out}"
        return {"text": txt}

    ds = ds.map(format_row, remove_columns=ds["train"].column_names)
    ds = ds["train"].train_test_split(test_size=0.1, seed=42)
    train_data, eval_data = ds["train"], ds["test"]
    print(f"📊 Tamaños → Train: {len(train_data)} | Eval: {len(eval_data)}")

    # =====================
    # Argumentos de entrenamiento
    # =====================
    args_kwargs = dict(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        logging_steps=LOG_STEPS,
        evaluation_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        num_train_epochs=EPOCHS,
        max_steps=MAX_STEPS,
        bf16=torch.cuda.is_bf16_supported(),
        learning_rate=2e-4,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        report_to="none",
    )

    # =====================
    # Entrenador
    # =====================
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_data,
        eval_dataset=eval_data,
        args_kwargs=args_kwargs,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print("🚦 Entrenando...")
    trainer.train()

    # =====================
    # Evaluación y métricas
    # =====================
    print("\n📈 Evaluando...")
    metrics = trainer.evaluate()
    print("✅ Métricas:", metrics)

    ppl_final = None
    if "eval_loss" in metrics:
        ppl_final = math.exp(metrics["eval_loss"])
        print(f"🔢 Perplejidad final: {ppl_final:.3f}")

    # =====================
    # Perplejidad por evaluación
    # =====================
    if trainer.state.log_history:
        print("\n📊 Perplejidad por evaluación:")
        for log in trainer.state.log_history:
            if "eval_loss" in log:
                step = log.get("step", "?")
                try:
                    ppl = math.exp(log["eval_loss"])
                except OverflowError:
                    ppl = float("inf")
                print(f"  Step {step} | eval_loss={log['eval_loss']:.4f} | perplexity={ppl:.2f}")

    # =====================
    # Guardar resultados
    # =====================
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trainer.save_model(OUTPUT_DIR)

    summary = {
        "train_loss": metrics.get("train_loss"),
        "eval_loss": metrics.get("eval_loss"),
        "eval_perplexity": ppl_final,
        "total_runtime_sec": round(time.time() - start_ts, 2),
        "epochs": EPOCHS,
    }
    with open(os.path.join(OUTPUT_DIR, "training_summary_full.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"📝 Resumen guardado en {OUTPUT_DIR}/training_summary_full.json")

except Exception as e:
    print("❌ Error durante la ejecución:\n", e)
    traceback.print_exc()

