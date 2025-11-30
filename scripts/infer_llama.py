from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import sys
import argparse

# Argumentos de línea de comando
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True, help="Ruta del modelo entrenado")
parser.add_argument("--prompt", type=str, required=True, help="Texto de entrada para generar respuesta")
args = parser.parse_args()

# Cargar modelo y tokenizer
print(f"🔄 Cargando modelo desde: {args.model_path}")
tokenizer = AutoTokenizer.from_pretrained(args.model_path)
model = AutoModelForCausalLM.from_pretrained(
    args.model_path,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

# Crear pipeline de inferencia
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200,
    temperature=0.7,
    top_p=0.9
)

# Generar texto
print(f"\n🧠 Prompt: {args.prompt}\n")
outputs = generator(args.prompt)
print("💬 Respuesta generada:\n")
print(outputs[0]["generated_text"])

