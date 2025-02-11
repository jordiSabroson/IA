import torch
import tiktoken
from transformers import AutoModelForCausalLM

# Cargar el tokenizer de tiktoken
tokenizer = tiktoken.get_encoding("cl100k_base")  # Ajusta según tu modelo específico

# Cargar el modelo desde el fichero model.pth
model_path = "model.pth"  # Asegúrate de que la ruta sea correcta
model = AutoModelForCausalLM.from_pretrained(model_path)  # Ajusta según tu modelo específico

# Si tu modelo está en GPU:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def generate_text(prompt, max_length=50):
    # Tokenizar la frase de entrada usando tiktoken
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids]).to(device)
    
    # Generar texto adicional
    with torch.no_grad():
        output = model.generate(
            input_tensor,
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            early_stopping=True
        )
    
    # Decodificar la salida generada usando tiktoken
    generated_ids = output[0].cpu().numpy()
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_text

# Ejemplo de uso
frase_inicial = "Estas son las diez palabras iniciales del texto que quiero continuar"
continuacion = generate_text(frase_inicial, max_length=150)  # Ajusta la longitud según sea necesario
print(continuacion)