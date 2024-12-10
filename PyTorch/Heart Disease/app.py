from flask import Flask, request, render_template
import torch
from torch import nn
import numpy as np

app = Flask(__name__)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

class HeartModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(13,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,1)
        )
    
    def forward(self,x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
model = HeartModel().to(device)
model.load_state_dict(torch.load("heart_model.pth"))  # Carga el modelo entrenado
model.eval()

@app.route('/')
def form():
    return render_template('form.html')

@app.route('/enviar', methods=['POST'])
def predict():
    data = request.form
    try:
        # Convertir los datos en una lista de entrada
        input_data = [
            float(data['age']),
            1 if data['sex'] == 'home' else 0,  # Codifica 'home' como 1 y 'dona' como 0
            int(data['cp']),
            float(data['trestbps']),
            float(data['chol']),
            int(data['fbs']),
            int(data['restecg']),
            float(data['thalach']),
            0 if data['exang'] == 'Si' else 1,  # Codifica 'Si' como 0 y 'No' como 1
            float(data['oldpeak']),
            int(data['slope']),
            int(data['ca']),
            int(data['thal'])
        ]

        # Convertir a tensor para el modelo
        input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)  # Añade una dimensión para batch

        # Mover el tensor al mismo dispositivo que el modelo
        input_tensor = input_tensor.to(device)

        # Hacer predicción
        model.to(device) 
        prediction = model(input_tensor).item()

        # Interpretar la predicción
        result = "No hi ha risc de malaltia cardíaca" if prediction < 0.5 else "Possible malaltia cardíaca"
        return f"<h1>Resultado: {result}</h1>"

    except Exception as e:
        return f"<h1>Error procesando la entrada: {e}</h1>"
    
if __name__ == '__main__':
    app.run(debug=True)