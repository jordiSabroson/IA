import torch
import numpy as np
from torch.autograd import backward
import torch.nn as nn
from torch.nn.modules import ReLU
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
#Pasando de Imagen a Tensor
transform = transforms.Compose([
transforms.Resize((224,224)),
transforms.ToTensor()
])

#Definiendo que es el Dataset y preparando el DataLoader
directory = 'archive/DermMel'
train_dataset = datasets.ImageFolder(f"{directory}/train_sep",transform=transform)
train_loader = DataLoader(train_dataset,batch_size=64,shuffle=True)
test_dataset = datasets.ImageFolder(f"{directory}/test",transform=transform)
test_loader = DataLoader(test_dataset,batch_size=64,shuffle=True)
#Ejemplo del train dataset print(train_dataset[10][0][RGB,px,py])
#Modelo
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f'Device: {device}')
class MelanomaDetector(nn.Module):
    def __init__ (self):
        super().__init__ ()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size=5,stride=2),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=2),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0),
            nn.Flatten(),
            nn.Linear(10816,100),
            nn.ReLU(),
            nn.Linear(100,2)
        )
    def forward(self,X):
        X = self.features(X)
        return X

#Instancia del modelo 
model = MelanomaDetector().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr = 0.001)

#Funcion de entrenamiento
def train_model(model,train_loader,criterion,optimizer,num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs,labels = inputs.to(device),labels.to(device)

            #Paso forward
            outputs = model(inputs)
            loss = criterion(outputs,labels)

            #Paso backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
train_model(model,train_loader,criterion,optimizer)

#Funcion de eval
def evaluate_model(model,test_loader):
    model.eval()
    correct = 0
    correct_sum = 0
    total = 0
    total_sum = 0
    for inputs,labels in test_loader:
            inputs,labels = inputs.to(device),labels.to(device)
            outputs = model(inputs)
            _,predicted = torch.max(outputs.data,1)
            total =+ labels.size(0)
            total_sum =+ total
            correct =+ (predicted == labels).sum().item()
            correct_sum =+ correct
            print(f'Accuracy of the network on the {total} test images: {100 * correct / total:.2f}%')

    print(f'TotalTotal  Accuracy of the network on the {total_sum} test images: {100 * correct_sum / total_sum:.2f}%')
evaluate_model(model,test_loader)


