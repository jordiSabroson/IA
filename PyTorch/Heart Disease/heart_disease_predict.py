import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split

class HeartDataset(Dataset):
    def __init__(self, file, transform=None, target_transform=None):
        data = pd.read_csv(file)
        self.X = torch.tensor(data.iloc[:, :-1].values, dtype=torch.float32)
        self.y = torch.tensor(data.iloc[:, -1].values, dtype=torch.float32).unsqueeze(1)

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        features = self.X[index]
        label = self.y[index]

        if self.transform:
            features = self.transform(features)
        if self.target_transform:
            label = self.target_transform(label)

        return features, label
    

dataset = HeartDataset(file='/home/iticbcn/Escritorio/UA/PyTorch/Heart Disease/data/heart.csv')

dataset_len = len(dataset)
train_size = int(dataset_len * 0.8) # Les dades d'entrenament son el 80% del dataset
test_size = dataset_len - train_size # Les dades de test son el 20% restant
train_data, test_data = random_split(dataset, [train_size, test_size])

bs = 32
train_dataloader = DataLoader(train_data, batch_size=bs, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=bs, shuffle=True)

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

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
print(model)

learning_rate = 0.003

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * bs + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += ((torch.sigmoid(pred) > 0.5) == y).type(torch.float).sum().item()
            
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")

# Guardar el estado del modelo entrenado
torch.save(model.state_dict(), "heart_model.pth")

# Executar: CUDA_LAUNCH_BLOCKING=1 /home/iticbcn/Escritorio/UA/ex-basics/bin/python "/home/iticbcn/Escritorio/UA/PyTorch/Heart Disease/heart_disease_predict.py"
