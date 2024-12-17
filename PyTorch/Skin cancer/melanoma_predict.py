import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.optim as optim

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

directory = 'data'
bs = 64

train_dataset = datasets.ImageFolder(f"{directory}/train",transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True)

test_dataset = datasets.ImageFolder(f"{directory}/test", transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=bs, shuffle=True)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f'Device: {device}')

class MelanomaDetector(nn.Module):
    def __init__(self):
        super().__init__()
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
    
    def forward(self, X):
        X = self.features(X)
        return X
    
model = MelanomaDetector().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr = 0.001)

def train_model(model,train_dataloader,criterion,optimizer):
    model.train()
    size = len(train_dataloader.dataset)
    running_loss = 0.0
    for batch, (inputs, labels) in enumerate(train_dataloader):
        inputs,labels = inputs.to(device),labels.to(device)

        #Paso forward
        outputs = model(inputs)
        loss = criterion(outputs,labels)

        #Paso backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

        if batch % 10 == 0:
            current = batch * bs + len(inputs)
            print(f'Loss: {running_loss/size:.4f} [{current:>5d}/{size:>5d}]')
            
def evaluate_model(model,test_dataloader):
    model.eval()
    correct = 0
    correct_sum = 0
    total = 0
    total_sum = 0
    for inputs,labels in test_dataloader:
            inputs,labels = inputs.to(device),labels.to(device)
            outputs = model(inputs)
            _,predicted = torch.max(outputs.data,1)
            total =+ labels.size(0)
            total_sum =+ total
            correct =+ (predicted == labels).sum().item()
            correct_sum =+ correct
            print(f'Accuracy of the network on the {total} test images: {100 * correct / total:.2f}%')

    print(f'Total accuracy of the network on the {total_sum} test images: {100 * correct_sum / total_sum:.2f}%')


epochs = 5
for e in range(epochs):
    print(f"Epoch {e+1}\n-------------------------------")
    train_model(model,train_dataloader,criterion,optimizer)
    print("\n")
print("Done!\n")
evaluate_model(model,test_dataloader)