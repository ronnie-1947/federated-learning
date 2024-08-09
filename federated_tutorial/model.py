import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Note the model and functions here defined do not have any FL-specific components.
class Net(nn.Module):
  def __init__(self, num_classes: int) -> None:
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(1, 6, 5)
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(6, 16, 5)
    self.fc1 = nn.Linear(16 * 4 * 4, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, num_classes)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = x.view(-1, 16 * 4 * 4)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x

def train(model: Net, trainloader:DataLoader, optimizer, epochs:int, device: str):
  criterion = torch.nn.CrossEntropyLoss()
  model.train()
  model.to(device)
  for _ in range(epochs):
    for images, labels in trainloader:
      images, labels = images.to(device), labels.to(device)
      optimizer.zero_grad()
      loss = criterion(model(images), labels)
      loss.backward()
      optimizer.step()


def test(model:Net, testloader:DataLoader, device:str):
  try:
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    model.eval()
    model.to(device)
    with torch.no_grad():
      for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        loss += criterion(outputs, labels).item()
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy
  except Exception as e:
    print('EXCEPTION ERROR',e)