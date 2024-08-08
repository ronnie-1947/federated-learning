from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Compose

def get_mnist(data_path: str = './data'):
  
  transformFunc = Compose([ToTensor()])

  trainset = MNIST(data_path, train=True, download=True, transform=transformFunc)
  testset = MNIST(data_path, train=False, download=True, transform=transformFunc)
  
  return trainset, testset

