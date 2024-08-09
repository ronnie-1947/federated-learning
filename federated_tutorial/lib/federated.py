import flwr as fl
import numpy as np
import torch
from typing import Optional, List, Tuple
from torch.nn import Module
from torch.optim import Optimizer
from typing import Dict, List, Callable
from flwr.common import NDArrays, Scalar, Metrics
from flwr.server.strategy import Strategy
from torch.utils.data import DataLoader
from collections import OrderedDict
import random

# Train and Test function structure
def train_func(model: Module, trainloader:DataLoader, optimizer:Optimizer, epochs:int, device: str, cb: Callable):
  return cb(model=model, trainloader=trainloader, optimizer=optimizer, epochs=epochs, device=device)
  
def test_func(model: Module, testloader:DataLoader, device: str, cb: Callable):
  return cb(model=model, testloader=testloader, device=device)

# Flower client
class flwr_client(fl.client.NumPyClient):
  def __init__(self, model: Module, optimizer:Optimizer, valloader:DataLoader, trainloader:DataLoader, train:Callable, test:Callable, epochs:int, device:Optional[str]):
    
    if not device:
      device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    self.model = model
    self.trainloader = trainloader
    self.valloader = valloader
    self.train = train
    self.test = test
    self.device = device
    self.epochs = epochs
    self.optim = optimizer
    
  def get_parameters(self, config: Dict[str, Scalar]) -> List[np.ndarray]:
    # Return model parameters as a list of Numpy ndarrays
    return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
  
  def set_parameters(self, parameters: List[np.ndarray]):
    self.model.train()
    
    params_dict = zip(self.model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    self.model.load_state_dict(state_dict, strict=True)
    
  def fit(self, parameters, config):
    self.set_parameters(parameters=parameters)
    train_func(model=self.model, trainloader=self.trainloader, optimizer=self.optim, epochs=self.epochs, device=self.device, cb=self.train)
    return self.get_parameters(config={}), len(self.trainloader), {}
  
  def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
    self.set_parameters(parameters)
    loss, accuracy = test_func(model=self.model, testloader=self.valloader, device=self.device, cb=self.test)
    return float(loss), len(self.valloader), {"accuracy": accuracy}
  
  
def start_server(host_addr:str='localhost:5050', num_rounds:Optional[int]=10, strategy: Optional[Strategy] = None):
  
  config = fl.server.ServerConfig(num_rounds=num_rounds)
  
  if not strategy :
    def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
  
      # Multiply accuracy of each client by number of examples used
      accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
      examples = [num_examples for num_examples, _ in metrics]

      # Aggregate and return custom metric (weighted average)
      accuracy = (sum(accuracies) / sum(examples)) * 10

      return {"accuracy": (sum(accuracies) / sum(examples))}

    # Define strategy
    strategy = fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=weighted_average) 
  
  fl.server.start_server(
    server_address=host_addr,
    config=config,
    strategy=strategy
  )
  
def start_client(server_address:str='localhost:5050', client:fl.client=None):
  
  if not client:
    return None
  
  fl.client.start_client(server_address=server_address, client=client)