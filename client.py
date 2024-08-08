
import argparse
import flwr as fl
import torch
from dataset import get_mnist
from model import Net, train, test
from lib.federated import flwr_client as fl_client, start_client
from lib.data import prepare_dataset
from opacus import PrivacyEngine

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Parser
parser = argparse.ArgumentParser(description='Federated environment')

parser.add_argument('--client-id', type=int, required=True, default=1)
args = parser.parse_args()

client_id = args.client_id

# Download Dataset
trainset, testset = get_mnist()

# Prepare dataset
trainloaders, valloaders, testloader = prepare_dataset(trainset=trainset , testset=testset, batch_size=32, num_partitions=3)

# Prepare federated client
model = Net(num_classes=10)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

privacy_engine = PrivacyEngine()
model, optimizer, data_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=trainloaders[int(client_id)],
    noise_multiplier=1.1,
    max_grad_norm=1.0,
)

# Initiate client
client = fl_client(
  model=model,
  epochs=10,
  optimizer=optimizer,
  trainloader=data_loader,
  valloader=valloaders[int(client_id)],
  train=train,
  test=test,
  device=device).to_client()

# Connect to server and start learning
start_client(server_address='localhost:5050', client=client)

