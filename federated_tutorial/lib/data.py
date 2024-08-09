from typing import List
from torch.utils.data import random_split, DataLoader
import torch

def prepare_dataset(trainset, testset, num_partitions:int=3, partition_ratio: List[int]=[1,1,1], batch_size:int=32, val_ratio:float=0.1):

  # Validate partition_ratio
  if not len(partition_ratio) == num_partitions:
    partition_ratio = [1] * num_partitions
  
  # Config to split trainset into num_partitions for separate clients
  total_ratio = sum(partition_ratio)
  partition_ratio = [r / total_ratio for r in partition_ratio]
  partition_len = [int(len(trainset) * r) for r in partition_ratio]

  # Split trainset
  trainsets = random_split(dataset=trainset, lengths=partition_len, generator=torch.Generator().manual_seed(42))
  
  # Create dataloaders
  testLoader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=True)
  
  trainLoaders =[]
  valLoaders=[]

  for trainset in trainsets:

    # Further split the trainset for training and validation
    num_totaldata = len(trainset)
    num_val = int(val_ratio * num_totaldata)
    num_train = num_totaldata - num_val

    train, val = random_split(dataset=trainset, lengths=[num_train, num_val], generator=torch.Generator().manual_seed(42))

    # Append to the loaders
    trainLoaders.append(DataLoader(dataset=train, batch_size=batch_size, shuffle=True, num_workers=2))
    valLoaders.append(DataLoader(dataset=val, batch_size=batch_size, shuffle=True, num_workers=2))


  return trainLoaders, valLoaders, testLoader