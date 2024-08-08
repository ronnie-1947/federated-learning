from torch.utils.data import random_split, DataLoader
import torch

def prepare_dataset(trainset, testset, num_partitions:int=3, partition_ratio=[1,1,1], batch_size:int=32, val_ratio:float=0.1):

  # Config to split trainset into num_partitions for separate clients
  num_data_for_each_client = int(len(trainset)//num_partitions)
  partition_len = [num_data_for_each_client] * num_partitions

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