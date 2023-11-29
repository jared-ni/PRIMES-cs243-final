from collections import OrderedDict
from typing import Dict, Tuple
from flwr.common import NDArrays, Scalar
import torch
import flwr as fl
from model import Net, train, test
import dataset
import warnings
from collections import OrderedDict

import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm

import torch
from torch.utils.data import random_split, DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision.datasets import MNIST


warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test(net, testloader):
    """Validate the model on the test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for images, labels in tqdm(testloader):
            outputs = net(images.to(DEVICE))
            labels = labels.to(DEVICE)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy

def get_mnist(data_path: str = "./data"):
    tr = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    trainset = MNIST(data_path, train=True, download=True, transform=tr)
    testset = MNIST(data_path, train=False, download=True, transform=tr)
    return trainset, testset

def prepare_data(num_partitions: int, batch_size: int, val_ratio: float = 0.1):
    trainset, testset = get_mnist()
    # return DataLoader(trainset, batch_size=32, shuffle=True), DataLoader(testset)
    num_images = len(trainset) // num_partitions
    partition_len = [num_images] * num_partitions
    trainsets = random_split(
        trainset, partition_len, torch.Generator().manual_seed(2023)
    )
    trainloaders = []
    valloaders = []
    for trainset_ in trainsets:
        num_total = len(trainset_)
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val

        for_train, for_val = random_split(trainset_, 
            [num_train, num_val], torch.Generator().manual_seed(2023)
        )
        trainloaders.append(
            DataLoader(for_train, batch_size=batch_size, shuffle=True) # remove 2 worker...
        )
        valloaders.append(
            DataLoader(for_val, batch_size=batch_size, shuffle=False)
        )
    # testloader = DataLoader(testset, batch_size=128)
    return trainloaders[0], valloaders[0]


# Load model and data (simple CNN, CIFAR-10)
net = Net().to(DEVICE)
# trainloader, testloader = load_data()
trainloader, testloader = prepare_data(3, 32)
# trainloaders, testloaders, _ = dataset.prepare_dataset(100, 20)
# trainloader, testloader = trainloaders[0], testloaders[0]
print(trainloader)
print("data loaded")
print(testloader)

# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters: NDArrays):
        """Receive parameters and apply them to the local model."""
        params_dict = zip(net.state_dict().keys(), parameters)
        # print("Parameters shapes:")
        # print([len(x) for x in parameters])
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)


    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(net, trainloader, epochs=1, device=DEVICE)
        return self.get_parameters(config={}), len(trainloader), {}


    def evaluate(self, parameters: NDArrays, config):
        self.set_parameters(parameters)
        loss, accuracy = test(net, testloader)
        return loss, len(testloader.dataset), {"accuracy": accuracy}


# Start Flower client
fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=FlowerClient(),
)
