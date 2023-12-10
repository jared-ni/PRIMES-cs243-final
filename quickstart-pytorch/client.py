from collections import OrderedDict
from typing import Dict, Tuple
from flwr.common import NDArrays, Scalar
import torch
import flwr as fl
from model import Net, train, test
import dataset
import warnings
from collections import OrderedDict
import random
import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torchvision.transforms import Compose, Normalize, ToTensor, RandomErasing
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from dataset import random_split
from torchvision.datasets import MNIST


warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# client data quality
data_quality = random.uniform(0, 1)
data_amount = random.randint(15, 1000)

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


# def crop_my_image(image: PIL.Image.Image) -> PIL.Image.Image:
#     """Crop the images so only a specific region of interest is shown to my PyTorch model"""
#     left, right, width, height = 20, 80, 40, 60
#     return transforms.functional.crop(image, left=left, top=top, width=width, height=height)


def get_mnist(data_path: str = "./data"):
    # determines data quality
    # erase_probability = random.uniform(0, 1)
    erase_probability = 1
    
    tr = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))
                  ])
                #   ,transforms.Lambda(crop_my_image)])
    
    trainset = MNIST(data_path, train=True, download=True, transform=tr)
    testset = MNIST(data_path, train=False, download=True, transform=tr)

    return trainset, testset


def prepare_data(num_partitions: int, batch_size: int, val_ratio: float = 0.1):
    trainset, testset = get_mnist()
    # num_partitions = random.randint(15, 1000)
    # pick random among {5, 100, 1000}
    num_partitions = random.choice([5, 100, 1000])
    # num_partitions = 1000
    # return DataLoader(trainset, batch_size=32, shuffle=True), DataLoader(testset)
    num_images = len(trainset) // num_partitions
    partition_len = [num_images] * num_partitions

    print("num_images", num_images)

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
    # cid = random.randint(0, num_partitions - 1)
    cid = 0
    cid = random.randint(0, num_partitions - 1)
    return trainloaders[cid], valloaders[cid]


# Load model and data (simple CNN, CIFAR-10)
net = Net().to(DEVICE)
# trainloader, testloader = load_data()
trainloader, testloader = prepare_data(5000, 20)
# trainloaders, testloaders, _ = dataset.prepare_dataset(100, 20)
# trainloader, testloader = trainloaders[0], testloaders[0]

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
        nextStepLoss, _, _ = self.nextStepEvaluate(parameters, {})
        return loss, len(testloader.dataset), {"accuracy": accuracy, "next_step_loss": nextStepLoss}
    

    def nextStepEvaluate(self, parameters: NDArrays, config):
        self.set_parameters(parameters)
        nextStepLoss, accuracy = test(net, trainloader)
        return nextStepLoss, len(trainloader.dataset), {"accuracy": accuracy}
    




# Start Flower client
fl.client.start_numpy_client(
    # server_address="172.31.31.180:8080",
    server_address="0.0.0.0:8080",
    client=FlowerClient(),
)
