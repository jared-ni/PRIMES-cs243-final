from collections import OrderedDict
from typing import Dict, Tuple
from flwr.common import NDArrays, Scalar

import torch
import flwr as fl

from model import Net, train, test

import utils
from torch.utils.data import DataLoader
import torchvision.datasets
import torch
import flwr as fl
import argparse
from collections import OrderedDict
import warnings
import dataset
import random

NUM_ROUNDS = 10 # number of rounds of federated learning
NUM_CLIENTS = 100 # number of total clients available (this is also the number of partitions we need to create)
BATCH_SIZE = 20 # batch size to use by clients during training
NUM_CLASSES = 10 # number of classes in our dataset (we use MNIST) -- this tells the model how to setup its output fully-connected layer
NUM_CLIENTS_PER_ROUND_FIT = 10 # number of clients to involve in each fit round (fit  round = clients receive the model from the server and do local training)
NUM_CLIENTS_PER_ROUND_EVAL = 25 # number of clients to involve in each evaluate round (evaluate round = client only evaluate the model sent by the server on their local dataset without training it)
CONFIG = {
    "lr": 0.1,
    "momentum": 0.9,
    "local_epochs": 1,
}

class FlowerClient(fl.client.NumPyClient):
    """Define a Flower Client."""

    def __init__(self, trainloader, vallodaer, num_classes) -> None:
        super().__init__()

        # the dataloaders that point to the data associated to this client
        self.trainloader = trainloader
        self.valloader = vallodaer

        # a model that is randomly initialised at first
        self.model = Net(num_classes)

        # figure out if this client has access to GPU support or not
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    def set_parameters(self, parameters: NDArrays):
        """Receive parameters and apply them to the local model."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        # print("Parameters shapes:")
        # print([len(x) for x in parameters])

        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})

        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config: Dict[str, Scalar]):
        """Extract model parameters and return them as a list of numpy arrays."""

        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        """Train model received by the server (parameters) using the data.

        that belongs to this client. Then, send it back to the server.
        """

        # copy parameters sent by the server into client's local model
        self.set_parameters(parameters)
        # print("Parameters shapes:")
        # print([len(x) for x in parameters])

        # print parameters
        # print("Parameters to set:")
        # print(parameters)

        # fetch elements in the config sent by the server. Note that having a config
        # sent by the server each time a client needs to participate is a simple but
        # powerful mechanism to adjust these hyperparameters during the FL process. For
        # example, maybe you want clients to reduce their LR after a number of FL rounds.
        # or you want clients to do more local epochs at later stages in the simulation
        # you can control these by customising what you pass to `on_fit_config_fn` when
        # defining your strategy.
        lr = CONFIG["lr"]
        momentum = CONFIG["momentum"]
        epochs = CONFIG["local_epochs"]

        # a very standard looking optimiser
        optim = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)

        # do local training. This function is identical to what you might
        # have used before in non-FL projects. For more advance FL implementation
        # you might want to tweak it but overall, from a client perspective the "local
        # training" can be seen as a form of "centralised training" given a pre-trained
        # model (i.e. the model received from the server)

        # loss, accuracy = test(self.model, self.valloader, self.device)
        # print(f"Client Loss Before: {loss}, Accuracy: {accuracy}")


        train(self.model, self.trainloader, optim, epochs, self.device)

        # we can do this to get before and after accuracy. however, this is the 
        # loss, accuracy = test(self.model, self.valloader, self.device)
        # print(f"Client Loss After: {loss}, Accuracy: {accuracy}")

        # Flower clients need to return three arguments: the updated model, the number
        # of examples in the client (although this depends a bit on your choice of aggregation
        # strategy), and a dictionary of metrics (here you can add any additional data, but these
        # are ideally small data structures)
        return self.get_parameters({}), len(self.trainloader), {}
    

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        self.set_parameters(parameters)

        loss, accuracy = test(self.model, self.valloader, self.device)
        print(f"Client Loss: {loss}, Accuracy: {accuracy}")

        return float(loss), len(self.valloader), {"accuracy": accuracy}


def main() -> None:
    # Parse command line argument `partition`
    # parser = argparse.ArgumentParser(description="Flower")
    # if args.dry:
    #     client_dry_run(device)
    # else:
    #     # Load a subset of CIFAR-10 to simulate the local data partition
    #     trainset, testset = utils.load_partition(args.partition)

    #     if args.toy:
    #         trainset = torch.utils.data.Subset(trainset, range(10))
    #         testset = torch.utils.data.Subset(testset, range(10))

    # Start Flower client (need client local data)
    
    trainloaders, validationloaders, testloader = dataset.prepare_dataset(
        NUM_CLIENTS, BATCH_SIZE
    )

    # client_train, client_val = dataset.get_client_data(NUM_CLIENTS, BATCH_SIZE)
    cid = random.randint(0, NUM_CLIENTS - 1)
    client = FlowerClient(trainloader=trainloaders[cid], vallodaer=validationloaders[cid], num_classes=NUM_CLASSES)

    # client = FlowerClient(trainloader=client_train, vallodaer=client_val, num_classes=NUM_CLASSES)

    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)


if __name__ == "__main__":
    main()