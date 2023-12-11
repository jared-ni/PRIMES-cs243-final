from collections import OrderedDict
from typing import Dict, Tuple
from flwr.common import NDArrays, Scalar

import torch
import flwr as fl

from model import Net, train, test

import dataset
import random
import argparse
from server import get_client_data

NUM_ROUNDS = 10 # number of rounds of federated learning
NUM_CLIENTS = 2 # number of total clients available (this is also the number of partitions we need to create)
BATCH_SIZE = 20 # batch size to use by clients during training
NUM_CLASSES = 10 # number of classes in our dataset (we use MNIST) -- this tells the model how to setup its output fully-connected layer
NUM_CLIENTS_PER_ROUND_FIT = 2 # number of clients to involve in each fit round (fit  round = clients receive the model from the server and do local training)
NUM_CLIENTS_PER_ROUND_EVAL = 2 # number of clients to involve in each evaluate round (evaluate round = client only evaluate the model sent by the server on their local dataset without training it)
LR = 0.1
MOMENTUM = 0.9
LOCAL_EPOCHS = 1


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
        lr = LR
        momentum = MOMENTUM
        epochs = LOCAL_EPOCHS

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
        print(self.valloader)

        return float(loss), len(self.valloader), {"accuracy": accuracy}


def main() -> None:
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--partition",
        type=int,
        default=0,
        choices=range(0, NUM_CLIENTS-1),
        required=False,
        help="Specifies the artificial data partition of CIFAR10 to be used. \
        Picks partition 0 by default",
    )
    args = parser.parse_args()

    # Start Flower client (need client local data)
    
    # trainloaders, validationloaders, testloader = dataset.prepare_dataset(
    #     NUM_CLIENTS, BATCH_SIZE
    # )
    cid = random.randint(0, NUM_CLIENTS - 1)
    client_train, client_val = get_client_data(cid)

    # client_train, client_val = dataset.get_client_data(NUM_CLIENTS, BATCH_SIZE)
    cid = random.randint(0, NUM_CLIENTS - 1)
    # cid = args.partition
    # cid = 0
    client = FlowerClient(trainloader=client_train, 
                          vallodaer=client_val, 
                          num_classes=NUM_CLASSES)
    # client = FlowerClient(trainloader=trainloaders[-1], 
    #                     vallodaer=validationloaders[-1], 
    #                     num_classes=NUM_CLASSES)

    # client = FlowerClient(trainloader=client_train, vallodaer=client_val, num_classes=NUM_CLASSES)

    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)


if __name__ == "__main__":
    main()