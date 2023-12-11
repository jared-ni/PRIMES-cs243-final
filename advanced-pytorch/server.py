from typing import Dict, Optional, Tuple
from collections import OrderedDict
import argparse
from torch.utils.data import DataLoader

import flwr as fl
import torch

import utils

import warnings
from dataset import prepare_dataset
import server_helper

import sys 
sys.path.append('../')

from CustomStrategy2 import CustomStrategy2

warnings.filterwarnings("ignore")

NUM_ROUNDS = 10 # number of rounds of federated learning
NUM_CLIENTS = 2 # number of total clients available (this is also the number of partitions we need to create)
BATCH_SIZE = 20 # batch size to use by clients during training
NUM_CLASSES = 10 # number of classes in our dataset (we use MNIST) -- this tells the model how to setup its output fully-connected layer
NUM_CLIENTS_PER_ROUND_FIT = 2 # number of clients to involve in each fit round (fit  round = clients receive the model from the server and do local training)
NUM_CLIENTS_PER_ROUND_EVAL = 2 # number of clients to involve in each evaluate round (evaluate round = client only evaluate the model sent by the server on their local dataset without training it)
LR = 0.1
MOMENTUM = 0.9
LOCAL_EPOCHS = 1

trainloaders, validationloaders, testloader = prepare_dataset(
    NUM_CLIENTS, BATCH_SIZE
)

def get_client_data(cid: int):
    return trainloaders[cid], validationloaders[cid]


def fit_config(server_round: int):
    """Return training configuration dict for each round.

    Keep batch size fixed at 32, perform two rounds of training with one local epoch,
    increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 16,
        "local_epochs": 1 if server_round < 2 else 2,
    }
    return config


def evaluate_config(server_round: int):
    """Return evaluation configuration dict for each round.

    Perform five local evaluation steps on each client (i.e., use five batches) during
    rounds one to three, then increase to ten local evaluation steps.
    """
    val_steps = 5 if server_round < 4 else 10
    return {"val_steps": val_steps}


def get_evaluate_fn(model: torch.nn.Module, toy: bool):
    """Return an evaluation function for server-side evaluation."""

    # Load data and model here to avoid the overhead of doing it in `evaluate` itself
    trainset, _, _ = utils.load_data()

    n_train = len(trainset)
    if toy:
        # use only 10 samples as validation set
        valset = torch.utils.data.Subset(trainset, range(n_train - 10, n_train))
    else:
        # Use the last 5k training examples as a validation set
        valset = torch.utils.data.Subset(trainset, range(n_train - 5000, n_train))

    valLoader = DataLoader(valset, batch_size=16)

    # The `evaluate` function will be called after every round
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        # Update model with the latest parameters
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        loss, accuracy = utils.test(model, valLoader)
        return loss, {"accuracy": accuracy}

    return evaluate


def main():
    """Load model for
    1. server-side parameter initialization
    2. server-side parameter evaluation
    """

    # Parse command line argument `partition`
    # parser = argparse.ArgumentParser(description="Flower")
    # parser.add_argument(
    #     "--toy",
    #     type=bool,
    #     default=False,
    #     required=False,
    #     help="Set to true to use only 10 datasamples for validation. \
    #         Useful for testing purposes. Default: False",
    # )

    # args = parser.parse_args()

    
    # 2) Load dataset


    # model = utils.load_efficientnet(classes=10)
    # model_parameters = [val.cpu().numpy() for _, val in model.state_dict().items()]

    # Create strategy
    # strategy = fl.server.strategy.FedAvg(
    #     fraction_fit=0.2,
    #     fraction_evaluate=0.2,
    #     min_fit_clients=NUM_CLIENTS_PER_ROUND_FIT,
    #     min_evaluate_clients=NUM_CLIENTS_PER_ROUND_EVAL,
    #     min_available_clients=2, # min_available_clients has to >= min running clients
    #     evaluate_fn=server_helper.get_evaluate_fn(NUM_CLASSES, testloader),
    #     on_fit_config_fn=fit_config,
    #     on_evaluate_config_fn=evaluate_config,
    # )
    strategy = fl.server.strategy.FedAvg()
    # strategy = CustomStrategy2(
    #     fraction_fit=0.0,  # in simulation, since all clients are available at all times, we can just use `min_fit_clients` to control exactly how many clients we want to involve during fit
    #     min_fit_clients=NUM_CLIENTS_PER_ROUND_FIT,  # number of clients to sample for fit()
    #     fraction_evaluate=0.0,  # similar to fraction_fit, we don't need to use this argument.
    #     min_evaluate_clients=NUM_CLIENTS_PER_ROUND_EVAL,  # number of clients to sample for evaluate()
    #     min_available_clients=NUM_CLIENTS,  # total clients in the simulation
    #     on_fit_config_fn=server_helper.get_on_fit_config(LR, MOMENTUM, LOCAL_EPOCHS),
    #     evaluate_fn=server_helper.get_evaluate_fn(NUM_CLASSES, testloader),
    # )  

    # Start Flower server for four rounds of federated learning
    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
