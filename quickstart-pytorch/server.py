from typing import List, Tuple

import flwr as fl
from flwr.common import Metrics
from PrimesStrategy import PrimesStrategy
from NormalFedAvg import NormalFedAvg
from ClippingStrategy import ClippingStrategy
from server_helper import get_on_fit_config, get_evaluate_fn
from dataset import prepare_dataset
import argparse
from clientManager import CustomClientManager


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


parser = argparse.ArgumentParser(description="Flower")
parser.add_argument(
    "--dataset",
    type=int,
    default=False,
    required=False,
    help="decides the dataset: MNIST (1) or CIFAR10 (2)",
)
args = parser.parse_args()

_trainloaders, _validationloaders, testloader = prepare_dataset(
    dataset=args.dataset, num_partitions=2, batch_size=20, val_ratio=0.1
)

channels = 1 if args.dataset == 1 else 3
print("channels: ", channels)
# Define strategy
<<<<<<< HEAD
<<<<<<< Updated upstream
strategy = NormalFedAvg(
    fraction_fit=0.1,  # in simulation, since all clients are available at all times, we can just use `min_fit_clients` to control exactly how many clients we want to involve during fit
=======
# strategy = fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=weighted_average)
# strategy = CustomStrategy2(evaluate_metrics_aggregation_fn=weighted_average)
strategy = CustomStrategy2(
=======
# strategy = NormalFedAvg(
#     fraction_fit=0.2,  # in simulation, since all clients are available at all times, we can just use `min_fit_clients` to control exactly how many clients we want to involve during fit
#     min_fit_clients=3,  # number of clients to sample for fit()
#     fraction_evaluate=1.0,  # similar to fraction_fit, we don't need to use this argument.
#     min_evaluate_clients=3,  # number of clients to sample for evaluate()
#     min_available_clients=3,  # total clients in the simulation
#     evaluate_metrics_aggregation_fn=weighted_average,
#     on_fit_config_fn=get_on_fit_config(0.1, 0.9, 1), 
#     evaluate_fn=get_evaluate_fn(10, testloader),
# )

# strategy = ClippingStrategy(
#     fraction_fit=0.2,  # in simulation, since all clients are available at all times, we can just use `min_fit_clients` to control exactly how many clients we want to involve during fit
#     min_fit_clients=10,  # number of clients to sample for fit()
#     fraction_evaluate=1.0,  # similar to fraction_fit, we don't need to use this argument.
#     min_evaluate_clients=3,  # number of clients to sample for evaluate()
#     min_available_clients=3,  # total clients in the simulation
#     evaluate_metrics_aggregation_fn=weighted_average,
#     on_fit_config_fn=get_on_fit_config(0.1, 0.9, 1), 
#     evaluate_fn=get_evaluate_fn(10, testloader),
# )

strategy = PrimesStrategy(
>>>>>>> 09c1f1fe25e64e438b7e57e37718adce8facb91b
    fraction_fit=0.2,  # in simulation, since all clients are available at all times, we can just use `min_fit_clients` to control exactly how many clients we want to involve during fit
>>>>>>> Stashed changes
    min_fit_clients=3,  # number of clients to sample for fit()
    fraction_evaluate=1.0,  # similar to fraction_fit, we don't need to use this argument.
    min_evaluate_clients=3,  # number of clients to sample for evaluate()
    min_available_clients=3,  # total clients in the simulation
    evaluate_metrics_aggregation_fn=weighted_average,
    on_fit_config_fn=get_on_fit_config(0.1, 0.9, 1), 
    evaluate_fn=get_evaluate_fn(10, testloader, channels=channels),
)

# strategy = PrimesStrategy(
#     fraction_fit=0.2,  # in simulation, since all clients are available at all times, we can just use `min_fit_clients` to control exactly how many clients we want to involve during fit
#     min_fit_clients=10,  # number of clients to sample for fit()
#     fraction_evaluate=1.0,  # similar to fraction_fit, we don't need to use this argument.
#     min_evaluate_clients=3,  # number of clients to sample for evaluate()
#     min_available_clients=3,  # total clients in the simulation
#     evaluate_metrics_aggregation_fn=weighted_average,
#     on_fit_config_fn=get_on_fit_config(0.1, 0.9, 1), 
#     evaluate_fn=get_evaluate_fn(num_classes=10, testloader=testloader, channels=channels),
# )

# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=41),
    strategy=strategy,
    client_manager=CustomClientManager(),
)
