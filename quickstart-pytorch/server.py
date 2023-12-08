from typing import List, Tuple

import flwr as fl
from flwr.common import Metrics
from CustomStrategy2_nan import CustomStrategy2
from server_helper import get_on_fit_config, get_evaluate_fn
from dataset import prepare_dataset

from clientManager import CustomClientManager

# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


_trainloaders, _validationloaders, testloader = prepare_dataset(
    2, 20
)


# Define strategy
# strategy = fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=weighted_average)
# strategy = CustomStrategy2(evaluate_metrics_aggregation_fn=weighted_average)
strategy = CustomStrategy2(
    fraction_fit=0.2,  # in simulation, since all clients are available at all times, we can just use `min_fit_clients` to control exactly how many clients we want to involve during fit
    min_fit_clients=2,  # number of clients to sample for fit()
    fraction_evaluate=0.2,  # similar to fraction_fit, we don't need to use this argument.
    min_evaluate_clients=5,  # number of clients to sample for evaluate()
    min_available_clients=15,  # total clients in the simulation
    evaluate_metrics_aggregation_fn=weighted_average,
    on_fit_config_fn=get_on_fit_config(0.1, 0.9, 1), 
    evaluate_fn=get_evaluate_fn(10, testloader),
)

# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=10),
    strategy=strategy,
    client_manager=CustomClientManager(),
)
