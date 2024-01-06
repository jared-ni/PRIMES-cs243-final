# Price Incentive Model Efficiency System

### Installing Dependencies

Project dependencies (such as `torch` and `flwr`) are defined in `./PRIMES_FLrequirements.txt`. To install those dependencies and manage your virtual environment, use [pip](https://pip.pypa.io/en/latest/development/), but feel free to use a different way of installing dependencies and managing virtual environments if you have other preferences.

### Running Experiments
To run experiments using a select strategy, uncomment the strategy among {PrimesStrategy, ClippingStrategy, NormalFedAvg} you wish to use in ./pytorch_quickstart/server.py and uncomment the rest. 

Then, configure the number of clients in pytorch_quickstart/run.sh. Note that there are three configurations: --dataset (int: {1 (for MNIST), 2 (for CIFAR-10)}), 
--fraction (int: larger = smaller data size for client), and --corruption (float: [0, 1], determines proportion of that client's data that is corrupted/zeroed out). 

After doing these, run: 

`python primes_server.py`

Then in another terminal, run: 

`./pytorch_quickstart/run.sh`

to see the federated learning experiment begin. 

### Replicate Results
For 10 rounds accuracy overtime on MNIST, change the total rounds in `server.py` to 10, set 50 clients in `run.sh`, with 10 clients in each of these corruption levels: 0.1, 0.3, 0.5, 0.7, 0.9. Set the fraction to 50 (so each client has 1/50 of the total data distribution). Set dataset to 1 for MNIST. After these are ready, uncomment the desired strategy in server.py. 

For other tests, set the total training rounds in `server.py` to 50. For running CIFAR-10, set dataset to 2 in `run.sh`. 

For the bad/good client ratio test, set 50 clients in `run.sh`, with varying ratios of clients in these corruption levels: 0.6 (bad clients), 0.1 (good clients). 


