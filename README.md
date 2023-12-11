### Installing Dependencies

Project dependencies (such as `torch` and `flwr`) are defined in `./PRIMES_FLrequirements.txt`. To install those dependencies and manage your virtual environment, use [pip](https://pip.pypa.io/en/latest/development/), but feel free to use a different way of installing dependencies and managing virtual environments if you have other preferences.

### running experiments
To run experiments using a select strategy, uncomment the strategy among {PrimesStrategy, ClippingStrategy, NormalFedAvg} you wish to use in ./pytorch_quickstart/server.py and uncomment the rest. 

Then, configure the number of clients in pytorch_quickstart/run.sh. Note that there are three configurations: --dataset (int: {1 (for MNIST), 2 (for CIFAR-10)}), 
--fraction (int: larger = smaller data size for client), and --corruption (float: [0, 1], determines proportion of that client's data that is corrupted/zeroed out). 

After doing these, run: 

```
python primes_server.py
./pytorch_quickstart/run.sh
```
to see federated learning experiment begins. 
