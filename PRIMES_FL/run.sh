#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/

# Download the CIFAR-10 dataset
python -c "from torchvision.datasets import CIFAR10; CIFAR10('./data', download=True)"

echo "Starting server"
python server.py --dataset 1 &
sleep 5  # Sleep for 3s to give the server enough time to start

for i in `seq 0 9`; do
    echo "Starting client $i"
    python client.py --fraction 20 --corruption 0.9 --dataset 1 &
done

for i in `seq 10 19`; do
    echo "Starting client $i"
    python client.py --fraction 20 --corruption 0.7 --dataset 1 &
done

for i in `seq 20 29`; do
    echo "Starting client $i"
    python client.py --fraction 20 --corruption 0.5 --dataset 1 &
done

for i in `seq 30 39`; do
    echo "Starting client $i"
    python client.py --fraction 20 --corruption 0.3 --dataset 1 &
done

for i in `seq 40 49`; do
    echo "Starting client $i"
    python client.py --fraction 20 --corruption 0.1 --dataset 1 &
done



# Enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
