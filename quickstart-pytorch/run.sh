#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/

# Download the CIFAR-10 dataset
python -c "from torchvision.datasets import CIFAR10; CIFAR10('./data', download=True)"

echo "Starting server"
python server.py &
sleep 5  # Sleep for 3s to give the server enough time to start

for i in `seq 0 3`; do
    echo "Starting client $i"
    python client.py --fraction 15 --corruption 0.9 &
done

echo "Starting client 4"
python client.py --fraction 15 --corruption 0.1 &

# for i in `seq `; do
#     echo "Starting client $i"
#     python client.py --fraction 1000 &
# done

# Enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
