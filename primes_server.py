from concurrent import futures
import time

import grpc
import primes_pb2 as primes
import primes_pb2_grpc as rpc

import torch
import flwr as fl
import threading

from FL.model import Net, train, test


class PrimesServicer(rpc.PrimesServicer):
    def __init__(self):
        self.server_clients = {}
        self.next_step_clients = {}

        self.server_lock = threading.Lock()
        self.next_step_lock = threading.Lock()


    # function to gather current round of loss
    def getNextStepLoss(self, request: primes.lossAndAccuracyRequest, context):
        # get current round of loss
        step_data = zip(request.cids, request.losses, request.accuracies)
        for cid, loss, accuracy in step_data:
            self.next_step_clients[cid] = (loss, accuracy)
        
        return primes.ServerReply(status="OK")


    # function to gather server's version of client loss
    def getServerClientLoss(self, request: primes.lossAndAccuracyRequest, context):
        # get current round of loss
        print("getServerClientLoss")
        step_data = zip(request.cids, request.losses, request.accuracies)
        for cid, loss, accuracy in step_data:
            if cid in self.server_clients:
                self.server_clients[cid].append((loss, accuracy))
            else:
                self.server_clients[cid] = [(loss, accuracy)]
        
        print(self.server_clients)
        return primes.ServerReply(status="OK")


if __name__ == '__main__':
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    rpc.add_PrimesServicer_to_server(PrimesServicer(), server)
    print('Starting server. Listening on port 12345.')
    server.add_insecure_port('127.0.0.1:12345')
    server.start()

    server.wait_for_termination()