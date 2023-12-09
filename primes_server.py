from concurrent import futures
import time

import grpc
import primes_pb2 as primes
import primes_pb2_grpc as rpc

import torch
import flwr as fl
import threading

from FL.model import Net, train, test

WEIGHTS = {"NEXT_STEP":0.8,
           "SERVER_LOSS": 0.2}
K = 10

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
    
    def getNextClients(self, request: primes.nextClientsRequest, context):
        ranked_clients = []
        for cid in self.server_clients:
            ranked_clients.append((cid,WEIGHTS["NEXT_STEP"]*self.next_step_clients[cid] + WEIGHTS["SERVER_LOSS"]*self.server_clients[cid]))
        ranked_client_tuples = sorted(ranked_clients, key=lambda client: client[1])
        ranked_clients = [cid for (cid, weight) in ranked_client_tuples]
        return primes.nextClientsReply(cids=ranked_clients)



if __name__ == '__main__':
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    rpc.add_PrimesServicer_to_server(PrimesServicer(), server)
    print('Starting server. Listening on port 12345.')
    server.add_insecure_port('172.31.31.180:12345')
    server.start()

    server.wait_for_termination()