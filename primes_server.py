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

        # locks for server_clients and next_step_clients
        self.server_clients_lock = threading.Lock()
        self.next_step_clients_lock = threading.Lock()

        # TODO 
        self.payments = {}


    # function to gather current round of loss
    def getNextStepLoss(self, request: primes.lossAndAccuracyRequest, context):
        # get current round of loss
        step_data = zip(request.cids, request.losses, request.accuracies)
        print("getNextStepLoss")
        for cid, loss, accuracy in step_data:
            if cid in self.next_step_clients:
                self.next_step_clients[cid].append((loss, accuracy))
            else:
                self.next_step_clients[cid] = [(loss, accuracy)]
            
            print(f"cid: {cid}, loss: {loss}, accuracy: {accuracy}")
            
        return primes.ServerReply(status="OK")


    # function to gather server's version of client loss
    def getServerClientLoss(self, request: primes.lossAndAccuracyRequest, context):
        # get current round of loss
        # print("getServerClientLoss")
        step_data = zip(request.cids, request.losses, request.accuracies)
        for cid, loss, accuracy in step_data:
            if cid in self.server_clients:
                self.server_clients[cid].append((loss, accuracy))
            else:
                self.server_clients[cid] = [(loss, accuracy)]

            # print without printing \n

            print(cid, loss, accuracy, end=" ")
        
        return primes.ServerReply(status="OK")
    

    # config fit: get next step's clients
    def getNextClients(self, request: primes.nextClientsRequest, context):
        k = request.k
        ranked_clients = []

        for cid in self.next_step_clients:
            # selection is 100% based on next step loss
            key = self.next_step_clients[cid][-1][0]
            ranked_clients.append((cid, key))

            """client payment function"""
            # # what if client hasn't been selected yet? 
            # if cid in self.next_step_clients and cid in self.server_clients:
            #     key = (WEIGHTS["NEXT_STEP"] * self.next_step_clients[cid][-1][0] + 
            #            WEIGHTS["SERVER_LOSS"] *  self.server_clients[cid][-1][0])
            # elif cid in self.next_step_clients:
            #     key = (WEIGHTS["NEXT_STEP"] * self.next_step_clients[cid][-1][0] + 
            #            WEIGHTS["SERVER_LOSS"] *  avg_server_loss)

        ranked_clients = sorted(ranked_clients, key=lambda client: client[1])
        print(f"ranked_clients: {ranked_clients}")
        print("_______________________")
        print()
        
        ranked_cids = [cid for (cid, _weight) in ranked_clients]

        selected_cids = ranked_cids[:k]
        return primes.nextClientsReply(cids=selected_cids)


if __name__ == '__main__':
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    rpc.add_PrimesServicer_to_server(PrimesServicer(), server)
    print('Starting server. Listening on port 12345.')
    # server.add_insecure_port('172.31.31.180:12345')
    server.add_insecure_port('127.0.0.1:12345')
    server.start()

    server.wait_for_termination()