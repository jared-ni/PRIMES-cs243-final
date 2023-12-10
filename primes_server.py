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
        print("getNextStepLoss")
        for cid, loss, accuracy in step_data:
            if cid in self.next_step_clients:
                self.next_step_clients[cid].append((loss, accuracy))
            else:
                self.next_step_clients[cid] = [(loss, accuracy)]
            print(cid, loss, accuracy, end=" ")
            
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

            # print without printing \n

            print(cid, loss, accuracy, end=" ")
        
        return primes.ServerReply(status="OK")
    
    def getNextClients(self, request: primes.nextClientsRequest, context):
        k = request.k
        ranked_clients = []
        print("getNextClients")
        print(self.next_step_clients)
        print(":::")
        print(self.server_clients)
        print("______")

        # average client server loss

        avg_server_loss = sum([self.server_clients[cid][-1][0] for cid in self.server_clients]) / len(self.server_clients)

        for cid in self.server_clients:
            print("cid", cid)
            print("self.next_step_clients[cid]", self.next_step_clients[cid])
            print("self.server_clients[cid]", self.server_clients[cid])

            print("1")

            # what if client hasn't been selected yet? 
            if cid in self.next_step_clients and cid in self.server_clients:
                key = (WEIGHTS["NEXT_STEP"] * self.next_step_clients[cid][-1][0] + 
                       WEIGHTS["SERVER_LOSS"] *  self.server_clients[cid][-1][0])
            elif cid in self.next_step_clients:
                key = (WEIGHTS["NEXT_STEP"] * self.next_step_clients[cid][-1][0] + 
                       WEIGHTS["SERVER_LOSS"] *  avg_server_loss)
            
            print("2) key", key)

            ranked_clients.append((cid, key))

        print("3) ranked_clients", ranked_clients)
        ranked_clients = sorted(ranked_clients, key=lambda client: client[1])
        print("4) ranked_clients", ranked_clients)
        ranked_cids = [cid for (cid, _weight) in ranked_clients]

        selected_cids = ranked_cids[:k]

        print("5) selected_cids", selected_cids)

        print("getNextClients")
        print(primes.nextClientsReply(cids=selected_cids))
        return primes.nextClientsReply(cids=selected_cids)


if __name__ == '__main__':
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    rpc.add_PrimesServicer_to_server(PrimesServicer(), server)
    print('Starting server. Listening on port 12345.')
    # server.add_insecure_port('172.31.31.180:12345')
    server.add_insecure_port('127.0.0.1:12345')
    server.start()

    server.wait_for_termination()