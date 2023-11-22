from concurrent import futures
import time

import grpc
import helper_pb2_grpc as rpc
import helper_pb2 as helper

import torch
import flwr as fl

from flwr.common import NDArrays, Scalar
from FL.model import Net, train, test

class PrimesServicer(rpc.PrimesServicer):

    def __init__(self, num_classes):
        self.model = Net(num_classes)


    # def TestAccuracy(self, request: helper.Empty, context):
    #     print("TestAccuracy")
        
    #     return helper.AccuracyResponse(accuracy=0.5)
    
    def TestAccuracy(self, request: NDArrays):
        
        print("Test Accuracy: ")
        print(request)
        
        return helper.AccuracyResponse(accuracy=0.5)

if __name__ == '__main__':
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    rpc.add_PrimesServicer_to_server(PrimesServicer(), server)
    print('Starting server. Listening on port 50051.')
    server.add_insecure_port('[::]:50051')
    server.start()

    server.wait_for_termination()