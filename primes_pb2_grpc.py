# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import primes_pb2 as primes__pb2


class PrimesStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.getNextStepLoss = channel.unary_unary(
                '/helper.Primes/getNextStepLoss',
                request_serializer=primes__pb2.lossAndAccuracyRequest.SerializeToString,
                response_deserializer=primes__pb2.ServerReply.FromString,
                )
        self.getServerClientLoss = channel.unary_unary(
                '/helper.Primes/getServerClientLoss',
                request_serializer=primes__pb2.lossAndAccuracyRequest.SerializeToString,
                response_deserializer=primes__pb2.ServerReply.FromString,
                )


class PrimesServicer(object):
    """Missing associated documentation comment in .proto file."""

    def getNextStepLoss(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def getServerClientLoss(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_PrimesServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'getNextStepLoss': grpc.unary_unary_rpc_method_handler(
                    servicer.getNextStepLoss,
                    request_deserializer=primes__pb2.lossAndAccuracyRequest.FromString,
                    response_serializer=primes__pb2.ServerReply.SerializeToString,
            ),
            'getServerClientLoss': grpc.unary_unary_rpc_method_handler(
                    servicer.getServerClientLoss,
                    request_deserializer=primes__pb2.lossAndAccuracyRequest.FromString,
                    response_serializer=primes__pb2.ServerReply.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'helper.Primes', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class Primes(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def getNextStepLoss(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/helper.Primes/getNextStepLoss',
            primes__pb2.lossAndAccuracyRequest.SerializeToString,
            primes__pb2.ServerReply.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def getServerClientLoss(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/helper.Primes/getServerClientLoss',
            primes__pb2.lossAndAccuracyRequest.SerializeToString,
            primes__pb2.ServerReply.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
