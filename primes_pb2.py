# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: primes.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0cprimes.proto\x12\x06helper\"\x07\n\x05\x45mpty\"-\n\x0f\x41\x63\x63uracyRequest\x12\x1a\n\x12ndarray_parameters\x18\x01 \x01(\x0c\"2\n\x10\x41\x63\x63uracyResponse\x12\x10\n\x08\x61\x63\x63uracy\x18\x01 \x01(\x02\x12\x0c\n\x04loss\x18\x02 \x01(\x02\"J\n\x16lossAndAccuracyRequest\x12\x0c\n\x04\x63ids\x18\x01 \x03(\t\x12\x0e\n\x06losses\x18\x02 \x03(\x02\x12\x12\n\naccuracies\x18\x03 \x03(\x02\"\x1f\n\x12nextClientsRequest\x12\t\n\x01k\x18\x01 \x01(\x05\" \n\x10nextClientsReply\x12\x0c\n\x04\x63ids\x18\x01 \x03(\t\"\x1d\n\x0bServerReply\x12\x0e\n\x06status\x18\x01 \x01(\t2\xe4\x01\n\x06Primes\x12\x46\n\x0fgetNextStepLoss\x12\x1e.helper.lossAndAccuracyRequest\x1a\x13.helper.ServerReply\x12J\n\x13getServerClientLoss\x12\x1e.helper.lossAndAccuracyRequest\x1a\x13.helper.ServerReply\x12\x46\n\x0egetNextClients\x12\x1a.helper.nextClientsRequest\x1a\x18.helper.nextClientsReplyb\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'primes_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _EMPTY._serialized_start=24
  _EMPTY._serialized_end=31
  _ACCURACYREQUEST._serialized_start=33
  _ACCURACYREQUEST._serialized_end=78
  _ACCURACYRESPONSE._serialized_start=80
  _ACCURACYRESPONSE._serialized_end=130
  _LOSSANDACCURACYREQUEST._serialized_start=132
  _LOSSANDACCURACYREQUEST._serialized_end=206
  _NEXTCLIENTSREQUEST._serialized_start=208
  _NEXTCLIENTSREQUEST._serialized_end=228
  _NEXTCLIENTSREPLY._serialized_start=230
  _NEXTCLIENTSREPLY._serialized_end=262
  _SERVERREPLY._serialized_start=264
  _SERVERREPLY._serialized_end=293
  _PRIMES._serialized_start=296
  _PRIMES._serialized_end=452
# @@protoc_insertion_point(module_scope)
