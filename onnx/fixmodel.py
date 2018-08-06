"""onnx model fixer

This utility will fix up common model errors.

1. Add a domain to the model object if there isn't one.
2. Fix names of nodes, graphs, values, etc. to conform to the standard.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import onnx
import google.protobuf.message
from onnx import TensorProto, AttributeProto, ValueInfoProto, TensorShapeProto, \
    NodeProto, ModelProto, GraphProto, OperatorSetIdProto, TypeProto, IR_VERSION
from onnx import defs
from onnx.defs import OpSchema
from typing import Text, Sequence, Any, Optional, Dict, Union, TypeVar, Callable, Tuple, List, cast
import onnx.shape_inference

import argparse

modified = False


def isempty(str):  # type: (Text) -> bool
    return str is None or str == ''


def replaceInvalidChars(name):  # type: (Text) -> Text

    global modified

    if name == '':
        return name

    n = name

    if name[0].isnumeric():
        n = "_" + n
    n = n.replace("/", "_").replace("-", "_")

    if n != name:
        modified = True

    return n


def examine_tensor(tensor):  # type: (TensorProto) -> None
    if not isempty(tensor.name):
        tensor.name = replaceInvalidChars(tensor.name)


def examine_attribute(attr):  # type: (AttributeProto) -> None

    # Note: we don't mess with the attribute name since it corresponds to a
    # attribute formal parameter name on the node operator,
    # so we can't change it.

    if attr.type == onnx.AttributeProto.GRAPH:
        examine_graph(attr.g)
    elif attr.type == onnx.AttributeProto.GRAPHS:
        for graph in attr.graphs:
            examine_graph(graph)
    elif attr.type == onnx.AttributeProto.TENSOR:
        examine_tensor(attr.t)
    elif attr.type == onnx.AttributeProto.TENSORS:
        for tensor in attr.tensors:
            examine_tensor(tensor)


def examine_node(node):  # type: (NodeProto) -> None

    if not isempty(node.name):
        node.name = replaceInvalidChars(node.name)

    for vi in range(len(node.input)):
        if not isempty(node.input[vi]):
            node.input[vi] = replaceInvalidChars(node.input[vi])
    for vi in range(len(node.output)):
        if not isempty(node.output[vi]):
            node.output[vi] = replaceInvalidChars(node.output[vi])

    for attr in node.attribute:
        examine_attribute(attr)


def examine_graph(graph):  # type: (GraphProto) -> None

    graph.name = replaceInvalidChars(graph.name)

    for node in graph.node:
        examine_node(node)

    for init in graph.initializer:
        if not isempty(init.name):
            init.name = replaceInvalidChars(init.name)

    for vinfo in graph.value_info:
        if not isempty(vinfo.name):
            vinfo.name = replaceInvalidChars(vinfo.name)

    for vinfo in graph.input:
        if not isempty(vinfo.name):
            vinfo.name = replaceInvalidChars(vinfo.name)

    for vinfo in graph.output:
        if not isempty(vinfo.name):
            vinfo.name = replaceInvalidChars(vinfo.name)


def examine_model(model, domain):  # type: (ModelProto, Text) -> bool

    if not isempty(model.domain):
        print("Domain: " + model.domain)
    if not isempty(model.producer_name):
        print("Producer name: " + model.producer_name)
    if not isempty(model.producer_version):
        print("Producer version: " + model.producer_version)
    for entry in model.metadata_props:
        print(entry.key + ': ' + entry.value)

    global modified

    modified = False

    if isempty(model.domain):
        model.domain = domain
        modified = True

    examine_graph(model.graph)

    return modified


def main():  # type: () -> None
    parser = argparse.ArgumentParser(
        description='Generates .proto file variations from .in.proto')
    parser.add_argument('-d', '--domain', default='org.tempuri.onnx',
                        help='domain name to use if missing in the model'
                        ' (default: %(default)s)')
    parser.add_argument('-o', '--output',
                        default=os.path.realpath(os.path.curdir),
                        help='output directory (default: %(default)s)')
    parser.add_argument('files', nargs='*',
                        help='list of ONNX files ')
    args = parser.parse_args()

    for file in args.files:
        m = onnx.load(file)
        print('\n==== Examining ' + file + ' ====')
        prefix = args.output + os.path.sep
        outpath = prefix + file.replace(".onnx", ".new.onnx")
        if examine_model(m, args.domain):
            onnx.save(m, outpath)
            print('Wrote modified file to: ' + outpath)
        else:
            print('No modifications were made.')

    print('\n')


if __name__ == '__main__':
    main()
