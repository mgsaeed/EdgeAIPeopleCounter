#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 
 NOTES: The initial code in this file is adapted from 
 https://github.com/intel-iot-devkit/people-counter-python  
 
"""

import os
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore


class Network:
    def __init__(self):
        ### TODO: Initialize any class variables desired ###
        self.net = None
        ie = None
        self.input_blob = None
        self.out_blob = None
        self.net_plugin = None
        self.infer_request_handle = None

    def load_model(self, model, device, input_size, output_size, num_requests, cpu_extension):
        ### TODO: Load the model ###
        xml_file = model
        bin_file = os.path.splitext(xml_file)[0] + ".bin"
        
        log.info("Creating Inference Engine...")
        inference_engine = IECore()
        if cpu_extension and 'CPU' in device:
            inference_engine.add_extension(cpu_extension, "CPU")
                   
        # Read IR
        log.info("Reading IR...")
        self.net = IENetwork(model=xml_file, weights=bin_file)
        
        ### TODO: Check for supported layers ###
        if "CPU" in device:
            supported_layers = inference_engine.query_network(self.net, "CPU")
            not_supported_layers = [l for l in self.net.layers.keys() if l not in supported_layers]
            if len(not_supported_layers) != 0:
                sys.exit(1)
            
        if num_requests == 0:
            self.net_plugin = inference_engine.load_network(network=self.net, device_name=device)
        else:
            self.net_plugin = inference_engine.load_network(network=self.net, device_name=device, num_requests=num_requests)

        self.input_blob = next(iter(self.net.inputs))
        self.out_blob = next(iter(self.net.outputs))
        assert len(self.net.inputs.keys()) == input_size, "Supports only {} input topologies".format(len(self.net.inputs))
        assert len(self.net.outputs) == output_size, "Supports only {} output topologies".format(len(self.net.outputs))
        ### TODO: Return the loaded inference plugin ###
        return inference_engine, self.get_input_shape()

    def get_input_shape(self):
        ### TODO: Return the shape of the input layer ###
        return self.net.inputs[self.input_blob].shape

    def exec_net(self, request_id, frame):
        ### TODO: Start an asynchronous request ###
        self.infer_request_handle = self.net_plugin.start_async(request_id=request_id, inputs={self.input_blob: frame})
        ### TODO: Return any necessary information ###
        return self.net_plugin

    def wait(self, request_id):
        ### TODO: Wait for the request to be complete. ###
        infer_status = self.net_plugin.requests[request_id].wait(-1)
        ### TODO: Return any necessary information ###
        return infer_status

    def get_output(self, request_id, output=None):
        ### TODO: Extract and return the output results
        if output:
            res = self.infer_request_handle.outputs[output]
        else:
            res = self.net_plugin.requests[request_id].outputs[self.out_blob]
        return res

    def clean(self):
        del self.net_plugin
        del self.net